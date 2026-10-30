#!/usr/bin/env python3
"""Add opt-in model-owned Q4_K metadata expansion to the pinned port runtime.

Pack during complete weight upload, keep original weights, release cache with
its GPU buffer, invalidate it on writes, and use upstream when unavailable.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

PIN = '169e4a7ff201e666c2325bcd35afb64573e39d09'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
a = p.parse_args()
if subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
    raise SystemExit('Wrong upstream revision')
directory = a.source / 'ggml/src/ggml-cuda'
path = directory / 'ggml-cuda.cu'
data = path.read_text()
marker = '// Abra model-owned metadata cache.'
if marker not in data:
    include = '#include <cstdlib>'
    data = data.replace(include, include + '\n#include "abra-packed-q4.cuh"')
    data = data.replace('    void * dev_ptr = nullptr;\n    std::string name;', '    void * dev_ptr = nullptr;\n    std::string name;\n    // Abra model-owned metadata cache.\n    std::mutex packed_mutex;\n    std::unordered_map<const void *, void *> packed_q4;')
    destructor = '    ~ggml_backend_cuda_buffer_context() {\n        CUDA_CHECK(cudaFree(dev_ptr));'
    if data.count(destructor) != 1:
        raise SystemExit('Unexpected buffer destructor')
    data = data.replace(destructor, '    ~ggml_backend_cuda_buffer_context() {\n        ggml_cuda_set_device(device);\n        for (auto &entry : packed_q4) CUDA_CHECK(cudaFree(entry.second));\n        CUDA_CHECK(cudaFree(dev_ptr));')
    helper = '''
static void abra_invalidate_q4(ggml_backend_cuda_buffer_context *ctx, const void *address) {
    std::lock_guard<std::mutex> lock(ctx->packed_mutex);
    auto it = ctx->packed_q4.find(address);
    if (it != ctx->packed_q4.end()) {
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaFree(it->second));
        ctx->packed_q4.erase(it);
    }
}

const void *abra_get_packed_q4(const ggml_tensor *tensor) {
    if (!tensor || !tensor->buffer || !ggml_backend_buffer_is_cuda(tensor->buffer)) return nullptr;
    auto *ctx = static_cast<ggml_backend_cuda_buffer_context *>(tensor->buffer->context);
    std::lock_guard<std::mutex> lock(ctx->packed_mutex);
    auto it = ctx->packed_q4.find(tensor->data);
    return it == ctx->packed_q4.end() ? nullptr : it->second;
}

static void abra_prepare_q4(ggml_backend_cuda_buffer_context *ctx, ggml_tensor *tensor, size_t offset, size_t size) {
    const char *opt = std::getenv("ABRA_PREPACK");
    if (!opt || std::strcmp(opt, "1") || tensor->type != GGML_TYPE_Q4_K || offset != 0 ||
        size != ggml_nbytes(tensor) || !ggml_is_contiguous(tensor) || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
        ggml_cuda_info().devices[ctx->device].cc != 610) return;
    const int64_t blocks = ggml_nelements(tensor) / QK_K;
    void *packed = nullptr;
    const size_t packed_bytes = blocks * sizeof(abra_packed_q4);
    if (cudaMalloc(&packed, packed_bytes) != cudaSuccess) {
        cudaGetLastError();
        GGML_LOG_WARN("abra: Q4_K packing allocation failed; retaining upstream weights\\n");
        return;
    }
    abra_pack_q4_kernel<<<(blocks + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        static_cast<const block_q4_K *>(tensor->data), static_cast<abra_packed_q4 *>(packed), blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    std::lock_guard<std::mutex> lock(ctx->packed_mutex);
    ctx->packed_q4[tensor->data] = packed;
    GGML_LOG_INFO("abra: packed %s (%zu bytes)\\n", tensor->name, packed_bytes);
}

'''
    anchor = 'static void * ggml_backend_cuda_buffer_get_base('
    if data.count(anchor) != 1:
        raise SystemExit('Unexpected buffer functions')
    data = data.replace(anchor, helper + anchor)
    for name in ['memset_tensor', 'set_tensor', 'set_tensor_2d']:
        start = data.index('static void ggml_backend_cuda_buffer_' + name + '(')
        device = data.index('    ggml_cuda_set_device(ctx->device);', start)
        data = data[:device] + '    abra_invalidate_q4(ctx, tensor->data);\n' + data[device:]
    start = data.index('static void ggml_backend_cuda_buffer_set_tensor(')
    end = data.index('\n}', start)
    data = data[:end] + '\n    abra_prepare_q4(ctx, tensor, offset, size);' + data[end:]
    anchor = '        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);'
    data = data.replace(anchor, '        abra_invalidate_q4(dst_ctx, dst->data);\n' + anchor)
    start = data.index('static void ggml_backend_cuda_buffer_clear(')
    device = data.index('    ggml_cuda_set_device(ctx->device);', start)
    clear = '    {\n        std::lock_guard<std::mutex> lock(ctx->packed_mutex);\n        ggml_cuda_set_device(ctx->device);\n        for (auto &entry : ctx->packed_q4) CUDA_CHECK(cudaFree(entry.second));\n        ctx->packed_q4.clear();\n    }\n'
    data = data[:device] + clear + data[device:]
    path.write_text(data)
mmvq = directory / 'mmvq.cu'
text = mmvq.read_text()
marker = '    // Abra: obtain immutable packed model weights for decode only.'
if marker not in text:
    anchor = '    mul_mat_vec_q_switch_type(\n        src0->data, src0->type, src1_q8_1.get()'
    if text.count(anchor) != 1:
        raise SystemExit('Unexpected full-tensor MMVQ call')
    get = """    // Abra: obtain immutable packed model weights for decode only.
    static const bool prepack_enabled = [] {
        const char *v = std::getenv("ABRA_PREPACK");
        return v && std::strcmp(v, "1") == 0;
    }();
    const void *packed_weights = prepack_enabled && ncols_dst == 1 ? abra_get_packed_q4(src0) : nullptr;
    const void *packed_gate = prepack_enabled && fusion && fusion->gate ? abra_get_packed_q4(fusion->gate) : nullptr;

"""
    text = text.replace(anchor, get + anchor)
    end = 'ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);'
    if text.count(end) != 1:
        raise SystemExit('Unexpected full-tensor MMVQ arguments')
    text = text.replace(end, end[:-2] + ', packed_weights, packed_gate);')
    mmvq.write_text(text)
common = directory / 'common.cuh'
text = common.read_text()
declaration = 'const void *abra_get_packed_q4(const ggml_tensor *tensor);'
if declaration not in text:
    common.write_text(text + '\n' + declaration + '\n')
root = Path(__file__).resolve().parent.parent
source_header, destination = root / 'kernels/q4k/abra-packed-q4.cuh', directory / 'abra-packed-q4.cuh'
if not destination.exists() or destination.read_bytes() != source_header.read_bytes():
    shutil.copyfile(source_header, destination)
print('Prepared model-owned Q4_K metadata cache')
