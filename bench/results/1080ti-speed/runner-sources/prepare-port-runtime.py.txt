#!/usr/bin/env python3
"""Apply the narrow opt-in Pascal graph patch to the pinned upstream worktree.

The caller creates a separate git worktree at PIN first. Original upstream
sources remain the control. Refuses other revisions or unexpected source.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

PIN = '169e4a7ff201e666c2325bcd35afb64573e39d09'
OLD = '''        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {'''
NEW = '''        // Legacy AI: explicitly opt in only the qualified Pascal SM 6.1 path.
        // Preserve the architecture guard and all upstream graph compatibility,
        // warmup, recapture, update and explicit-disable checks by default.
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        const char * opt_in = std::getenv("LEGACY_AI_PASCAL_GRAPHS");
        const bool pascal_graphs = cc == 610 && opt_in && std::strcmp(opt_in, "1") == 0;
        if (cc < GGML_CUDA_CC_VOLTA && !pascal_graphs) {'''

KERNEL_GUARD = '''
    // Legacy AI: preserve upstream for all unqualified shapes/layouts/fusions.
    static const bool legacy_q4k = [] {
        const char * value = std::getenv("LEGACY_AI_Q4K");
        return value && std::strcmp(value, "1") == 0;
    }();
    static const bool legacy_q6k = [] {
        const char * value = std::getenv("LEGACY_AI_Q6K");
        return value && std::strcmp(value, "1") == 0;
    }();
    static const bool legacy_tensor_mmvq = [] {
        const char * value = std::getenv("LEGACY_AI_TENSOR_MMVQ");
        return value && std::strcmp(value, "1") == 0;
    }();
    const bool canonical_q4 =
        (ncols_x == 2560 && (nrows_x == 4096 || nrows_x == 1024 || nrows_x == 9728)) ||
        (ncols_x == 4096 && nrows_x == 2560) ||
        (ncols_x == 9728 && nrows_x == 2560) ||
        (legacy_tensor_mmvq && ((ncols_x == 2560 && (nrows_x == 512 || nrows_x == 2048 || nrows_x == 4864)) ||
                              ((ncols_x == 2048 || ncols_x == 4864) && nrows_x == 2560)));
    const bool canonical_q6 =
        (ncols_x == 2560 && (nrows_x == 1024 || nrows_x == 151936)) ||
        (ncols_x == 9728 && nrows_x == 2560) ||
        (legacy_tensor_mmvq && ((ncols_x == 2560 && (nrows_x == 512 || nrows_x == 75968)) ||
                              (ncols_x == 4864 && nrows_x == 2560)));
    const bool supported_type = (legacy_q4k && type_x == GGML_TYPE_Q4_K && canonical_q4) ||
                                (legacy_q6k && type_x == GGML_TYPE_Q6_K && canonical_q6);
    if (supported_type && ncols_dst == 1 && !ids &&
        nchannels_x == 1 && nchannels_y == 1 && nchannels_dst == 1 &&
        nsamples_x == 1 && nsamples_dst == 1 &&
        stride_row_x == ncols_x / QK_K &&
        !fusion.x_bias && !fusion.gate_bias && !fusion.x_scale && !fusion.gate_scale &&
        ggml_cuda_info().devices[ggml_cuda_get_device()].cc == 610) {
        const bool q6 = type_x == GGML_TYPE_Q6_K;
        const char * env_wpr = std::getenv(q6 ? "LEGACY_AI_Q6K_WPR" : "LEGACY_AI_Q4K_WPR");
        const char * env_wpb = std::getenv(q6 ? "LEGACY_AI_Q6K_WPB" : "LEGACY_AI_Q4K_WPB");
        const int wpr = env_wpr ? std::atoi(env_wpr) : 2;
        const int wpb = env_wpb ? std::atoi(env_wpb) : 8;
        const char * sums_env = std::getenv("LEGACY_AI_Q4K_SUMS");
        const bool precompute = !q6 && sums_env && std::strcmp(sums_env, "1") == 0;
        const bool packed = !q6 && packed_weights && (!fusion.gate || packed_gate);
        auto packed_fusion = fusion;
        if (packed) packed_fusion.gate = packed_gate;
        const auto * x = static_cast<const block_q8_1 *>(vy);
        const bool dispatched = q6 ?
            legacy_ai_dispatch_mmvq<GGML_TYPE_Q6_K>(wpr, wpb, vx, x, dst, nrows_x, ncols_x, fusion, stream) :
            legacy_ai_dispatch_mmvq<GGML_TYPE_Q4_K>(wpr, wpb, packed ? packed_weights : vx, x, dst, nrows_x, ncols_x, packed ? packed_fusion : fusion, stream, precompute, packed);
        if (dispatched) {
            static std::atomic_flag reported_q4 = ATOMIC_FLAG_INIT;
            static std::atomic_flag reported_q6 = ATOMIC_FLAG_INIT;
            if (!(q6 ? reported_q6 : reported_q4).test_and_set()) {
                GGML_LOG_INFO("legacy_ai: Q%d_K runtime dispatch active, WPR=%d WPB=%d packed=%d K=%d N=%d tensor=%d\\n", q6 ? 6 : 4, wpr, wpb, packed, ncols_x, nrows_x, legacy_tensor_mmvq);
            }
            return;
        }
    }
'''

PROFILE_SCOPE = '''
// Diagnostic only: event intervals include exposed host launch gaps and each
// operation is synchronized. These samples cannot serve as throughput claims.
struct legacy_ai_profile_scope {
    bool enabled;
    ggml_tensor * node;
    cudaStream_t stream;
    cudaEvent_t start, stop;
    int64_t wall_start;
    legacy_ai_profile_scope(ggml_backend_cuda_context * ctx, ggml_tensor * tensor, bool captured)
        : enabled(!captured && std::getenv("LEGACY_AI_PROFILE") != nullptr), node(tensor) {
        if (!enabled) return;
        stream = ctx->stream();
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        wall_start = ggml_time_us();
        CUDA_CHECK(cudaEventRecord(start, stream));
    }
    ~legacy_ai_profile_scope() {
        if (!enabled) return;
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const auto * src = node->src[0];
        std::fprintf(stderr, "LEGACY_PROFILE,%s,%s,%s,%lld,%lld,%lld,%.6f,%lld\\n",
            node->name, ggml_op_name(node->op), src ? ggml_type_name(src->type) : "none",
            (long long)(src ? src->ne[0] : 0), (long long)(src ? src->ne[1] : 0),
            (long long)node->ne[1], ms, (long long)(ggml_time_us() - wall_start));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
};

'''



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--kernels', action='store_true', help='Integrate kernels preserving the pinned upstream reduction order, opt-in at runtime')
    p.add_argument('--profile', action='store_true', help='Install optional diagnostic per-operation event timing')
    a = p.parse_args()
    head = subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != PIN:
        raise SystemExit(f'Expected pinned revision {PIN}, got {head}')
    path = a.source / 'ggml/src/ggml-cuda/ggml-cuda.cu'
    data = path.read_text()
    if '#include <cstring>' not in data:
        data = data.replace('#include <cstdlib>', '#include <cstdlib>\n#include <cstring>')
    if NEW in data:
        if path.read_text() != data:
            path.write_text(data)
        print('Pascal graph patch already applied')
    else:
        if data.count(OLD) != 1:
            raise SystemExit('Unexpected upstream graph guard; refusing patch')
        instantiate = '            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));'
        if data.count(instantiate) != 1:
            raise SystemExit('Unexpected graph instantiation site; refusing patch')
        report = instantiate + '''
            if (ggml_cuda_info().devices[cuda_ctx->device].cc == 610) {
                GGML_LOG_INFO("legacy_ai: Pascal CUDA graph instantiated on device %d\\n", cuda_ctx->device);
            }'''
        path.write_text(data.replace(OLD, NEW).replace(instantiate, report))
        print('Applied opt-in Pascal CUDA Graph patch:', path)
    if a.kernels:
        mmvq = a.source / 'ggml/src/ggml-cuda/mmvq.cu'
        data = mmvq.read_text()
        if KERNEL_GUARD not in data:
            marker = '    // Legacy AI: preserve upstream for all unqualified shapes/layouts/fusions.'
            if marker in data:
                first = data.index(marker)
                last = data.index('    switch (type_x) {', first)
                data = data[:first] + data[last:]
            anchor = 'static void mul_mat_vec_q_switch_type('
            if data.count(anchor) != 1:
                raise SystemExit('Unexpected MMVQ dispatcher; refusing kernel integration')
            switch = data.index('    switch (type_x) {', data.index(anchor))
            data = data[:switch] + KERNEL_GUARD + '\n' + data[switch:]
            if '#include "legacy-ai-mmvq.cuh"' not in data:
                data = data.replace('#include "vecdotq.cuh"', '#include "vecdotq.cuh"\n#include "legacy-ai-mmvq.cuh"\n#include <cstring>\n#include <atomic>')
            data = data.replace('#include "legacy-ai-q4k.cuh"', '#include "legacy-ai-mmvq.cuh"')
            mmvq.write_text(data)
        # Two optional pointers are passed only by the full-tensor runtime path.
        data = mmvq.read_text()
        signature = '        const int ids_stride, cudaStream_t stream) {'
        start = data.index('static void mul_mat_vec_q_switch_type(')
        end = data.index('{', start)
        if 'packed_weights' not in data[start:end]:
            before = data[:start]
            tail = data[start:]
            if tail.count(signature) != 1:
                raise SystemExit('Unexpected MMVQ signature')
            tail = tail.replace(signature, '        const int ids_stride, cudaStream_t stream, const void * packed_weights = nullptr, const void * packed_gate = nullptr) {', 1)
            data = before + tail
            mmvq.write_text(data)
        root = Path(__file__).resolve().parent.parent
        for name in ['legacy-ai-mmvq.cuh', 'legacy-ai-packed-q4.cuh']:
            source_header, destination = root / 'kernels/q4k' / name, mmvq.parent / name
            if not destination.exists() or destination.read_bytes() != source_header.read_bytes():
                shutil.copyfile(source_header, destination)
        print('Integrated opt-in exact-order kernels with upstream fallback:', mmvq)
    if a.profile:
        data = path.read_text()
        anchor = 'static void ggml_cuda_graph_evaluate_and_capture('
        if PROFILE_SCOPE not in data:
            if data.count(anchor) != 1:
                raise SystemExit('Unexpected graph evaluator; refusing profile instrumentation')
            data = data.replace(anchor, PROFILE_SCOPE + anchor)
            site = '                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);'
            if data.count(site) != 1:
                raise SystemExit('Unexpected fusion site; refusing profile instrumentation')
            data = data.replace(site, '                legacy_ai_profile_scope profile(cuda_ctx, node, use_cuda_graph);\n' + site)
            path.write_text(data)
        print('Installed opt-in diagnostic event timing:', path)


if __name__ == '__main__':
    main()
