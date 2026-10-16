#!/usr/bin/env python3
"""Install opt-in, graph-local Q8 activation reuse into the pinned CUDA source."""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PIN = '169e4a7ff201e666c2325bcd35afb64573e39d09'
DECLARATION = '''
// Legacy AI: ownership and invalidation are scoped to one graph evaluation.
class legacy_ai_q8_reuse_scope {
    void * state = nullptr;
    void * previous = nullptr;
public:
    legacy_ai_q8_reuse_scope(ggml_backend_cuda_context & ctx, bool compatible);
    ~legacy_ai_q8_reuse_scope();
    void invalidate_write(const ggml_tensor * node);
    legacy_ai_q8_reuse_scope(const legacy_ai_q8_reuse_scope &) = delete;
    legacy_ai_q8_reuse_scope & operator=(const legacy_ai_q8_reuse_scope &) = delete;
};
'''
QUANTIZATION_OLD = '''    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }'''
QUANTIZATION_NEW = '''    // Legacy AI: only an unchanged, contiguous Qwen3 single-token input can
    // reuse Q8 bytes. Its owner invalidates writes and ends at graph boundaries.
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    auto * cache = legacy_ai_active_q8_reuse;
    const bool reuse_eligible = cache && cache->ctx == &ctx && cache->eligible(src0, src1, ids, stream);
    const bool reuse_hit = reuse_eligible && cache->input == src1 &&
                          cache->input_data == src1_d && cache->stream == stream;
    char * quantized;
    if (reuse_eligible) {
        quantized = cache->quantized.get();
    } else {
        quantized = src1_q8_1.alloc(ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    }
    if (!reuse_hit) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, quantized, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        if (reuse_eligible) {
            cache->input = src1;
            cache->input_data = src1_d;
            cache->stream = stream;
            ++cache->quantizations;
        }
    } else {
        ++cache->hits;
        static std::atomic_flag reported = ATOMIC_FLAG_INIT;
        if (!reported.test_and_set()) {
            GGML_LOG_INFO("legacy_ai: graph-local Q8 activation reuse active\\n");
        }
    }'''


def replace_once(data, old, new):
    if new in data:
        return data
    if data.count(old) != 1:
        raise SystemExit('Unexpected pinned source; refusing activation reuse patch')
    return data.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    a = p.parse_args()
    head = subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != PIN:
        raise SystemExit('Expected pinned source revision')
    cuda = a.source / 'ggml/src/ggml-cuda'
    header = cuda / 'mmvq.cuh'
    data = header.read_text()
    if DECLARATION not in data:
        header.write_text(data + DECLARATION)
    implementation = cuda / 'mmvq.cu'
    data = implementation.read_text()
    data = replace_once(data, '#include <type_traits>', '#include <type_traits>\n#include "legacy-ai-q8-reuse.cuh"')
    data = replace_once(data, QUANTIZATION_OLD, QUANTIZATION_NEW)
    data = replace_once(data, 'src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,',
                        'src0->data, src0->type, quantized, ids_d, fusion_local, dst_d, ne00,')
    implementation.write_text(data)
    graph = cuda / 'ggml-cuda.cu'
    data = graph.read_text()
    old = '    bool graph_evaluated_or_captured = false;'
    new = old + '''
    // Never share across graphs, captures, concurrent branches, devices or streams.
    legacy_ai_q8_reuse_scope q8_reuse(*cuda_ctx, !use_cuda_graph &&
                                    cuda_ctx->stream_context().concurrent_events.empty());'''
    data = replace_once(data, old, new)
    old = '                    i += nodes_to_skip;\n                    continue;'
    new = '''                    for (int j = i; j <= i + nodes_to_skip; ++j) {
                        q8_reuse.invalidate_write(cgraph->nodes[j]);
                    }
''' + old
    data = replace_once(data, old, new)
    data = replace_once(data, '                GGML_ASSERT(ok);\n',
                        '                GGML_ASSERT(ok);\n                q8_reuse.invalidate_write(node);\n')
    graph.write_text(data)
    shutil.copyfile(ROOT / 'kernels/q4k/legacy-ai-q8-reuse.cuh', cuda / 'legacy-ai-q8-reuse.cuh')
    print('Installed opt-in graph-local activation reuse:', a.source)


if __name__ == '__main__':
    main()
