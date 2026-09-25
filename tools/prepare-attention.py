#!/usr/bin/env python3
"""Integrate opt-in, exact-order Qwen3-4B attention matvec on SM 6.1."""
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
path = a.source / 'ggml/src/ggml-cuda/mmvf.cu'
data = path.read_text()
include = '#include "legacy-ai-attention.cuh"'
if include not in data:
    anchor = '#include "convert.cuh"'
    data = data.replace(anchor, anchor + '\n#include <cstdlib>\n#include <cstring>\n#include <atomic>\n' + include)
    start = data.index('void ggml_cuda_mul_mat_vec_f(')
    site = data.index('    GGML_TENSOR_BINARY_OP_LOCALS;', start)
    data = data[:site] + '    if (legacy_ai_attention_dispatch(ctx, src0, src1, ids, dst, fusion)) return;\n\n' + data[site:]
    path.write_text(data)
root = Path(__file__).resolve().parent.parent
source_header, destination = root / 'kernels/q4k/legacy-ai-attention.cuh', path.parent / 'legacy-ai-attention.cuh'
if not destination.exists() or destination.read_bytes() != source_header.read_bytes():
    shutil.copyfile(source_header, destination)
print('Integrated opt-in exact-order attention kernel')
