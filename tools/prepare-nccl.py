#!/usr/bin/env python3
"""Localize NCCL compilation and keep Pascal collective arithmetic in FP32."""
import argparse
from pathlib import Path
import subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path)
a=p.parse_args()
pin='169e4a7ff201e666c2325bcd35afb64573e39d09'
if subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()!=pin: raise SystemExit('Unexpected runtime source')
cmake=a.source/'ggml/src/ggml-cuda/CMakeLists.txt'
s=cmake.read_text()
old='            add_compile_definitions(GGML_USE_NCCL)\n            target_link_libraries(ggml-cuda PRIVATE NCCL::NCCL)'
new='''            # NCCL is used only by ggml-cuda.cu; keep unrelated kernel compilation unchanged.
            set_source_files_properties(ggml-cuda.cu PROPERTIES
                COMPILE_DEFINITIONS GGML_USE_NCCL INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
            target_link_libraries(ggml-cuda PRIVATE "${NCCL_LIBRARY}")'''
if new not in s:
 if s.count(old)!=1: raise SystemExit('Unexpected NCCL CMake site')
 cmake.write_text(s.replace(old,new))
cu=a.source/'ggml/src/ggml-cuda/ggml-cuda.cu'
s=cu.read_text()
marker='    // For small tensors, simply reduce them as FP32.'
start=s.index(marker)
newcond='    if (ggml_cuda_info().devices[static_cast<ggml_backend_cuda_context *>(comm_ctx->backends[0]->context)->device].cc == 610 || (n_backends <= 2'
if newcond not in s:
 cond=s.index('    if ((n_backends <= 2',start)
 s=s[:cond]+s[cond:].replace('    if ((n_backends <= 2',newcond,1)
 cu.write_text(s)
print('Prepared optional NCCL; Pascal uses FP32 for all collective sizes')
