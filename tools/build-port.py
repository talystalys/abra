#!/usr/bin/env python3
"""Build an isolated SM 6.1 runtime from the existing pinned control checkout."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PIN = '169e4a7ff201e666c2325bcd35afb64573e39d09'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream', type=Path, default=ROOT / '.deps/llama.cpp')
p.add_argument('--overlay', type=Path, default=ROOT / '.cuda_compat/include')
p.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda-12.9'))
p.add_argument('--jobs', type=int, default=8)
p.add_argument('--profile', action='store_true')
p.add_argument('--reuse-q8', action='store_true', help='Install optional graph-local activation reuse')
p.add_argument('--split-tied-output', action='store_true', help='Install optional Qwen3 tied output row splitting')
p.add_argument('--nccl-prefix',type=Path,help='Optional locally extracted NCCL prefix; no system installation')
a = p.parse_args()

def run(argv):
    print('Executing:', ' '.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True)

head = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
if head != PIN:
    raise SystemExit(f'Control must be at {PIN}, found {head}')
run(['git', '-C', a.upstream, 'diff', '--exit-code', 'HEAD'])
if not (a.upstream / 'build-default/bin/libllama.so').exists():
    raise SystemExit('Build the original control runtime first; existing libraries are required for independent validation')
if not (a.overlay / 'crt/math_functions.h').exists():
    raise SystemExit('Provide the existing CUDA/glibc overlay with --overlay')
if a.jobs < 1:
    raise SystemExit('--jobs must be positive')
source = ROOT / '.deps/llama.cpp-port'
if not source.exists():
    source.parent.mkdir(parents=True, exist_ok=True)
    run(['git', '-C', a.upstream, 'worktree', 'add', '--detach', source, PIN])
run(['python3', ROOT / 'tools/prepare-port-runtime.py', source, '--kernels'] + (['--profile'] if a.profile else []))
run(['python3', ROOT / 'tools/prepare-packed-weights.py', source])
run(['python3', ROOT / 'tools/prepare-attention.py', source])
if a.reuse_q8:
    run(['python3', ROOT / 'tools/prepare-q8-reuse.py', source])
if a.split_tied_output:
    run(['python3', ROOT / 'tools/prepare-output-split.py', source])
nccl_flags = []
if a.nccl_prefix:
    include = a.nccl_prefix.resolve() / 'usr/include'
    library = a.nccl_prefix.resolve() / 'usr/lib/x86_64-linux-gnu/libnccl.so'
    if not (include/'nccl.h').exists() or not library.exists(): raise SystemExit('NCCL headers/library missing')
    run(['python3',ROOT/'tools/prepare-nccl.py',source])
    nccl_flags = [f'-DNCCL_INCLUDE_DIR={include}',f'-DNCCL_LIBRARY={library}','-DGGML_CUDA_NCCL=ON']
flags = '-I' + str(a.overlay.absolute()) + ' -Wno-deprecated-gpu-targets'
common = [f'-DCMAKE_CUDA_COMPILER={a.cuda}/bin/nvcc', f'-DCMAKE_CUDA_FLAGS={flags}',
          '-DCMAKE_CUDA_ARCHITECTURES=61', '-DCMAKE_BUILD_TYPE=Release']
run(['cmake', '-S', source, '-B', source / 'build-port', '-DGGML_CUDA=ON',
     '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_CURL=OFF', *nccl_flags, *common])
run(['cmake', '--build', source / 'build-port', '--target', 'llama-bench', 'llama-cli', 'llama-completion', 'llama-perplexity', '--parallel', a.jobs])
run(['cmake', '-S', ROOT, '-B', ROOT / 'build-sm61',
     '-DLEGACY_AI_BUILD_Q8_REUSE_TEST=' + ('ON' if a.reuse_q8 else 'OFF'),
     f'-DLLAMA_SOURCE_DIR={a.upstream.resolve()}',
     f'-DLLAMA_LIB_DIR={a.upstream.resolve()}/build-default/bin', *common])
run(['cmake', '--build', ROOT / 'build-sm61', '--target', 'list_cuda_devices',
     'q4k_q8_test', 'runtime_logits', 'lookup_decode', 'runtime_kernel_test', 'attention_kernel_test', 'value_kernel_test',
     *(['q8_reuse_test'] if a.reuse_q8 else []), '--parallel', a.jobs])
