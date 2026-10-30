#!/usr/bin/env python3
"""Compare immutable control, fallback, activation reuse and tensor CUDA graphs."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
KERNEL_VARIANTS = ['q4-w1', 'q4-w2', 'q6-w1', 'q6-w2', 'both-w1', 'both-w2', 'combined-w1']
VARIANTS = ['fallback', 'reuse-q8', 'graphs', 'split-output', 'split-output-reuse', *KERNEL_VARIANTS,
            *['output-' + name for name in KERNEL_VARIANTS]]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--final', action='store_true', help='Five excluded warmups and 20 balanced measured samples at three depths')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=['fallback', 'reuse-q8', 'graphs'])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    configurations = ['baseline', *a.variants]
    for depth in ([0, 2048, 8192] if a.final else [2048]):
        for block, order in enumerate([configurations, list(reversed(configurations))]):
            for name in order:
                source = ROOT / '.deps' / ('llama.cpp-speed-control' if name == 'baseline' else 'llama.cpp-port')
                command = [sys.executable, str(ROOT / 'tools/port-benchmark.py'),
                           '--source', str(source), '--binary', str(source / 'build-port/bin/llama-bench'),
                           '--out', str(a.out / f'd{depth}-block{block}-{name}'),
                           '--gpu', 'both', '--split-mode', 'tensor', '--allreduce', 'nccl',
                           '--nccl-proof', '--offload-proof', '--proof-timeout', '45',
                           '--flash-attn', 'on', '--depth', str(depth),
                           '--prompt', '512' if a.final else '0', '--gen', '128',
                           '--reps', '10' if a.final else '3', '--warmup-samples', '5' if a.final else '2']
                if name == 'reuse-q8':
                    command += ['--reuse-q8']
                if name.startswith('split-output'):
                    command += ['--split-tied-output']
                    if name.endswith('-reuse'):
                        command += ['--reuse-q8']
                kernel_name = name.removeprefix('output-')
                if name.startswith('output-'):
                    command += ['--split-tied-output', '--reuse-q8']
                if kernel_name.startswith(('q4-', 'q6-', 'both-', 'combined-')):
                    command += ['--tensor-kernels']
                    wpr = '1' if kernel_name.endswith('w1') else '2'
                    if kernel_name.startswith(('q4-', 'both-', 'combined-')):
                        command += ['--q4k', '--q4k-wpr', wpr, '--q4k-wpb', '8']
                    if kernel_name.startswith(('q6-', 'both-', 'combined-')):
                        command += ['--q6k', '--q6k-wpr', wpr, '--q6k-wpb', '8']
                    if kernel_name.startswith('combined-') and not name.startswith('output-'):
                        command += ['--reuse-q8']
                if name == 'graphs':
                    command += ['--graphs']
                print('Executing', depth, block, name, flush=True)
                subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/analyze-port-results.py'), str(a.out),
                    '--out', str(a.out / 'report.json')], check=True)

if __name__ == '__main__':
    main()
