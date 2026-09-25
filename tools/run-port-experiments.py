#!/usr/bin/env python3
"""Run serial screening or balanced matched inference measurements on BigBoy."""
import argparse
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parent.parent
BASE_SOURCE = Path('/home/taly/legacy-ai/.deps/llama.cpp')
PORT_SOURCE = ROOT / '.deps/llama.cpp-port'

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--stage', choices=['screen', 'sums', 'packed', 'attention', 'exact-attention', 'value', 'flash-graphs', 'final'], default='screen')
p.add_argument('--gpu', choices=['primary', 'secondary', 'both'], default='primary')
p.add_argument('--attention-v', action='store_true')
p.add_argument('--include-flash', action='store_true', help='Include separately quality-qualified upstream Flash Attention in final measurements')
p.add_argument('--include-graphs', action='store_true', help='Include graph-only final measurements; earlier screens found no gain')
p.add_argument('--skip-exact', action='store_true', help='Final comparison of baseline and Flash Attention only')
p.add_argument('--attention-v-rows', choices=[2, 4], type=int, default=2)
p.add_argument('--attention-rows', choices=[4, 8], type=int, default=4)
a = p.parse_args()
if socket.gethostname() != 'BigBoy':
    raise SystemExit('Run on BigBoy')
if a.out.exists():
    raise SystemExit('Refusing to overwrite experiment results')
a.out.mkdir(parents=True)
base = ('baseline', BASE_SOURCE, 'build-default', [])
graphs = ('graphs', PORT_SOURCE, 'build-port', ['--graphs'])
optimized = (f'exact-attention-rows{a.attention_rows}', PORT_SOURCE, 'build-port', ['--attention', '--attention-rows', str(a.attention_rows)])
if a.attention_v:
    optimized = (optimized[0] + '-value', *optimized[1:3], optimized[3] + ['--attention-v', '--attention-v-rows', str(a.attention_v_rows)])
if a.stage == 'flash-graphs':
    configs = [('flash-attention', PORT_SOURCE, 'build-port', ['--flash-attn','on']),
               ('flash-attention-graphs', PORT_SOURCE, 'build-port', ['--flash-attn','on','--graphs'])]
    jobs = [(depth,0,config,3) for depth in [0,2048] for config in configs]
elif a.stage == 'value':
    configs = [base, ('exact-qk', PORT_SOURCE, 'build-port', ['--attention'])]
    for rows in [2, 4]:
        configs += [(f'exact-value-rows{rows}', PORT_SOURCE, 'build-port', ['--attention-v', '--attention-v-rows', str(rows)]),
                    (f'exact-qk-value-rows{rows}', PORT_SOURCE, 'build-port', ['--attention', '--attention-v', '--attention-v-rows', str(rows)])]
    jobs = [(depth, 0, config, 3) for depth in [0, 2048] for config in configs]
elif a.stage == 'packed':
    configs = [base]
    configs += [(f'q4-packed-wpr{wpr}-wpb{wpb}', PORT_SOURCE, 'build-port',
                 ['--q4k', '--prepack', '--q4k-wpr', str(wpr), '--q4k-wpb', str(wpb)])
                for wpr, wpb in [(1, 4), (1, 8), (2, 4), (2, 8), (4, 4)]]
    jobs = [(0, 0, config, 3) for config in configs]
elif a.stage == 'exact-attention':
    configs = [base]
    configs += [(f'exact-attention-rows{rows}', PORT_SOURCE, 'build-port', ['--attention', '--attention-rows', str(rows)]) for rows in [4, 8]]
    jobs = [(depth, 0, config, 3) for depth in [0, 2048] for config in configs]
elif a.stage == 'attention':
    configs = [base, ('flash-attention', BASE_SOURCE, 'build-default', ['--flash-attn', 'on'])]
    jobs = [(depth, 0, config, 3) for depth in [0, 2048] for config in configs]
elif a.stage in ['screen', 'sums', 'packed', 'attention', 'exact-attention', 'value']:
    configs = [base, graphs]
    configs += [(f'{"q4-sums" if a.stage == "sums" else "q4q6"}-wpr{wpr}-wpb{wpb}', PORT_SOURCE, 'build-port',
                 ['--q4k', '--q4k-sums' if a.stage == 'sums' else '--q6k', '--q4k-wpr', str(wpr), '--q6k-wpr', str(wpr),
                  '--q4k-wpb', str(wpb), '--q6k-wpb', str(wpb)])
                for wpr in [1, 2, 4] for wpb in [4, 8, 16]]
    jobs = [(0, 0, config, 3) for config in configs]
else:
    # Forward/reverse order: twenty measured samples/config/depth.
    configs = [base] if a.skip_exact else [base, optimized]
    if a.include_graphs:
        configs += [graphs]
    if a.include_flash:
        configs += [('flash-attention', PORT_SOURCE, 'build-port', ['--flash-attn', 'on'])]
    jobs = [(depth, block, config, 10) for depth in [0, 2048]
            for block, order in enumerate([configs, list(reversed(configs))]) for config in order]
for depth, block, (label, source, build, extra), reps in jobs:
    destination = a.out / f'd{depth}-block{block}-{label}'
    command = ['python3', str(ROOT / 'tools/port-benchmark.py'), '--source', str(source),
               '--binary', str(source / build / 'bin/llama-bench'), '--out', str(destination),
               '--gpu', a.gpu, '--depth', str(depth), '--reps', str(reps)] + extra
    if a.stage in ['screen', 'sums', 'packed', 'attention', 'exact-attention', 'value', 'flash-graphs']:
        command += ['--prompt', '0']
    if a.stage == 'final':
        command += ['--warmup-samples', '5', '--offload-proof']
    print('Executing', destination.name, flush=True)
    subprocess.run(command, check=True)
