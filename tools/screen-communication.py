#!/usr/bin/env python3
"""Balanced exploratory Pascal NCCL placement/protocol screen on BigBoy."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
CASES = [
    ('control', []),
    ('both-cpu', ['--cpu-nodes', '0,1']),
    ('both-interleave', ['--cpu-nodes', '0,1', '--memory-policy', 'interleave']),
    ('ignore-affinity', ['--nccl-option', 'NCCL_IGNORE_CPU_AFFINITY=1']),
    ('ll-one', ['--cpu-nodes', '0,1', '--nccl-option', 'NCCL_PROTO=LL',
                '--nccl-option', 'NCCL_MIN_NCHANNELS=1', '--nccl-option', 'NCCL_MAX_NCHANNELS=1']),
    ('ll-two', ['--cpu-nodes', '0,1', '--nccl-option', 'NCCL_PROTO=LL',
                '--nccl-option', 'NCCL_MIN_NCHANNELS=2', '--nccl-option', 'NCCL_MAX_NCHANNELS=2']),
    ('simple-one', ['--cpu-nodes', '0,1', '--nccl-option', 'NCCL_PROTO=Simple',
                    '--nccl-option', 'NCCL_MIN_NCHANNELS=1', '--nccl-option', 'NCCL_MAX_NCHANNELS=1']),
    ('simple-copy', ['--cpu-nodes', '0,1', '--nccl-option', 'NCCL_PROTO=Simple',
                     '--nccl-option', 'NCCL_MIN_NCHANNELS=1', '--nccl-option', 'NCCL_MAX_NCHANNELS=1',
                     '--nccl-option', 'NCCL_SHM_USE_CUDA_MEMCPY=1', '--nccl-option', 'NCCL_SHM_MEMCPY_MODE=3']),
]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cases', nargs='+', choices=[x[0] for x in CASES])
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--depth', type=int, default=2048)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cases = [x for x in CASES if not a.cases or x[0] in a.cases]
    rows = []
    for block, order in enumerate([cases, list(reversed(cases))]):
        for name, options in order:
            dest = a.out / f'block{block}-{name}'
            command = [sys.executable, str(ROOT / 'tools/port-benchmark.py'),
                       '--source', str(ROOT / '.deps/llama.cpp-port'),
                       '--binary', str(ROOT / '.deps/llama.cpp-port/build-port/bin/llama-bench'),
                       '--out', str(dest), '--gpu', 'both', '--split-mode', 'tensor',
                       '--allreduce', 'nccl', '--nccl-proof', '--offload-proof',
                       '--proof-timeout', '45',
                       '--flash-attn', 'on', '--depth', str(a.depth), '--prompt', '0',
                       '--gen', '128', '--reps', str(a.reps), '--warmup-samples', str(a.warmups),
                       *options]
            print('Executing', block, name, flush=True)
            result = subprocess.run(command)
            rows.append({'block': block, 'case': name, 'options': options,
                         'exit_code': result.returncode,
                         'summary': json.loads((dest / 'summary.json').read_text()) if result.returncode == 0 else None})
            (a.out / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    combined = []
    for name, options in cases:
        rates = []
        for row in rows:
            if row['case'] != name or row['exit_code']:
                continue
            raw = json.loads((a.out / f"block{row['block']}-{name}" / 'raw.json').read_text())
            for result in raw:
                if result['n_gen'] == 128 and result['n_prompt'] == 0:
                    rates.extend(128e9 / ns for ns in result['samples_ns'][a.warmups:])
        combined.append({'case': name, 'options': options, 'samples': len(rates),
                         'median_tps': statistics.median(rates) if rates else None,
                         'min_tps': min(rates) if rates else None, 'max_tps': max(rates) if rates else None,
                         'spread_tps': max(rates) - min(rates) if rates else None})
    (a.out / 'report.json').write_text(json.dumps({'scope': 'exploratory only; requires final measurement and correctness', 'results': combined}, indent=2) + '\n')
    print(json.dumps(combined, indent=2), flush=True)

if __name__ == '__main__':
    main()
