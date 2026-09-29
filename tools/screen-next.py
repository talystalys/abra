#!/usr/bin/env python3
"""Serial exploratory placement/cache screen; failures are retained."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--kind',choices=['placement','nccl'],default='placement')
a=p.parse_args()
out = root / ('bench/results/1080ti-port/next-screen' if a.kind=='placement' else 'bench/results/1080ti-port/next-nccl-screen')
out.mkdir(exist_ok=False)
configs = [
    ('f16', []), ('q8', ['--cache-k','q8_0','--cache-v','q8_0']),
    ('q4', ['--cache-k','q4_0','--cache-v','q4_0']),
    ('cpu-kv', ['--no-kv-offload']),
    ('layers32', ['--gpu-layers','32','--threads','16']),
    ('layers24', ['--gpu-layers','24','--threads','16']),
    ('threads1', ['--threads','1']), ('threads8', ['--threads','8']),
    ('tensor', ['--gpu','both','--split-mode','tensor']),
]
results = []
if a.kind=='nccl':
    configs=[('single',[]),('butterfly',['--gpu','both','--split-mode','tensor','--allreduce','none']),('nccl',['--gpu','both','--split-mode','tensor','--allreduce','nccl','--nccl-proof'])]
for depth in ([2048,8192] if a.kind=='placement' else [0,2048,8192]):
    for name, flags in configs:
        dest = out / f'd{depth}-{name}'
        cmd = [sys.executable,str(root/'tools/port-benchmark.py'),
               '--source',str(root/'.deps/llama.cpp-port'),
               '--binary',str(root/'.deps/llama.cpp-port/build-port/bin/llama-bench'),
               '--out',str(dest),'--gpu','secondary','--flash-attn','on',
               '--depth',str(depth),'--prompt','0','--gen','128',
               '--reps','3','--warmup-samples','1',*flags]
        print('SCREEN', depth,name,flush=True)
        with (out/f'd{depth}-{name}-runner.txt').open('w') as log:
            completed = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        row = {'depth':depth,'config':name,'exit_code':completed.returncode}
        if (dest/'summary.json').exists():
            row['summary'] = json.loads((dest/'summary.json').read_text())
        results.append(row)
        (out/'screen.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(row),flush=True)
