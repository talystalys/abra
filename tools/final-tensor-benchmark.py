#!/usr/bin/env python3
"""Balanced final Flash Attention comparison: one GPU versus tensor split."""
import argparse
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--allreduce',choices=['none','nccl'],default='none')
p.add_argument('--include-fallback',action='store_true')
a = p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
configs = [('baseline','secondary','none'),('tensor-nccl' if a.allreduce=='nccl' else 'tensor','both','tensor')]
if a.include_fallback and a.allreduce=='nccl': configs.insert(1,('tensor','both','tensor'))
for depth in [0,2048,8192]:
 for block,order in enumerate([configs,list(reversed(configs))]):
  for name,gpu,split in order:
   command = [sys.executable,str(root/'tools/port-benchmark.py'),
    '--source',str(root/'.deps/llama.cpp-port'),
    '--binary',str(root/'.deps/llama.cpp-port/build-port/bin/llama-bench'),
    '--out',str(a.out/f'd{depth}-block{block}-{name}'),
    '--gpu',gpu,'--split-mode',split,'--flash-attn','on',
    '--depth',str(depth),'--prompt','512','--gen','128',
    '--reps','10','--warmup-samples','5','--offload-proof',
    '--allreduce','none' if name=='tensor' else a.allreduce]
   if name=='tensor-nccl': command += ['--nccl-proof']
   print('Executing',depth,block,name,flush=True)
   subprocess.run(command,check=True)
subprocess.run([sys.executable,str(root/'tools/analyze-port-results.py'),str(a.out),'--out',str(a.out/'report.json')],check=True)
