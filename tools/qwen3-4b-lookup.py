#!/usr/bin/env python3
"""Experimental greedy prompt lookup for repeated code/text, using Qwen3-4B only.

Batched numerics can change greedy outputs on prose. This mode is explicit
and remains separate from the quality-qualified ordinary chat launcher.
"""
import argparse
import os
from pathlib import Path
import runpy
import socket
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parent.parent
helpers=runpy.run_path(str(ROOT/'tools/port-benchmark.py'))
DEVICES=helpers['DEVICES']
p=argparse.ArgumentParser(description=__doc__)
g=p.add_mutually_exclusive_group(required=True)
g.add_argument('--prompt'); g.add_argument('--file',type=Path,help='Plain user prompt, wrapped in the Qwen nonthinking template')
p.add_argument('--gpu',choices=[*DEVICES,'both'],default='secondary')
p.add_argument('--tokens',type=int,default=2048)
p.add_argument('--draft',type=int,choices=range(0,65),default=16,metavar='0..64')
p.add_argument('--output',type=Path,help='Write generated text here; existing files are refused')
p.add_argument('--allreduce',choices=['none','nccl'],default='none')
a=p.parse_args()
if socket.gethostname()!='BigBoy': raise SystemExit('Run on the qualified BigBoy host')
if a.tokens<1 or (a.output and a.output.exists()): raise SystemExit('Invalid token limit or existing output file')
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,pci.bus_id,compute_cap','--format=csv,noheader'],text=True)
roles=list(DEVICES) if a.gpu=='both' else [a.gpu]
for role in roles:
 uuid,pci,_=DEVICES[role]; rows=[x for x in inventory.splitlines() if uuid in x]
 if len(rows)!=1: raise SystemExit('GPU absent')
 fields=list(map(str.strip,rows[0].split(',')))
 if fields[0]!='NVIDIA GeForce GTX 1080 Ti' or fields[3]!='6.1' or fields[2].lower().lstrip('0')!=pci.lower().lstrip('0'): raise SystemExit('GPU identity mismatch')
model=ROOT/'models/Qwen3-4B-Q4_K_M.gguf'; binary=ROOT/'build-sm61/lookup_decode'; libs=ROOT/'.deps/llama.cpp-port/build-port/bin'
if not model.exists() or not binary.exists(): raise SystemExit('Build the port and provide the canonical model first')
node=0 if a.gpu=='both' else DEVICES[a.gpu][2]
env={k:v for k,v in os.environ.items() if not k.startswith('LEGACY_AI_')}
env.update(CUDA_VISIBLE_DEVICES=','.join(DEVICES[role][0] for role in roles),GGML_CUDA_DISABLE_GRAPHS='1',GGML_CUDA_ALLREDUCE=a.allreduce,LEGACY_AI_TEST_TENSOR='1' if a.gpu=='both' else '0',LD_LIBRARY_PATH=str(libs)+':/usr/local/cuda-12.9/lib64')
prompt=a.file.read_text() if a.file else a.prompt
with tempfile.TemporaryDirectory(prefix='qwen3-lookup-') as directory:
 dest=Path(directory); input_file=dest/'prompt.txt'; prefix=dest/'result'
 input_file.write_text('<|im_start|>user\n'+prompt+' /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
 command=['numactl',f'--cpunodebind={node}',f'--membind={node}',str(binary),str(model),str(input_file),str(prefix),str(a.tokens),str(a.draft),'4','1']
 completed=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 if completed.returncode: raise SystemExit(completed.stderr)
 text=prefix.with_suffix('.txt').read_text()
 if a.output:
  with a.output.open('x') as output: output.write(text)
 else: print(text,end='')
