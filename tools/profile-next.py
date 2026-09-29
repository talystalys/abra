#!/usr/bin/env python3
"""Record a synchronous Qwen3-4B decode profile, separately from speed tests."""
import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
root=Path(__file__).resolve().parent.parent
helpers=runpy.run_path(str(root/'tools/port-benchmark.py'))
digest=helpers['digest'];uuid,pci,node=helpers['DEVICES']['primary']
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if socket.gethostname()!='BigBoy' or a.out.exists():raise SystemExit('Run on BigBoy with a fresh directory')
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,pci.bus_id,compute_cap,driver_version,power.limit','--format=csv,noheader'],text=True)
rows=[x for x in inventory.splitlines() if uuid in x]
if len(rows)!=1:raise SystemExit('Missing primary GPU')
fields=[x.strip() for x in rows[0].split(',')]
if fields[0]!='NVIDIA GeForce GTX 1080 Ti' or fields[3]!='6.1' or fields[2].lower().lstrip('0')!=pci.lower().lstrip('0'):raise SystemExit('GPU identity mismatch')
a.out.mkdir(parents=True)
source=root/'.deps/llama.cpp-port';libs=source/'build-port/bin';model=root/'models/Qwen3-4B-Q4_K_M.gguf';probe=root/'build-sm61/runtime_logits'
env={k:v for k,v in os.environ.items() if not k.startswith('LEGACY_AI_')}
env.update(CUDA_VISIBLE_DEVICES=uuid,LEGACY_AI_PROFILE='1',LEGACY_AI_VALIDATE_FLASH_ATTN='1',LEGACY_AI_VALIDATE_SPLIT='none',GGML_CUDA_DISABLE_GRAPHS='1',GGML_CUDA_ALLREDUCE='none',LD_LIBRARY_PATH=str(libs)+':/usr/local/cuda-12.9/lib64')
command=['numactl',f'--cpunodebind={node}',f'--membind={node}',str(probe),str(model),str(a.out/'logits.tmp'),str(a.out/'tokens.json'),'8','2048']
manifest={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hardware':inventory,'model_sha256':digest(model),'probe_sha256':digest(probe),'runner_sha256':digest(Path(__file__)),'cpp_sha256':digest(root/'tests/runtime_logits.cpp'),'source_head':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'libraries':{x.name:digest(x) for x in libs.glob('*.so') if not x.is_symlink()},'argv':command,'environment':{k:v for k,v in env.items() if k.startswith('LEGACY_AI_') or k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_ALLREDUCE']},'note':'Synchronizes every profiled operation; diagnostic timing only, never a throughput benchmark.'}
(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(a.out/'source.diff').write_text(subprocess.check_output(['git','-C',str(source),'diff','HEAD'],text=True))
with (a.out/'stdout.txt').open('w') as out,(a.out/'stderr.txt').open('w') as err:subprocess.run(command,env=env,stdout=out,stderr=err,check=True,timeout=300)
subprocess.run(['python3',str(root/'tools/analyze-fa-profile.py'),str(a.out/'stderr.txt'),'--out',str(a.out/'report.json')],check=True)
