#!/usr/bin/env python3
"""Exercise actual chat, completion, fallback and experimental batch launchers."""
import argparse
import datetime
import json
from pathlib import Path
import re
import runpy
import socket
import subprocess

root=Path(__file__).resolve().parent.parent
digest=runpy.run_path(str(root/'tools/port-benchmark.py'))['digest']
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if socket.gethostname()!='BigBoy' or a.out.exists():raise SystemExit('Run on BigBoy with a fresh output directory')
a.out.mkdir(parents=True)
prompt=a.out/'prompt.txt'
prompt.write_text('<|im_start|>user\nWhat is 17 times 23? Output only the integer. /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
normal=['python3',str(root/'tools/qwen3-4b.py')]
lookup=['python3',str(root/'tools/qwen3-4b-lookup.py')]
tests=[
 ('default-chat',normal+['--','--reasoning','off','--single-turn','--temp','0','--seed','1','-p','What is 17 times 23? Output only the integer. /no_think','-n','32'],'391',0),
 ('default-completion',normal+['--completion','--','-f',str(prompt),'-no-cnv','--no-display-prompt','--simple-io','--temp','0','--seed','1','-n','32'],'391',0),
 ('flash-completion',normal+['--runtime','flash','--gpu','secondary','--completion','--','-f',str(prompt),'-no-cnv','--no-display-prompt','--simple-io','--temp','0','--seed','1','-n','32'],'391',0),
 ('lookup-both',lookup+['--gpu','both','--allreduce','nccl','--prompt','What is the capital of France? Output only the city name.','--tokens','32'],'Paris',0),
 ('lookup-no-draft',lookup+['--draft','0','--prompt','What is the capital of France? Output only the city name.','--tokens','32'],'Paris',0),
 ('invalid-gpu',normal+['--runtime','tensor','--gpu','secondary'],'Tensor mode requires both GPUs',1),
 ('existing-output',lookup+['--prompt','Paris','--output',str(prompt)],'existing output file',1),
]
manifest={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_sha256':digest(root/'models/Qwen3-4B-Q4_K_M.gguf'),'scripts':{name:digest(root/'tools'/name) for name in ['qwen3-4b.py','qwen3-4b-lookup.py',Path(__file__).name]},'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,pci.bus_id,compute_cap,driver_version,power.limit','--format=csv,noheader'],text=True),'commands':[]}
bins=root/'.deps/llama.cpp-port/build-port/bin'
manifest['runtime_files']={name:digest(bins/name) for name in ['llama-cli','llama-completion','libllama.so','libggml-cuda.so']}
manifest['lookup_binary_sha256']=digest(root/'build-sm61/lookup_decode')
success=True
for label,command,expected,exit_code in tests:
 print('Executing',label,flush=True)
 completed=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=120)
 (a.out/(label+'-stdout.txt')).write_text(completed.stdout)
 (a.out/(label+'-stderr.txt')).write_text(completed.stderr)
 output=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',completed.stdout)
 if label.endswith('completion'):output=output.replace('[end of text]','')
 if label=='default-chat':passed=bool(re.search(r'\n391\s*\n',output))
 elif exit_code:passed=expected in completed.stderr
 else:passed=output.strip()==expected
 passed=passed and completed.returncode==exit_code
 manifest['commands'].append({'label':label,'argv':command,'exit_code':completed.returncode,'expected':expected,'passed':passed})
 success=success and passed
 (a.out/'report.json').write_text(json.dumps({**manifest,'all_passed':success},indent=2)+'\n')
 print(label,passed,flush=True)
if not success:raise SystemExit('Launcher smoke failure; inspect raw outputs')
