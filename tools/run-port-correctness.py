#!/usr/bin/env python3
"""Run all port oracle suites with recorded physical-device assertions."""
import argparse
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess

ROOT = Path(__file__).resolve().parent.parent
helpers = runpy.run_path(str(ROOT / 'tools/port-benchmark.py'))
DEVICES, digest = helpers['DEVICES'], helpers['digest']
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
if socket.gethostname() != 'BigBoy' or a.out.exists():
    raise SystemExit('Run on BigBoy with a fresh output directory')
a.out.mkdir(parents=True)
inventory = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,pci.bus_id,compute_cap,driver_version','--format=csv,noheader'],text=True)
(a.out / 'hardware.csv').write_text(inventory)
commands = []
for role, (uuid,pci,node) in DEVICES.items():
    rows = [line for line in inventory.splitlines() if uuid in line]
    if len(rows) != 1:
        raise SystemExit('Expected GPU missing')
    name,_,found_pci,cc,_ = map(str.strip,rows[0].split(','))
    if name != 'NVIDIA GeForce GTX 1080 Ti' or cc != '6.1' or found_pci.lower().lstrip('0') != pci.lower().lstrip('0'):
        raise SystemExit('Physical GPU assertion failed')
    env = {key:value for key,value in os.environ.items() if not key.startswith('LEGACY_AI_')}
    env.update(CUDA_VISIBLE_DEVICES=uuid, GGML_CUDA_DISABLE_GRAPHS='1',
               LD_LIBRARY_PATH=str((ROOT / '.deps/llama.cpp/build-default/bin').resolve())+':/usr/local/cuda-12.9/lib64')
    for target in ['q4k_q8_test','runtime_kernel_test','attention_kernel_test','value_kernel_test']:
        binary = ROOT / 'build-sm61' / target
        command = ['numactl',f'--cpunodebind={node}',f'--membind={node}',str(binary)]
        if target == 'q4k_q8_test':
            command += ['--model',str((ROOT / 'models/Qwen3-4B-Q4_K_M.gguf').resolve()),
                        '--assert-name',name,'--assert-cc','6.1','--assert-pci',pci,'--assert-uuid',uuid,
                        '--envelope-out',str(a.out/(role+'-envelope.json'))]
        commands.append({'role':role,'gpu':rows[0],'argv':command,'binary_sha256':digest(binary),
                         'environment':{key:env[key] for key in ['CUDA_VISIBLE_DEVICES','GGML_CUDA_DISABLE_GRAPHS','LD_LIBRARY_PATH']}})
        (a.out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        print('Executing',role,target,flush=True)
        with (a.out/(role+'-'+target+'.txt')).open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
print('All oracle suites completed on both identified GTX 1080 Ti GPUs.',flush=True)
