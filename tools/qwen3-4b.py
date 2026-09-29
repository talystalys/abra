#!/usr/bin/env python3
"""Launch Qwen3-4B on an identified BigBoy card with the pinned runtime."""
import argparse
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parent.parent
DEVICES = {
    'primary': ('GPU-3007f159-e0cf-9303-0a81-41ea8e945315', '0000:03:00.0', 0),
    'secondary': ('GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c', '0000:81:00.0', 1),
}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu', choices=[*DEVICES,'both'], help='tensor uses both cards; other modes default to secondary')
p.add_argument('--allreduce',choices=['nccl','none'],default='nccl',help='Communication implementation for tensor mode')
p.add_argument('--runtime', choices=['baseline', 'flash', 'tensor', 'exact', 'graphs', 'packed'], default='tensor',
               help='tensor: two GPU Flash Attention; flash: single GPU attention; exact: validated QK/value kernels; packed: research only')
p.add_argument('--completion', action='store_true', help='Use the noninteractive completion tool instead of chat')
p.add_argument('--model', type=Path, default=ROOT / 'models/Qwen3-4B-Q4_K_M.gguf')
p.add_argument('llama_args', nargs=argparse.REMAINDER, help='Arguments after -- are forwarded to llama-cli')
a = p.parse_args()
if socket.gethostname() != 'BigBoy':
    raise SystemExit('This configuration identifies the qualified BigBoy GPUs')
selected = a.gpu or ('both' if a.runtime == 'tensor' else 'secondary')
if (selected == 'both') != (a.runtime == 'tensor'):
    raise SystemExit('Tensor mode requires both GPUs; other modes use one GPU')
roles = list(DEVICES) if selected == 'both' else [selected]
node = 0 if selected == 'both' else DEVICES[selected][2]
uuid = ','.join(DEVICES[role][0] for role in roles)
inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,pci.bus_id,compute_cap', '--format=csv,noheader'], text=True)
for role in roles:
    device_uuid, pci, _ = DEVICES[role]
    rows = [x.split(',') for x in inventory.splitlines() if device_uuid in x]
    if len(rows) != 1: raise SystemExit('Expected GPU is absent')
    name, found_uuid, found_pci, cc = map(str.strip, rows[0])
    if name != 'NVIDIA GeForce GTX 1080 Ti' or cc != '6.1' or found_pci.lower().lstrip('0') != pci.lower().lstrip('0'):
        raise SystemExit('GPU identity mismatch')
control = a.runtime == 'baseline' and not a.completion
source = ROOT / '.deps/llama.cpp' if control else ROOT / '.deps/llama.cpp-port'
binary = source / ('build-default' if control else 'build-port') / ('bin/llama-completion' if a.completion else 'bin/llama-cli')
if not binary.exists() or not a.model.exists():
    raise SystemExit('Build the runtime and provide the canonical model first')
if a.runtime == 'tensor' and a.allreduce == 'nccl':
    dependencies = subprocess.check_output(['ldd', str(binary.parent / 'libggml-cuda.so')], text=True)
    if 'libnccl.so' not in dependencies or 'not found' in dependencies:
        raise SystemExit('Build with --nccl-prefix first, or select --runtime flash for one GPU')
env = os.environ.copy()
for key in list(env):
    if key.startswith('LEGACY_AI_'):
        del env[key]
env.update(GGML_CUDA_ALLREDUCE=a.allreduce if a.runtime=='tensor' else 'none', CUDA_VISIBLE_DEVICES=uuid, LEGACY_AI_PASCAL_GRAPHS='1' if a.runtime == 'graphs' else '0',
           LEGACY_AI_ATTENTION='1' if a.runtime == 'exact' else '0', LEGACY_AI_ATTENTION_ROWS='4',
           LEGACY_AI_ATTENTION_V='1' if a.runtime == 'exact' else '0', LEGACY_AI_ATTENTION_V_ROWS='2',
           LEGACY_AI_Q4K='1' if a.runtime == 'packed' else '0', LEGACY_AI_Q6K='0',
           LEGACY_AI_Q4K_WPR='2', LEGACY_AI_Q4K_WPB='8', LEGACY_AI_Q4K_SUMS='0',
           LEGACY_AI_PREPACK='1' if a.runtime == 'packed' else '0',
           LD_LIBRARY_PATH=str(binary.parent.resolve()) + ':/usr/local/cuda-12.9/lib64')
env.pop('LEGACY_AI_PROFILE', None)
if a.runtime == 'graphs':
    env.pop('GGML_CUDA_DISABLE_GRAPHS', None)
else:
    env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
extra = a.llama_args[1:] if a.llama_args[:1] == ['--'] else a.llama_args
command = ['numactl', f'--cpunodebind={node}', f'--membind={node}', str(binary.resolve()),
           '-m', str(a.model.resolve()), '-ngl', '99', '-sm', 'tensor' if a.runtime=='tensor' else 'none', '-t', '4', '-tb', '4',
           '-c', '4096', '-b', '2048', '-ub', '512',
           '-ctk', 'f16', '-ctv', 'f16', '-fa', 'on' if a.runtime in ['flash','tensor'] else 'off', *(['-ts','1,1'] if a.runtime=='tensor' else []), *extra]
os.execvpe(command[0], command, env)
