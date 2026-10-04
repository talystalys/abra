#!/usr/bin/env python3
"""Validate opt-in/fallback runtime modes against pinned upstream logits."""
import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess

ROOT = Path(__file__).resolve().parent.parent
UUIDS = {'primary': 'GPU-3007f159-e0cf-9303-0a81-41ea8e945315',
         'secondary': 'GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c',
         'both': 'GPU-3007f159-e0cf-9303-0a81-41ea8e945315,GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runtime-lib', type=Path, required=True)
    p.add_argument('--baseline-lib', type=Path, default=Path('/home/taly/legacy-ai/.deps/llama.cpp/build-default/bin'))
    p.add_argument('--probe', type=Path, default=ROOT / 'build-sm61/runtime_logits')
    p.add_argument('--gpu', choices=list(UUIDS), default='primary')
    p.add_argument('--depths', type=int, nargs='+', default=[0, 2048])
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--kernels', action='store_true')
    p.add_argument('--tensor-kernels', action='store_true')
    p.add_argument('--q4k-wpr', type=int, choices=[1, 2, 4], default=2)
    p.add_argument('--q4k-wpb', type=int, choices=[4, 8, 16], default=8)
    p.add_argument('--q6k-wpr', type=int, choices=[1, 2, 4], default=2)
    p.add_argument('--q6k-wpb', type=int, choices=[4, 8, 16], default=8)
    p.add_argument('--q6k', action='store_true')
    p.add_argument('--attention-v', action='store_true')
    p.add_argument('--attention-v-rows', choices=[2, 4], type=int, default=2)
    p.add_argument('--attention', action='store_true')
    p.add_argument('--attention-rows', choices=[4, 8], type=int, default=4)
    p.add_argument('--q4k-sums', action='store_true')
    p.add_argument('--prepack', action='store_true', help='Lossless metadata expansion; extra model-owned VRAM cache')
    p.add_argument('--split-mode', choices=['row', 'layer', 'tensor'])
    p.add_argument('--reference-split-mode',choices=['none','tensor'],default='none')
    p.add_argument('--no-graphs',action='store_true',help='Omit graph-specific modes when qualifying communication changes')
    p.add_argument('--reuse-q8', action='store_true', help='Compare graph-local activation reuse with the frozen reference')
    p.add_argument('--split-tied-output', action='store_true')
    p.add_argument('--flash-attn', action='store_true')
    p.add_argument('--reference-flash-attn', action='store_true', help='Compare fallback/graphs against upstream Flash Attention, retaining strict checks')
    a = p.parse_args()
    if socket.gethostname() != 'BigBoy':
        raise SystemExit('Run on BigBoy')
    if (a.split_mode or a.reference_split_mode=='tensor') and a.gpu != 'both':
        raise SystemExit('Both GPUs are required for split validation')
    if a.out.exists():
        raise SystemExit('Refusing to overwrite validation results')
    a.out.mkdir(parents=True)
    devices = runpy.run_path(str(ROOT / 'tools/port-benchmark.py'))['DEVICES']
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,pci.bus_id,compute_cap', '--format=csv,noheader'], text=True)
    (a.out / 'hardware.csv').write_text(inventory)
    for role in list(devices) if a.gpu == 'both' else [a.gpu]:
        uuid, pci, _ = devices[role]
        lines = [line for line in inventory.splitlines() if uuid in line]
        if len(lines) != 1:
            raise SystemExit('Expected physical GPU missing')
        name, _, actual_pci, cc = map(str.strip, lines[0].split(','))
        if name != 'NVIDIA GeForce GTX 1080 Ti' or cc != '6.1' or actual_pci.lower().lstrip('0') != pci.lower().lstrip('0'):
            raise SystemExit('Physical GPU assertion failed')
    modes = {'baseline': (a.baseline_lib, False, False),
             'fallback': (a.runtime_lib, False, False),
             'graphs': (a.runtime_lib, True, False)}
    if a.kernels:
        modes.update(kernels=(a.runtime_lib, False, True), both=(a.runtime_lib, True, True))
    if a.attention_v:
        modes['exact-value'] = (a.runtime_lib, False, False)
        modes['exact-value-graphs'] = (a.runtime_lib, True, False)
        if a.attention:
            modes['exact-attention-value'] = (a.runtime_lib, False, False)
            modes['exact-attention-value-graphs'] = (a.runtime_lib, True, False)
    if a.attention:
        modes['exact-attention'] = (a.runtime_lib, False, False)
        modes['exact-attention-graphs'] = (a.runtime_lib, True, False)
    if a.flash_attn:
        modes['flash-attention'] = (a.runtime_lib, False, False)
    if a.reuse_q8:
        modes['reuse-q8'] = (a.runtime_lib, False, False)
        if a.kernels:
            modes['kernels-reuse-q8'] = (a.runtime_lib, False, True)
    if a.split_tied_output:
        modes['split-output'] = (a.runtime_lib, False, False)
        if a.reuse_q8:
            modes['split-output-reuse'] = (a.runtime_lib, False, False)
        if a.kernels:
            modes['split-output-kernels'] = (a.runtime_lib, False, True)
            if a.reuse_q8:
                modes['split-output-kernels-reuse'] = (a.runtime_lib, False, True)
    if a.split_mode:
        modes[a.split_mode] = (a.baseline_lib, False, False)
    if a.no_graphs: modes = {name:value for name,value in modes.items() if not value[1]}
    commands = []
    reports = []
    for depth in a.depths:
        for name, (libraries, graphs, kernels) in modes.items():
            prefix = a.out / f'{name}-d{depth}'
            env = {key: value for key, value in os.environ.items() if not key.startswith('LEGACY_AI_')}
            env.update(LEGACY_AI_ATTENTION_V='1' if 'value' in name else '0', LEGACY_AI_ATTENTION_V_ROWS=str(a.attention_v_rows), LEGACY_AI_ATTENTION='1' if name.startswith('exact-attention') else '0', LEGACY_AI_ATTENTION_ROWS=str(a.attention_rows), LEGACY_AI_VALIDATE_SPLIT=a.split_mode if name == a.split_mode else a.reference_split_mode, LEGACY_AI_VALIDATE_FLASH_ATTN='1' if name == 'flash-attention' or a.reference_flash_attn else '0', CUDA_VISIBLE_DEVICES=UUIDS[a.gpu],
                       LEGACY_AI_REUSE_Q8='1' if name in ['reuse-q8', 'kernels-reuse-q8', 'split-output-reuse', 'split-output-kernels-reuse'] else '0',
                       LEGACY_AI_TENSOR_MMVQ='1' if kernels and a.tensor_kernels else '0',
                       LEGACY_AI_SPLIT_TIED_OUTPUT='1' if name.startswith('split-output') else '0',
                       LD_LIBRARY_PATH=str(libraries) + ':/usr/local/cuda-12.9/lib64',
                       LEGACY_AI_PASCAL_GRAPHS='1' if graphs else '0',
                       LEGACY_AI_PREPACK='1' if kernels and a.prepack else '0', LEGACY_AI_Q4K='1' if kernels else '0', LEGACY_AI_Q4K_WPR=str(a.q4k_wpr),
                       LEGACY_AI_Q4K_WPB=str(a.q4k_wpb), LEGACY_AI_Q4K_SUMS='1' if kernels and a.q4k_sums else '0', LEGACY_AI_Q6K='1' if kernels and a.q6k else '0',
                       LEGACY_AI_Q6K_WPR=str(a.q6k_wpr), LEGACY_AI_Q6K_WPB=str(a.q6k_wpb))
            if graphs:
                env.pop('GGML_CUDA_DISABLE_GRAPHS', None)
            else:
                env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
            command = ['numactl', f'--cpunodebind={1 if a.gpu == "secondary" else 0}',
                       f'--membind={1 if a.gpu == "secondary" else 0}', str(a.probe),
                       '/home/taly/legacy-ai/models/Qwen3-4B-Q4_K_M.gguf',
                       str(prefix) + '.tmp', str(prefix) + '-tokens.json', str(a.steps), str(depth)]
            commands.append({'mode': name, 'argv': command, 'environment': {k: v for k, v in env.items() if k.startswith(('LEGACY_AI_', 'NCCL_')) or k in ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_ALLREDUCE']}})
            (a.out / 'commands.json').write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'commands': commands}, indent=2) + '\n')
            with open(str(prefix) + '.txt', 'w') as f:
                subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=300)
            if env['LEGACY_AI_REUSE_Q8'] == '1' and 'graph-local Q8 activation reuse active' not in Path(str(prefix) + '.txt').read_text():
                raise SystemExit('Activation reuse candidate did not reuse any activations')
            if name.startswith('split-output') and 'tied Qwen output row splitting active' not in Path(str(prefix) + '.txt').read_text():
                raise SystemExit('Tied output did not split in the candidate')
            if kernels and a.prepack and 'packed=1' not in Path(str(prefix) + '.txt').read_text():
                raise SystemExit('Packed Q4_K kernel did not dispatch')
            if name.startswith('exact-attention') and 'exact attention QK dispatch active' not in Path(str(prefix) + '.txt').read_text():
                raise SystemExit('Exact attention kernel did not dispatch')
            if 'value' in name and 'exact attention value dispatch active' not in Path(str(prefix) + '.txt').read_text():
                raise SystemExit('Exact value kernel did not dispatch')
            if name != 'baseline':
                ref = a.out / f'baseline-d{depth}'
                report = Path(str(prefix) + '-comparison.json')
                subprocess.run(['python3', str(ROOT / 'tools/compare-runtime-logits.py'), str(ref) + '.tmp', str(prefix) + '.tmp', str(ref) + '-tokens.json', str(prefix) + '-tokens.json', '--out', str(report)], check=True)
                reports.append(json.loads(report.read_text()))
                if graphs and 'Pascal CUDA graph instantiated' not in Path(str(prefix) + '.txt').read_text():
                    raise SystemExit('Graph mode did not instantiate a graph')
                if kernels and 'Q4_K runtime dispatch active' not in Path(str(prefix) + '.txt').read_text():
                    raise SystemExit('Custom kernel mode did not dispatch custom kernels')
        print(f'Validated GPU={a.gpu}, depth={depth}, modes={list(modes)}', flush=True)
    (a.out / 'summary.json').write_text(json.dumps({'gpu': a.gpu, 'reports': reports, 'accepted': all(x['accepted'] for x in reports)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
