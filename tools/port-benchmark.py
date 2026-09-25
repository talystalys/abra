#!/usr/bin/env python3
"""UUID-bound, logged inference benchmarks for the GTX 1080 Ti port.

Run on BigBoy. No clock or power changes. Each invocation produces a fresh
directory with exact argv, environment, hardware, source/binary/model hashes,
raw llama-bench samples, offload stderr, and GPU telemetry.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

DEVICES = {
    'primary': ('GPU-3007f159-e0cf-9303-0a81-41ea8e945315', '0000:03:00.0', 0),
    'secondary': ('GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c', '0000:81:00.0', 1),
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def output(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', choices=[*DEVICES, 'both'], default='primary')
    p.add_argument('--model', type=Path, default=Path('/home/taly/legacy-ai/models/Qwen3-4B-Q4_K_M.gguf'))
    p.add_argument('--split-mode', choices=['none', 'layer', 'row'], help='Explicit placement; row/layer require both GPUs')
    p.add_argument('--warmup-samples', type=int, default=0, help='Exclude this many full workload samples per row before measured samples')
    p.add_argument('--depth', type=int, default=0)
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--prompt', type=int, default=512)
    p.add_argument('--gen', type=int, default=128)
    p.add_argument('--graphs', action='store_true')
    p.add_argument('--q4k', action='store_true')
    p.add_argument('--q4k-wpr', choices=[1, 2, 4], type=int, default=2)
    p.add_argument('--q4k-wpb', choices=[4, 8, 16], type=int, default=8)
    p.add_argument('--attention-v', action='store_true')
    p.add_argument('--attention-v-rows', choices=[2, 4], type=int, default=2)
    p.add_argument('--attention', action='store_true')
    p.add_argument('--attention-rows', choices=[4, 8], type=int, default=4)
    p.add_argument('--q4k-sums', action='store_true')
    p.add_argument('--prepack', action='store_true', help='Lossless metadata expansion; extra model-owned VRAM cache')
    p.add_argument('--q6k', action='store_true')
    p.add_argument('--q6k-wpr', choices=[1, 2, 4], type=int, default=2)
    p.add_argument('--q6k-wpb', choices=[4, 8, 16], type=int, default=8)
    p.add_argument('--flash-attn', choices=['on', 'off', 'auto'], default='off')
    p.add_argument('--offload-proof', action='store_true', help='Collect verbose offload evidence in a separate untimed invocation')
    a = p.parse_args()
    if socket.gethostname() != 'BigBoy':
        raise SystemExit('GPU benchmarks must run on BigBoy')
    if a.split_mode in ['layer', 'row'] and a.gpu != 'both':
        raise SystemExit('Row/layer splitting requires both explicitly identified GPUs')
    if a.reps < 1 or a.gen < 1 or a.depth < 0 or a.warmup_samples < 0:
        raise SystemExit('Invalid benchmark dimensions')
    if a.out.exists():
        raise SystemExit(f'Refusing to overwrite results: {a.out}')
    a.out.mkdir(parents=True)
    inventory = output(['nvidia-smi', '--query-gpu=name,uuid,pci.bus_id,compute_cap,memory.total,driver_version,power.limit', '--format=csv,noheader'])
    (a.out / 'hardware.csv').write_text(inventory)
    selected = list(DEVICES) if a.gpu == 'both' else [a.gpu]
    for role in selected:
        uuid, pci, _ = DEVICES[role]
        lines = [line for line in inventory.splitlines() if uuid in line]
        if len(lines) != 1:
            raise SystemExit(f'Missing expected device: {uuid}')
        row = [item.strip() for item in lines[0].split(',')]
        if row[0] != 'NVIDIA GeForce GTX 1080 Ti' or row[3] != '6.1' or row[2].lower().lstrip('0') != pci.lower().lstrip('0'):
            raise SystemExit(f'Physical identity mismatch: {row}')
    env = os.environ.copy()
    env.pop('LEGACY_AI_PROFILE', None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(DEVICES[r][0] for r in selected),
               LEGACY_AI_PASCAL_GRAPHS='1' if a.graphs else '0',
               LEGACY_AI_ATTENTION_V='1' if a.attention_v else '0', LEGACY_AI_ATTENTION_V_ROWS=str(a.attention_v_rows), LEGACY_AI_ATTENTION='1' if a.attention else '0', LEGACY_AI_ATTENTION_ROWS=str(a.attention_rows), LEGACY_AI_PREPACK='1' if a.prepack else '0', LEGACY_AI_Q4K='1' if a.q4k else '0', LEGACY_AI_Q4K_WPR=str(a.q4k_wpr),
               LEGACY_AI_Q4K_WPB=str(a.q4k_wpb), LEGACY_AI_Q4K_SUMS='1' if a.q4k_sums else '0', LEGACY_AI_Q6K='1' if a.q6k else '0',
               LEGACY_AI_Q6K_WPR=str(a.q6k_wpr), LEGACY_AI_Q6K_WPB=str(a.q6k_wpb))
    if a.graphs:
        env.pop('GGML_CUDA_DISABLE_GRAPHS', None)
    else:
        env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    env['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.9/lib64:' + env.get('LD_LIBRARY_PATH', '')
    command = ['numactl', '--cpunodebind=0', '--membind=0'] if a.gpu == 'both' else ['numactl', f'--cpunodebind={DEVICES[a.gpu][2]}', f'--membind={DEVICES[a.gpu][2]}']
    command += [str(a.binary.resolve()), '-m', str(a.model.resolve()), '-p', str(a.prompt), '-n', str(a.gen), '-d', str(a.depth), '-b', '2048', '-ub', '512', '-ctk', 'f16', '-ctv', 'f16', '-ngl', '99', '-r', str(a.reps + a.warmup_samples), '-t', str(a.threads), '-fa', a.flash_attn, '-sm', a.split_mode or ('layer' if a.gpu == 'both' else 'none'), '-o', 'json']
    if a.gpu == 'both':
        command += ['-ts', '1/1']
    source_diff = output(['git', '-C', str(a.source), 'diff', 'HEAD'])
    (a.out / 'source.diff').write_text(source_diff)
    manifest = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host': socket.gethostname(), 'argv': command,
        'environment': {k: env.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'LEGACY_AI_PASCAL_GRAPHS', 'LEGACY_AI_ATTENTION_V', 'LEGACY_AI_ATTENTION_V_ROWS', 'LEGACY_AI_ATTENTION', 'LEGACY_AI_ATTENTION_ROWS', 'LEGACY_AI_PREPACK', 'LEGACY_AI_Q4K', 'LEGACY_AI_Q4K_WPR', 'LEGACY_AI_Q4K_WPB', 'LEGACY_AI_Q4K_SUMS', 'LEGACY_AI_Q6K', 'LEGACY_AI_Q6K_WPR', 'LEGACY_AI_Q6K_WPB', 'GGML_CUDA_DISABLE_GRAPHS', 'LD_LIBRARY_PATH']},
        'model_sha256': digest(a.model), 'binary_sha256': digest(a.binary),
        'source_commit': output(['git', '-C', str(a.source), 'rev-parse', 'HEAD']).strip(),
        'source_diff_sha256': hashlib.sha256(source_diff.encode()).hexdigest(),
        'warmup': 'llama-bench default prompt/decode warmup enabled; context prefill excluded from decode samples',
        'excluded_full_workload_warmup_samples': a.warmup_samples,
        'clocks': 'Dynamic clocks; no clock/power configuration changes',
        'toolchain': output(['/usr/local/cuda-12.9/bin/nvcc', '--version']),
    }
    runtime_kernel = a.source / 'ggml/src/ggml-cuda/legacy-ai-mmvq.cuh'
    if runtime_kernel.exists():
        manifest['runtime_kernel_sha256'] = digest(runtime_kernel)
        (a.out / 'runtime-kernel.txt').write_text(runtime_kernel.read_text())
    packed_header = a.source / 'ggml/src/ggml-cuda/legacy-ai-packed-q4.cuh'
    if packed_header.exists():
        manifest['packed_header_sha256'] = digest(packed_header)
        (a.out / 'packed-format.txt').write_text(packed_header.read_text())
    attention_header = a.source / 'ggml/src/ggml-cuda/legacy-ai-attention.cuh'
    if attention_header.exists():
        manifest['attention_header_sha256'] = digest(attention_header)
        (a.out / 'attention-kernel.txt').write_text(attention_header.read_text())
    library_files = sorted(a.binary.parent.glob('lib*.so*'))
    manifest['library_sha256'] = {x.name: digest(x) for x in library_files if x.is_file() and not x.is_symlink()}
    cache = a.binary.parent.parent / 'CMakeCache.txt'
    if cache.exists():
        (a.out / 'CMakeCache.txt').write_text(cache.read_text())
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if a.offload_proof:
        proof = command.copy()
        for flag, value in [('-p', '32'), ('-n', '0'), ('-d', '0'), ('-r', '1')]:
            proof[proof.index(flag) + 1] = value
        proof += ['-v']
        manifest['untimed_offload_proof_argv'] = proof
        with open(a.out / 'offload.json', 'w') as raw, open(a.out / 'offload-stderr.txt', 'w') as err:
            subprocess.run(proof, env=env, stdout=raw, stderr=err, check=True, timeout=300)
        if 'offloaded 37/37 layers' not in (a.out / 'offload-stderr.txt').read_text():
            raise SystemExit('Full Qwen3-4B GPU offload not confirmed')
    telemetry_args = ['nvidia-smi', '--query-gpu=timestamp,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used,pcie.link.gen.current,pcie.link.width.current', '--format=csv', '-lms', '250']
    with open(a.out / 'telemetry.csv', 'w') as telemetry, open(a.out / 'raw.json', 'w') as raw, open(a.out / 'stderr.txt', 'w') as stderr:
        sampler = subprocess.Popen(telemetry_args, stdout=telemetry, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            result = subprocess.run(command, env=env, stdout=raw, stderr=stderr, timeout=1800)
            manifest['elapsed_seconds'] = time.monotonic() - started
            manifest['exit_code'] = result.returncode
        finally:
            sampler.terminate()
            sampler.wait(timeout=10)
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(f'Benchmark failed ({result.returncode}); see {a.out}/stderr.txt')
    rows = json.loads((a.out / 'raw.json').read_text())
    summary = []
    for row in rows:
        if len(row['samples_ns']) != a.reps + a.warmup_samples:
            raise SystemExit('Invalid total sample count')
        ns = row['samples_ns'][a.warmup_samples:]
        tokens = row['n_prompt'] + row['n_gen']
        if len(ns) != a.reps or not all(x > 0 for x in ns):
            raise SystemExit('Invalid raw sample count or timing')
        rates = [tokens * 1e9 / x for x in ns]
        summary.append({'n_prompt': row['n_prompt'], 'n_gen': row['n_gen'], 'n_depth': row['n_depth'], 'samples': len(ns), 'median_tps': statistics.median(rates), 'min_tps': min(rates), 'max_tps': max(rates), 'spread_tps': max(rates) - min(rates), 'median_ns': statistics.median(ns), 'stddev_tps': statistics.pstdev(rates)})
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'out': str(a.out), 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
