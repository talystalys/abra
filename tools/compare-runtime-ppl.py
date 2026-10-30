#!/usr/bin/env python3
"""Fixed-corpus PPL for the three llama.cpp builds; Ollama has no equivalent API."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import runpy
import socket
import subprocess

ROOT = Path(__file__).resolve().parent.parent
helpers = runpy.run_path(str(ROOT / 'tools/compare-runtimes.py'))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--configs', help='Selected llama.cpp configurations; primary-only duplicates are omitted')
    a = p.parse_args()
    if socket.gethostname() != 'BigBoy':
        raise SystemExit('Run on BigBoy')
    with (ROOT / '.deps/comparison.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out / 'runner-source.py').write_bytes(Path(__file__).read_bytes())
        model = helpers['MODEL']
        if helpers['digest'](model) != helpers['MODEL_SHA']:
            raise SystemExit('Model hash mismatch')
        corpus = ROOT / '.deps/quality-data/wiki.test.raw'
        result = {'scope': '20 fixed WikiText chunks; not a cross-Ollama perplexity comparison', 'runtimes': {}}
        configurations = helpers['configurations']()
        names = a.configs.split(',') if a.configs else ['legacy-secondary', 'stock-pinned-secondary', 'stock-current-secondary']
        names = [name for name in names if configurations[name]['kind'] == 'llama' and not name.endswith('-primary')]
        for name in names:
            c = configurations[name]
            source = c['source']
            build = 'build-port' if c['family'] in ['legacy', 'port-control'] else c.get('build', 'build-comparison')
            directory = a.out / name
            directory.mkdir()
            bins = ROOT / '.deps' / source / build / 'bin'
            binary = bins / 'llama-perplexity'
            env = {k: v for k, v in os.environ.items() if not k.startswith(('ABRA_', 'GGML_', 'NCCL_'))}
            both = c['gpu'] == 'both'
            env.update(CUDA_VISIBLE_DEVICES=','.join(x[0] for x in helpers['DEVICES']) if both else helpers['DEVICES'][1][0], LD_LIBRARY_PATH=str(bins) + ':/usr/local/cuda-12.9/lib64')
            if c['family'] in ['legacy', 'port-control']:
                flag = '1' if c['family'] == 'legacy' else '0'
                env.update(ABRA_REUSE_Q8=flag, ABRA_SPLIT_TIED_OUTPUT=flag, GGML_CUDA_DISABLE_GRAPHS='1')
            if both and (c['family'] in ['legacy', 'port-control'] or c.get('nccl')):
                env['GGML_CUDA_ALLREDUCE'] = 'nccl'
            node = '0' if both else '1'
            argv = ['numactl', '--cpunodebind=' + node, '--membind=' + node, str(binary), '-m', str(model),
                    '-f', str(corpus), '--chunks', '20', '-c', '2048', '-ngl', '99', '-sm', c['split'],
                    '-t', '4', '-tb', '4', '-b', '2048', '-ub', '512', '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16']
            if both:
                argv += ['-ts', '1,1']
            manifest = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': argv,
                        'environment': {k: v for k, v in env.items() if k.startswith(('CUDA_VISIBLE', 'LD_LIBRARY', 'ABRA_', 'GGML_'))},
                        'binary_sha256': helpers['digest'](binary), 'model_sha256': helpers['MODEL_SHA'],
                        'corpus_sha256': helpers['digest'](corpus), 'inventory': helpers['inventory']()}
            helpers['save'](directory / 'manifest.json', manifest)
            (directory / 'ldd.txt').write_text(helpers['output'](['ldd', binary], env=env))
            print('Perplexity', name, flush=True)
            with (directory / 'output.log').open('w') as log:
                process = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            text = (directory / 'output.log').read_text(errors='replace')
            matches = re.findall(r'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)', text)
            result['runtimes'][name] = {'exit_code': process.returncode,
                                       'ppl': float(matches[-1][0]) if matches else None,
                                       'reported_uncertainty': float(matches[-1][1]) if matches else None}
            helpers['save'](a.out / 'report.json', result)

if __name__ == '__main__':
    main()
