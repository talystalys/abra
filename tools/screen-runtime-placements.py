#!/usr/bin/env python3
"""Reproduce the placement screen using the corrected unique-prompt protocol."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workloads', type=Path, required=True)
    a = p.parse_args()
    helpers = runpy.run_path(str(ROOT / 'tools/compare-runtimes.py'))
    configs = [name for name, c in helpers['configurations']().items() if not c.get('nccl') and not c.get('force_cuda')]
    argv = [sys.executable, str(ROOT / 'tools/compare-runtimes.py'), '--out', str(a.out),
            '--stage', 'screen', '--configs', ','.join(configs), '--workloads', str(a.workloads),
            '--depths', '2048', '--warmups', '1', '--reps', '3', '--blocks', '1']
    subprocess.run(argv, check=True)
    (a.out / 'invocation.json').write_text(json.dumps({'argv': argv}, indent=2) + '\n')


if __name__ == '__main__':
    main()
