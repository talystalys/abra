#!/usr/bin/env python3
"""Balanced full-response HTTP tests on coding, reasoning and ordinary prose."""
import argparse
import fcntl
import json
from pathlib import Path
import runpy
import socket

ROOT = Path(__file__).resolve().parent.parent
H = runpy.run_path(str(ROOT / 'tools/compare-runtimes.py'))
CASES = [
    {'id': 'code', 'prompt': 'Write a Python function that merges two sorted lists. Return only complete code.'},
    {'id': 'reasoning', 'prompt': 'Explain why the sum of integers from 1 to n equals n(n+1)/2.'},
    {'id': 'prose', 'prompt': 'Explain CPU cache lines, locality and memory bandwidth in three short paragraphs.'},
]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--configs', required=True)
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--warmups', type=int, default=2)
    a = p.parse_args()
    if socket.gethostname() != 'BigBoy' or H['digest'](H['MODEL']) != H['MODEL_SHA']:
        raise SystemExit('Wrong host or model')
    with (ROOT / '.deps/comparison.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out / 'runner-source.py').write_bytes(Path(__file__).read_bytes())
        (a.out / 'server-helper-source.py').write_bytes((ROOT / 'tools/compare-runtimes.py').read_bytes())
        configs = H['configurations']()
        names = a.configs.split(',')
        H['save'](a.out / 'inventory.json', H['inventory']())
        H['save'](a.out / 'protocol.json', {'runner_sha256': H['digest'](__file__),
             'server_helper_sha256': H['digest'](ROOT / 'tools/compare-runtimes.py'),
             'cases': CASES, 'configs': names, 'repetitions_per_case_block': a.reps,
             'blocks': a.blocks, 'excluded_warmups_per_case_block': a.warmups,
             'output_budget': 1024, 'eos': 'honored; actual output counts and budget hits reported',
             'cache': 'Unique early reference identifier on every request; raw identical bytes across runtimes',
             'scope': 'Natural short prompts. Full text retained. Different answer lengths are not treated as equal work.'})
        report = {}
        for block in range(a.blocks):
            for name in names if block % 2 == 0 else list(reversed(names)):
                sub = a.out / f'block{block}-{name}'
                print('Executing real tasks', sub.name, flush=True)
                with H['Server'](configs[name], sub) as server:
                    for case_index, case in enumerate(CASES):
                        rows = []
                        for iteration in range(a.warmups + a.reps):
                            serial = case_index * 1000 + block * (a.reps + a.warmups) + iteration
                            prompt = H['chat']('Request reference ' + H['identifier'](serial) +
                                ' (omit this reference from the answer).\n' + case['prompt'])
                            row = server.generate(prompt, limit=1024)
                            row.update(case=case['id'], iteration=iteration, excluded=iteration < a.warmups)
                            H['save'](sub / f'task-{case["id"]}-r{iteration:03}.json', row)
                            if not row['excluded']:
                                rows.append(row)
                        report[sub.name + '-' + case['id']] = {**H['summarise'](rows),
                            'output_budget_hits': sum(r['output_tokens'] >= 1024 for r in rows)}
                        H['save'](a.out / 'report.json', report)
        aggregate = {}
        for name in names:
            for case in CASES:
                rows = []
                for sub in sorted(a.out.glob('block*-' + name)):
                    for f in sorted(sub.glob('task-' + case['id'] + '-r*.json')):
                        row = json.loads(f.read_text())
                        if not row['excluded']:
                            rows.append(row)
                aggregate[name + '-' + case['id']] = {**H['summarise'](rows),
                    'output_budget_hits': sum(r['output_tokens'] >= 1024 for r in rows)}
        H['save'](a.out / 'aggregate.json', aggregate)

if __name__ == '__main__':
    main()
