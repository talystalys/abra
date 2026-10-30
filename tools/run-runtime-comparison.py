#!/usr/bin/env python3
"""Run the authorized benchmark campaign sequentially, saving stage decisions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / 'tools/compare-runtimes.py'

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

def run(argv):
    print('Executing:', ' '.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--workloads', type=Path, required=True)
    p.add_argument('--resume-after-quality', action='store_true', help='Reuse completed qualification after preserving a superseded matrix; requires no active matrix directory')
    a = p.parse_args()
    if a.resume_after_quality and (a.out / 'matrix').exists():
        raise RuntimeError('Preserve the previous matrix under a superseded name before resuming')
    a.out.mkdir(parents=True, exist_ok=a.resume_after_quality)
    state = {'status': 'running', 'stage': 'nccl-screen'}
    save(a.out / 'state.json', state)

    def stage(name, configs, extra):
        state['stage'] = name
        save(a.out / 'state.json', state)
        directory = a.out / name
        run([sys.executable, TOOL, '--out', directory, '--stage', name.split('-')[0],
             '--configs', ','.join(configs), '--workloads', a.workloads, *extra])
        return read(directory / 'report.json')

    if a.resume_after_quality:
        selected = read(a.out / 'selection.json')['selected_main_configurations']
        quality = read(a.out / 'quality/report.json')
        ppl = read(a.out / 'perplexity/report.json')['runtimes']
        if len(quality) != len(selected) or any(v.get('probe_responses') != 20 for v in quality.values()):
            raise RuntimeError('Qualification is incomplete')
        if len(ppl) != 7 or any(v['exit_code'] or v['ppl'] is None for v in ppl.values()):
            raise RuntimeError('Perplexity is incomplete')
        save(a.out / 'qualification-reuse.json', {'reason': 'Restart matrix after fixing identifier domains; unchanged weights, runtime libraries and completed quality/corpus requests.', 'selected': selected})
    else:

        nccl_configs = ['stock-pinned-nccl-both-tensor', 'stock-current-nccl-both-tensor']
        nccl = stage('screen-nccl', nccl_configs, ['--depths', '2048', '--warmups', '2', '--reps', '5', '--blocks', '1'])
        initial = read(a.screen / 'report.json')
        screen = {**initial, **nccl}
        selected = []
        decisions = {'criterion': 'Retain both physical single GPUs. For each stock revision, retain the successful dual-GPU winner for generation and the winner for request latency at 2048 tokens; screening numbers are not final claims.',
                     'stock_dual': {}}
        for family in ['stock-pinned', 'stock-current']:
            selected += [family + '-primary', family + '-secondary']
            candidates = [(k.removeprefix('block0-').removesuffix('-d2048'), v) for k, v in screen.items()
                          if k.startswith('block0-' + family + '-') and '-both-' in k and 'backend_decode_tps' in v]
            if not candidates:
                raise RuntimeError('No supported stock dual-GPU configuration for ' + family)
            fastest_decode = max(candidates, key=lambda x: x[1]['backend_decode_tps']['median'])[0]
            fastest_request = min(candidates, key=lambda x: x[1]['elapsed_s']['median'])[0]
            winners = list(dict.fromkeys([fastest_decode, fastest_request]))
            selected += winners
            decisions['stock_dual'][family] = {'generation_winner': fastest_decode, 'request_latency_winner': fastest_request,
                                              'screen_candidates': {k: v for k, v in candidates}}
        selected += ['legacy-primary', 'legacy-secondary', 'legacy-both-tensor', 'port-control-both-tensor']
        for family in ['ollama-current', 'ollama-compatible']:
            for mode in ['default', 'tuned']:
                for gpu in ['primary', 'secondary', 'both']:
                    name = f'{family}-{mode}-{gpu}'
                    if any(k == 'block0-' + name + '-d2048' and 'backend_decode_tps' in v for k, v in screen.items()):
                        selected.append(name)
        decisions['selected_main_configurations'] = selected
        decisions['unsupported_screen_configurations'] = {k: v for k, v in screen.items() if 'error' in v}
        save(a.out / 'selection.json', decisions)

        stage('quality', selected, ['--reps', '2', '--blocks', '1', '--warmups', '0'])
        state['stage'] = 'perplexity'
        save(a.out / 'state.json', state)
        run([sys.executable, ROOT / 'tools/compare-runtime-ppl.py', '--out', a.out / 'perplexity', '--configs', ','.join(selected)])
    stage('matrix', selected, ['--reps', '10', '--warmups', '2', '--blocks', '2'])
    run([sys.executable, ROOT / 'tools/analyze-runtime-comparison.py', a.out / 'matrix'])
    aggregate = read(a.out / 'matrix/aggregate.json')['conditions']
    serving = []
    scores = {}
    for group, prefix in [('ollama', 'ollama-'), ('upstream', 'stock-current-'), ('legacy', 'legacy-')]:
        candidates = [name for name in selected if name.startswith(prefix)]
        valid = []
        for name in candidates:
            rows = [aggregate.get(name + '-d' + str(depth)) for depth in [512, 2048, 8192]]
            if not all(r and r['minimum_sample_requirement_met'] for r in rows):
                continue
            score = sum(r['elapsed_s']['median'] for r in rows)
            valid.append((score, name))
        if not valid:
            raise RuntimeError('No qualified matrix result for serving group ' + group)
        score, winner = min(valid)
        serving.append(winner)
        scores[group] = {'chosen': winner, 'sum_of_median_request_seconds': score,
                         'candidate_scores': {name: value for value, name in valid}}
    save(a.out / 'serving-selection.json', {'criterion': 'Lowest sum of median full request latency at 512, 2048 and 8192 input tokens; equal weight per input length.',
                                          'groups': scores, 'order': serving})
    stage('cache', serving, ['--reps', '10', '--blocks', '2', '--warmups', '2'])
    for parallel in [2, 4]:
        stage(f'concurrency-p{parallel}', serving, ['--parallel', str(parallel), '--reps', '10', '--blocks', '1', '--warmups', '2'])

    stage('soak', serving, ['--seconds', '900', '--blocks', '2', '--warmups', '2'])
    state['stage'] = 'real-tasks'
    save(a.out / 'state.json', state)
    run([sys.executable, ROOT / 'tools/compare-real-workloads.py', '--out', a.out / 'real-tasks', '--configs', ','.join(serving)])
    state.update(status='complete', stage='complete')
    save(a.out / 'state.json', state)

if __name__ == '__main__':
    main()
