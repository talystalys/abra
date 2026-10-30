#!/usr/bin/env python3
"""Audit recorded comparison evidence without running inference or fabricating data."""
import argparse
import collections
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent.parent
helpers = runpy.run_path(str(ROOT / 'tools/compare-runtimes.py'))

def read(path):
    return json.loads(Path(path).read_text())

def require(value, description):
    if not value:
        raise AssertionError(description)

def audit(directory):
    selected = read(directory / 'selection.json')['selected_main_configurations']
    configs = helpers['configurations']()
    all_uuids = {x[0] for x in helpers['DEVICES']}
    matrix = directory / 'matrix'
    protocol = read(matrix / 'protocol.json')
    require(protocol['runner_sha256'] == helpers['digest'](matrix / 'runner-source.py'), 'Archived matrix runner hash')
    require(protocol['repetitions_per_depth_block'] == 10 and protocol['blocks'] == 2,
            'Final matrix requires two blocks of ten measured requests')
    require(protocol['warmups_per_depth_block'] == 2, 'Expected two excluded warmups per depth/block')
    require(protocol['model_sha256'] == helpers['MODEL_SHA'], 'Model hash')
    workloads = read(directory.parent / 'workloads-v3/workloads.json')
    require(protocol['workloads_sha256'] == helpers['digest'](directory.parent / 'workloads-v3/workloads.json'), 'Fixture hash')
    proof = read(directory.parent / 'workloads-v3/tokenization-proof.json')
    require(proof['numeric_labels_checked'] == 1000 and proof['numeric_label_token_lengths'] == [11], 'Identifier proof')
    for depth, counts in proof['full_prompt_counts_first_24_ids'].items():
        require(len(counts) == 24 and set(counts) == {int(depth)}, 'Fixed full prompt lengths')
    totals = collections.Counter()
    binary_hashes = collections.defaultdict(set)
    library_hashes = collections.defaultdict(lambda: collections.defaultdict(set))
    quality = {}
    matrix_text = {}
    telemetry = {}
    for name in selected:
        c = configs[name]
        expected = all_uuids if c['gpu'] == 'both' else {helpers['DEVICES'][0 if c['gpu'] == 'primary' else 1][0]}
        for block in range(2):
            sub = matrix / f'block{block}-{name}'
            require(not (sub / 'failure.json').exists(), 'Failed matrix configuration: ' + str(sub))
            manifest = read(sub / 'manifest.json')
            observed = set(manifest['observed_gpu_uuids'])
            require(observed and observed <= expected, 'Actual physical GPU placement: ' + str(sub))
            if c['kind'] == 'llama' and c['gpu'] == 'both':
                require(observed == expected, 'Both stock/Legacy GPUs must be observed')
            require(manifest['model_sha256'] == helpers['MODEL_SHA'], 'Per-server model hash')
            if c['kind'] == 'ollama':
                require(helpers['MODEL_SHA'] in manifest['model_show'].get('modelfile', ''), 'Ollama must import the exact content-addressed GGUF')
            require(any(manifest['process_mappings'].values()), 'Resolved runtime mappings missing')
            binary_hashes[name].add(manifest['binary_sha256'])
            for library, sha in manifest.get('libraries', {}).items():
                library_hashes[name][library].add(sha)
            for mappings in manifest['process_mappings'].values():
                for library, sha in mappings.items():
                    library_hashes[name][library].add(sha)
            if c['family'].startswith('stock'):
                require(not (sub / 'source.diff').read_text().strip(), 'Stock source must remain clean')
            for depth_index, depth in enumerate([512, 2048, 8192]):
                records = sorted(sub.glob(f'd{depth}-r*.json'))
                require(len(records) == 12, 'Twelve raw requests per depth/block')
                seen = set()
                for iteration, path in enumerate(records):
                    row = read(path)
                    require(row['excluded'] == (iteration < 2), 'Warmup selection')
                    serial = depth_index * 1000 + block * 12 + iteration
                    require(row['request']['prompt'] == helpers['fresh_prompt'](workloads, depth, serial), 'Identical cross-runtime input bytes')
                    require(row['request']['prompt'] not in seen, 'Fresh request repeated an identifier')
                    seen.add(row['request']['prompt'])
                    require(row['expected_input_tokens'] == depth, 'Actual input length definition')
                    total = row['prompt_tokens_total_reported'] if c['kind'] == 'ollama' else row['prompt_tokens_evaluated']
                    require(total == depth, 'Runtime truncated or retokenized input')
                    if c['kind'] == 'ollama' and row['cached_tokens_reported'] is not None:
                        require(row['cached_tokens_reported'] <= workloads[str(depth)][0]['identifier_end_token_index'], 'Long-prefix reuse in fresh matrix')
                    require(row['output_tokens'] == 128, 'Main output length must be 128 for a matched comparison')
                    for metric in ['backend_decode_tps', 'ttft_s', 'elapsed_s', 'decode_s', 'prompt_s']:
                        require(row[metric] > 0 and math.isfinite(row[metric]), 'Finite positive timing: ' + metric)
                    if not row['excluded']:
                        totals['measured_requests'] += 1
                    else:
                        totals['excluded_warmups'] += 1
                    matrix_text[name, depth, serial] = row['text']
            totals['server_blocks'] += 1
        require(len(binary_hashes[name]) == 1, 'Runtime binary changed between blocks')
        require(all(len(hashes) == 1 for hashes in library_hashes[name].values()), 'A resolved library or executable changed between blocks')
        qdir = directory / 'quality' / ('block0-' + name)
        require(not (qdir / 'failure.json').exists(), 'Quality server failed')
        probes = collections.defaultdict(list)
        complete = collections.defaultdict(list)
        for file in sorted(qdir.glob('quality-*.json')):
            row = read(file)
            probes[row['fixture']].append(row)
        require(len(probes) == 10 and all(len(rows) == 2 for rows in probes.values()), 'Ten probes repeated twice')
        for file in qdir.glob('complete-*.json'):
            case = file.name.rsplit('-', 1)[0].removeprefix('complete-')
            complete[case].append(read(file))
        require(len(complete) == 3 and all(len(rows) == 2 for rows in complete.values()), 'Three full-response tasks repeated twice')
        quality[name] = {'probe_score_per_repeat': [sum(rows[i]['correct'] for rows in probes.values()) for i in range(2)],
                         'probe_repeat_consistent': all(len({r['text'] for r in rows}) == 1 for rows in probes.values()),
                         'full_response_repeat_consistent': {case: len({r['text'] for r in rows}) == 1 for case, rows in complete.items()},
                         'successful_probe_ids': sorted(k for k, rows in probes.items() if all(r['correct'] for r in rows)),
                         'full_response_output_tokens': {case: [r['output_tokens'] for r in rows] for case, rows in complete.items()}}
    require(totals['measured_requests'] == len(selected) * 3 * 20, 'Twenty measured samples per condition')
    serving = read(directory / 'serving-selection.json')['order']
    soaks = {}
    concurrency = {}
    for name in serving:
        c = configs[name]
        soaks[name] = {}
        for block in range(2):
            sub = directory / 'soak' / f'block{block}-{name}'
            require(not (sub / 'failure.json').exists(), 'Sustained configuration failed')
            rows = [read(f) for f in sorted(sub.glob('soak-*.json'))]
            require(rows and rows[-1]['since_start_s'] >= 900, 'At least 15 minutes per sustained block')
            require(len({r['request']['prompt'] for r in rows}) == len(rows), 'Sustained fresh prompts must be unique')
            for r in rows:
                require(r['output_tokens'] == 128, 'Sustained output budget')
                if c['kind'] == 'ollama' and r['cached_tokens_reported'] is not None:
                    require(r['cached_tokens_reported'] <= workloads['2048'][0]['identifier_end_token_index'], 'Sustained long-prefix cache reuse')
            soaks[name][str(block)] = {'requests': len(rows), 'elapsed_s': rows[-1]['since_start_s'],
                                      'tokens': sum(r['output_tokens'] for r in rows)}
        for parallel in [2, 4]:
            sub = directory / f'concurrency-p{parallel}' / ('block0-' + name)
            require(not (sub / 'failure.json').exists(), 'Concurrency configuration failed')
            paths = sorted(sub.glob('batch-*.json'))
            require(len(paths) == parallel * 10, 'Concurrency request count')
            matches = []
            for file in paths:
                row = read(file)
                matches += [row['text'] == value for (runtime, depth, serial), value in matrix_text.items()
                            if runtime == name and depth == 2048 and row['request']['prompt'] == helpers['fresh_prompt'](workloads, depth, serial)]
            concurrency[name + f'-p{parallel}'] = {'requests': len(paths), 'serial_comparisons': len(matches),
                                                 'matching_serial_responses': sum(matches)}
        for block in range(2):
            sub = directory / 'cache' / f'block{block}-{name}'
            require(not (sub / 'failure.json').exists(), 'Cached-history configuration failed')
            rows = [read(f) for f in sub.glob('cached-*.json')]
            require(len(rows) == 10, 'Cached-history request count')
            require(len({r['request']['prompt'] for r in rows}) == 1, 'Cached-history prompt definition')
    natural = read(directory / 'real-tasks/aggregate.json')
    natural_rows = collections.defaultdict(dict)
    for name in serving:
        for case_index, case in enumerate(['code', 'reasoning', 'prose']):
            require(natural[name + '-' + case]['requests'] == 20, 'Twenty natural-task samples')
            for block in range(2):
                sub = directory / 'real-tasks' / f'block{block}-{name}'
                files = sorted(sub.glob('task-' + case + '-r*.json'))
                require(len(files) == 12, 'Natural-task raw sample count')
                for iteration, file in enumerate(files):
                    row = read(file)
                    require(row['excluded'] == (iteration < 2), 'Natural-task warmup selection')
                    key = case, block, iteration
                    natural_rows[key][name] = row['request']['prompt']
                    require(0 < row['output_tokens'] <= 1024, 'Natural task output count')
    require(all(len(values) == len(serving) and len(set(values.values())) == 1 for values in natural_rows.values()), 'Identical natural-task input bytes across runtimes')
    for path in directory.glob('*/block*/telemetry.jsonl'):
        if path.parents[1].name.startswith('matrix-superseded'):
            continue
        max_swap = 0
        sample_errors = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            sample_errors += 'error' in row
            max_swap = max([max_swap, *[int(v.get('VmSwap', '0 kB').split()[0]) for v in row.get('processes', {}).values()]])
        telemetry[str(path.relative_to(directory))] = {'sample_errors': sample_errors, 'max_sampled_process_swap_kib': max_swap}
    ppl = read(directory / 'perplexity/report.json')
    require(all(v['exit_code'] == 0 and v['ppl'] is not None and math.isfinite(v['ppl']) for v in ppl['runtimes'].values()), 'Fixed-corpus PPL runs completed')
    result = {'audit_completed': True, 'matrix_counts': dict(totals), 'configurations': len(selected),
              'quality': quality, 'perplexity': ppl, 'sustained': soaks,
              'concurrent_response_comparisons': concurrency, 'telemetry': telemetry,
              'natural_tasks': natural,
              'limitations': ['Task/PPL checks do not establish universal quality equivalence.',
                              'Concurrency response differences are reported; they are not silently reclassified as exact-output successes.',
                              'Sustained time is two opposing 15-minute blocks per runtime, not one continuous 30-minute run.',
                              'Current official Ollama uses Vulkan on driver 550; older compatible Ollama uses CUDA.']}
    helpers['save'](directory / 'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k in ['audit_completed', 'matrix_counts', 'configurations']}, indent=2))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    audit(a.directory)

if __name__ == '__main__':
    main()
