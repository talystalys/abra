#!/usr/bin/env python3
"""Aggregate actual HTTP samples across balanced blocks, preserving failures."""
import argparse
import collections
import json
from pathlib import Path
import runpy
import statistics

ROOT = Path(__file__).resolve().parent.parent
helpers = runpy.run_path(str(ROOT / 'tools/compare-runtimes.py'))
summarise = helpers['summarise']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--minimum-samples', type=int, default=20)
    a = p.parse_args()
    groups = collections.defaultdict(list)
    manifests = collections.defaultdict(list)
    failures = {}
    for directory in sorted(a.directory.glob('block*')):
        failure = directory / 'failure.json'
        if failure.exists():
            failures[directory.name] = json.loads(failure.read_text())
        if not (directory / 'manifest.json').exists():
            continue
        manifest = json.loads((directory / 'manifest.json').read_text())
        name = manifest['configuration']['name']
        manifests[name].append(manifest)
        for file in sorted(directory.glob('d*-r*.json')):
            row = json.loads(file.read_text())
            if not row['excluded']:
                row['source'] = str(file.relative_to(a.directory))
                groups[name, row['depth']].append(row)
    report = {'conditions': {}, 'failures': failures}
    for (name, depth), rows in sorted(groups.items()):
        entry = summarise(rows)
        entry['sample_files'] = [x['source'] for x in rows]
        entry['blocks'] = sorted({x['source'].split('/')[0].split('-')[0] for x in rows})
        entry['minimum_sample_requirement_met'] = len(rows) >= a.minimum_samples
        entry['all_outputs_128_tokens'] = all(x['output_tokens'] == 128 for x in rows)
        entry['cached_tokens_reported'] = sorted({x['cached_tokens_reported'] for x in rows if x['cached_tokens_reported'] is not None})
        entry['observed_gpu_uuids'] = sorted({u for m in manifests[name] for u in m.get('observed_gpu_uuids', [])})
        entry['observed_gpu_uuids_per_server_block'] = [m.get('observed_gpu_uuids', []) for m in manifests[name]]
        entry['binary_sha256'] = sorted({m['binary_sha256'] for m in manifests[name]})
        entry['source_commits'] = sorted({m['source_commit'] for m in manifests[name] if 'source_commit' in m})
        temperatures = collections.defaultdict(list)
        clocks = collections.defaultdict(list)
        powers = collections.defaultdict(list)
        memory = collections.defaultdict(list)
        active_temperature = collections.defaultdict(list)
        active_clocks = collections.defaultdict(list)
        active_power = collections.defaultdict(list)
        max_swap = 0
        max_rss = 0
        backends = set()
        for directory in sorted(a.directory.glob('block*-' + name)):
            intervals = [(r['started_monotonic_s'], r['completed_monotonic_s']) for r in rows
                         if r['source'].split('/')[0] == directory.name and 'started_monotonic_s' in r]
            text = (directory / 'server.log').read_text(errors='replace')
            if 'Vulkan' in text:
                backends.add('Vulkan-mentioned-in-log')
            if 'CUDA0 model buffer' in text or 'CUDA0 model buffer' in text.replace('  ', ' '):
                backends.add('CUDA-model-buffer')
            for line in (directory / 'telemetry.jsonl').read_text().splitlines():
                sample = json.loads(line)
                for gpu in sample.get('gpu_csv', '').splitlines():
                    fields = list(map(str.strip, gpu.split(',')))
                    if len(fields) != 7:
                        continue
                    uuid = fields[1]
                    for result, index in [(temperatures, 2), (clocks, 3), (powers, 4), (memory, 5)]:
                        try:
                            result[uuid].append(float(fields[index].split()[0]))
                        except ValueError:
                            pass
                    if uuid in entry['observed_gpu_uuids'] and any(start <= sample['monotonic_s'] <= end for start, end in intervals):
                        for result, index in [(active_temperature, 2), (active_clocks, 3), (active_power, 4)]:
                            try:
                                result[uuid].append(float(fields[index].split()[0]))
                            except ValueError:
                                pass
                rss = sum(int(v.get('VmRSS', '0 kB').split()[0]) for v in sample.get('processes', {}).values())
                max_rss = max(max_rss, rss)
                max_swap = max([max_swap, *[int(v.get('VmSwap', '0 kB').split()[0]) for v in sample.get('processes', {}).values()]])
        entry['telemetry_scope'] = 'Whole server lifetime including startup, warmups and all depths; not depth-specific extrema'
        entry['whole_server_telemetry'] = {uuid: {'temperature_c': [min(temperatures[uuid]), max(temperatures[uuid])],
                                                'sm_mhz': [min(clocks[uuid]), max(clocks[uuid])],
                                                'power_w': [min(powers[uuid]), max(powers[uuid])],
                                                'peak_allocated_mib': max(memory[uuid])} for uuid in temperatures}
        entry['max_sampled_process_tree_rss_kib'] = max_rss
        entry['max_sampled_process_swap_kib'] = max_swap
        entry['measured_request_telemetry'] = {uuid: {
            'samples': len(values), 'median_temperature_c': statistics.median(values),
            'max_temperature_c': max(values), 'median_sm_mhz': statistics.median(active_clocks[uuid]),
            'median_power_w': statistics.median(active_power[uuid])} for uuid, values in active_temperature.items()}
        entry['backend_log_markers'] = sorted(backends)
        report['conditions'][name + '-d' + str(depth)] = entry
    (a.directory / 'aggregate.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['| Configuration | Prompt tokens | Samples | Decode tok/s | TTFT (s) | Total (s) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, entry in report['conditions'].items():
        name, depth = key.rsplit('-d', 1)
        lines.append(f'| {name} | {depth} | {entry["requests"]} | {entry["backend_decode_tps"]["median"]:.2f} | {entry["ttft_s"]["median"]:.3f} | {entry["elapsed_s"]["median"]:.3f} |')
    (a.directory / 'table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if failures:
        print('Recorded failed configurations:', ', '.join(failures))


if __name__ == '__main__':
    main()
