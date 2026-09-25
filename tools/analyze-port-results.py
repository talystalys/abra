#!/usr/bin/env python3
"""Summarize recorded inference samples; never generate synthetic timings."""
import argparse
import json
from pathlib import Path
import re
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
groups = {}
model_hashes = set()
for directory in sorted(a.directory.iterdir()):
    match = re.fullmatch(r'd(\d+)-block(\d+)-(.+)', directory.name)
    if not match or not (directory / 'summary.json').exists():
        continue
    manifest = json.loads((directory / 'manifest.json').read_text())
    model_hashes.add(manifest['model_sha256'])
    excluded = manifest.get('excluded_full_workload_warmup_samples', 0)
    for row in json.loads((directory / 'raw.json').read_text()):
        key = (int(match[1]), match[3], row['n_prompt'], row['n_gen'])
        entry = groups.setdefault(key, {'samples_ns': [], 'directories': [], 'blocks': []})
        entry['samples_ns'].extend(row['samples_ns'][excluded:])
        entry['directories'].append(str(directory))
        entry['blocks'].append(int(match[2]))
if len(model_hashes) != 1 or not groups:
    raise SystemExit('Missing measurements or inconsistent model identity')
rows = []
for (depth, configuration, prompt, gen), entry in sorted(groups.items()):
    rates = [(prompt + gen) * 1e9 / value for value in entry['samples_ns']]
    rows.append({'depth': depth, 'configuration': configuration, 'prompt': prompt, 'gen': gen,
                 'samples': len(rates), 'median_tps': statistics.median(rates),
                 'min_tps': min(rates), 'max_tps': max(rates),
                 'spread_tps': max(rates) - min(rates), 'stddev_tps': statistics.pstdev(rates),
                 'median_ns': statistics.median(entry['samples_ns']), **entry})
for row in rows:
    baseline = next((x for x in rows if x['configuration'] == 'baseline' and
                     (x['depth'], x['prompt'], x['gen']) == (row['depth'], row['prompt'], row['gen'])), None)
    if baseline:
        row['speedup_vs_baseline'] = row['median_tps'] / baseline['median_tps']
report = {'model_sha256': next(iter(model_hashes)), 'results': rows,
          'note': 'Dynamic clocks. Screening is exploratory; final claims require balanced order and at least twenty measured samples.'}
a.out.write_text(json.dumps(report, indent=2) + '\n')
for row in rows:
    print(f"d{row['depth']} {row['configuration']} pp{row['prompt']} tg{row['gen']}: "
          f"{row['median_tps']:.3f} tokens/sec ({row['samples']} samples), "
          f"{row.get('speedup_vs_baseline', 1):.4f}x baseline")
