#!/usr/bin/env python3
"""Plot measured medians/ranges, keeping explicit single/dual profiles visible."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('matrix', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    conditions = json.loads((a.matrix / 'aggregate.json').read_text())['conditions']
    names = {key.rsplit('-d', 1)[0] for key in conditions}
    selected = []
    colors = {'ollama-current': '#bd6b25', 'ollama-compatible': '#926e3a',
              'stock-current': '#3472a6', 'legacy': '#288467'}
    labels = {'ollama-current': 'Ollama 0.40.6 / Vulkan', 'ollama-compatible': 'Ollama 0.11.10 / CUDA',
              'stock-current': 'Stock llama.cpp v0.7.0', 'legacy': 'Abra'}
    decisions = {}
    for family in colors:
        candidates = [name for name in names if name.startswith(family + '-')]
        singles = [name for name in candidates if name.endswith('-secondary')]
        if not singles:
            raise SystemExit('Missing single-GPU profile: ' + family)
        def score(name):
            return sum(conditions[name + '-d' + str(depth)]['elapsed_s']['median'] for depth in [512, 2048, 8192])
        single = min(singles, key=lambda name: (score(name), name))
        if family.startswith('ollama'):
            dual = family + '-tuned-both'
        else:
            dual = min([name for name in candidates if '-both-' in name], key=score)
        selected += [(single, family, False), (dual, family, True)]
        decisions[family] = {'single': single, 'dual': dual,
                             'selection': 'All single-GPU rows use the same physical secondary card; choose its tested profile with lowest sum of median request latency. Ollama dual uses the tuned spread profile.'}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'plot-selection.json').write_text(json.dumps(decisions, indent=2) + '\n')
    for depth in [2048, 8192]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
        y = list(range(len(selected)))
        row_labels = [labels[family] + (' — both visible / spread' if dual and family.startswith('ollama') else ' — 2 GPUs' if dual else ' — secondary GPU')
                      for name, family, dual in selected]
        for ax, (metric, title, unit) in zip(axes, [
                ('backend_decode_tps', 'Generation rate', 'tokens/s · higher is faster'),
                ('ttft_s', 'First content received', 'seconds · lower is faster'),
                ('elapsed_s', 'Full 128-token request', 'seconds · lower is faster')]):
            for i, (name, family, dual) in enumerate(selected):
                entry = conditions[name + '-d' + str(depth)]
                if entry['requests'] != 20 or not entry['all_outputs_128_tokens']:
                    raise SystemExit('Plot requires twenty matched 128-token samples: ' + name)
                values = entry[metric]
                median = values['median']
                ax.barh(i, median, color=colors[family], alpha=.9,
                        hatch='//' if dual else None, edgecolor='white', height=.7)
                ax.errorbar(median, i, xerr=[[median - values['min']], [values['max'] - median]],
                            fmt='none', ecolor='#222222', capsize=3, linewidth=1)
            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.set_xlabel(unit, fontsize=9)
            ax.set_xlim(left=0)
            ax.grid(axis='x', alpha=.18)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right', 'left']].set_visible(False)
        axes[0].set_yticks(y, row_labels, fontsize=9)
        axes[0].invert_yaxis()
        fig.suptitle(f'Qwen3-4B Q4_K_M · {depth:,} input tokens · BigBoy GTX 1080 Ti',
                     x=.02, ha='left', fontsize=14, fontweight='bold')
        fig.text(.02, .015, '20 measured requests/profile. Bars: medians; whiskers: min–max, not confidence intervals. '
                 'Dynamic clocks and temperatures; see the recorded telemetry.\n'
                 'Ollama dual rows request spreading across both visible GPUs; see raw allocation logs. '
                 'Current official Ollama CUDA was not qualified on driver 550.', fontsize=8)
        fig.tight_layout(rect=(0, .09, 1, .94))
        for suffix in ['png', 'svg']:
            metadata = {'Description': json.dumps({
                'host': 'BigBoy', 'driver': '550.163.01', 'cuda_toolkit_for_local_cuda_builds': '12.9.86',
                'gpus': ['0000:03:00.0 GPU-3007f159-e0cf-9303-0a81-41ea8e945315 SM6.1',
                         '0000:81:00.0 GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c SM6.1'],
                'model_sha256': '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5',
                'input_tokens': depth, 'output_tokens': 128, 'kv': 'F16',
                'samples': 20, 'excluded_warmups_per_depth_block': 2, 'blocks': 2,
                'source': 'matrix/aggregate.json and per-server manifests/raw requests',
                'profiles': decisions})}
            fig.savefig(a.out / f'runtime-comparison-{depth}.{suffix}', dpi=180, facecolor='white', metadata=metadata)
        plt.close(fig)

if __name__ == '__main__':
    main()
