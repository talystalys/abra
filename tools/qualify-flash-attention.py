#!/usr/bin/env python3
"""Separate quality qualification; never changes the strict logit comparator."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import runpy
import socket
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
benchmark_helpers = runpy.run_path(str(ROOT / 'tools/port-benchmark.py'))
DEVICES, digest = benchmark_helpers['DEVICES'], benchmark_helpers['digest']
URL = 'https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--gpu', choices=list(DEVICES), default='primary')
p.add_argument('--source', type=Path, default=ROOT / '.deps/llama.cpp-port')
p.add_argument('--chunks', type=int, default=20)
p.add_argument('--probes-only', action='store_true')
a = p.parse_args()
if socket.gethostname() != 'BigBoy' or a.out.exists():
    raise SystemExit('Run on BigBoy with a fresh result directory')
a.out.mkdir(parents=True)
uuid, pci, node = DEVICES[a.gpu]
inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,pci.bus_id,compute_cap', '--format=csv,noheader'], text=True)
row = [r for r in inventory.splitlines() if uuid in r]
if len(row) != 1:
    raise SystemExit('Expected physical GPU missing')
name, _, actual_pci, cc = map(str.strip, row[0].split(','))
if name != 'NVIDIA GeForce GTX 1080 Ti' or cc != '6.1' or actual_pci.lower().lstrip('0') != pci.lower().lstrip('0'):
    raise SystemExit('GPU identity mismatch')
env = os.environ.copy()
for key in list(env):
    if key.startswith('LEGACY_AI_'):
        del env[key]
env.update(CUDA_VISIBLE_DEVICES=uuid, GGML_CUDA_DISABLE_GRAPHS='1',
           LD_LIBRARY_PATH=str(a.source.resolve() / 'build-port/bin') + ':/usr/local/cuda-12.9/lib64')
model = ROOT / 'models/Qwen3-4B-Q4_K_M.gguf'
bins = a.source / 'build-port/bin'
fixture = ROOT / 'tools/quality-probes.json'
probes = json.loads(fixture.read_text())
manifest = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'gpu': row[0],
            'model_sha256': digest(model), 'source_head': subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip(),
            'fixture_sha256': digest(fixture), 'context':2048, 'chunks':a.chunks,
            'runner_sha256':digest(Path(__file__)),
            'prompt_template':'Qwen3 enable_thinking=false assistant prefix',
            'template_source':'https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json',
            'criterion':{'maximum_relative_ppl_regression':0.01,'lost_baseline_probe_successes':0},
            'binaries':{name:digest(bins/name) for name in ['llama-completion','llama-perplexity']},
            'libraries':{path.name:digest(path) for path in bins.glob('*.so') if not path.is_symlink()},
            'environment':{key:env[key] for key in ['CUDA_VISIBLE_DEVICES','GGML_CUDA_DISABLE_GRAPHS','LD_LIBRARY_PATH']}, 'commands':[]}
(a.out/'source.diff').write_text(subprocess.check_output(['git','-C',str(a.source),'diff','HEAD'],text=True))
(a.out/'fixtures.json').write_text(fixture.read_text())
def run(label, binary, args):
    command = ['numactl',f'--cpunodebind={node}',f'--membind={node}',str((bins/binary).resolve()),
               '-m',str(model.resolve()),'-ngl','99','-sm','none','-t','4','-tb','4',
               '-ctk','f16','-ctv','f16','-c','2048','-b','2048','-ub','512',*args]
    manifest['commands'].append({'label':label,'argv':command})
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Executing',label,flush=True)
    with (a.out/(label+'-stdout.txt')).open('w') as out, (a.out/(label+'-stderr.txt')).open('w') as err:
        subprocess.run(command,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,check=True,timeout=3600)
    return (a.out/(label+'-stdout.txt')).read_text() + '\n' + (a.out/(label+'-stderr.txt')).read_text()

result = {'perplexity':{},'probes':{},'limitations':'20 fixed corpus chunks and ten probes do not establish quality equivalence on all tasks.'}
if not a.probes_only:
    data = ROOT / '.deps/quality-data'
    data.mkdir(exist_ok=True,parents=True)
    corpus = data / 'wiki.test.raw'
    if not corpus.exists():
        archive = data / 'wikitext-2-raw-v1.zip'
        urllib.request.urlretrieve(URL,archive)
        with zipfile.ZipFile(archive) as z:
            corpus.write_bytes(z.read('wikitext-2-raw/wiki.test.raw'))
    manifest['corpus'] = {'url':URL,'sha256':digest(corpus),'bytes':corpus.stat().st_size}
    for mode in ['off','on']:
        log = run('perplexity-'+mode,'llama-perplexity',['-fa',mode,'-f',str(corpus),'--chunks',str(a.chunks)])
        matches = re.findall(r'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)',log)
        if len(matches) != 1:
            raise SystemExit('Missing unique perplexity result')
        result['perplexity'][mode] = {'ppl':float(matches[0][0]),'reported_uncertainty':float(matches[0][1])}
    result['relative_ppl_change'] = result['perplexity']['on']['ppl']/result['perplexity']['off']['ppl']-1
for probe in probes:
    prompt = probe['prompt']
    if probe.get('long_context'):
        # Approximately 1500 tokens, containing no answer outside the one record.
        notes = ['Record '+str(i)+': the shelf contains plain paper and a small box.' for i in range(90)]
        notes.insert(43,'The access word is cobalt-orchid-731.')
        prompt += '\n'+'\n'.join(notes)
    chat = '<|im_start|>user\n'+prompt+' /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    prompt_file = a.out/(probe['id']+'-prompt.txt')
    prompt_file.write_text(chat)
    result['probes'][probe['id']] = {}
    for mode in ['off','on']:
        run(probe['id']+'-'+mode,'llama-completion',['-fa',mode,'-f',str(prompt_file),'-no-cnv','--no-display-prompt','--simple-io','--temp','0','--seed','1','-n','128'])
        answer = (a.out/(probe['id']+'-'+mode+'-stdout.txt')).read_text()
        answer = re.sub(r'<think>.*?</think>','',answer,flags=re.S).replace('[end of text]','').strip()
        answer = re.sub(r'^```(?:json)?\s*|\s*```$','',answer).strip()
        if probe['kind'] == 'json':
            try:
                passed = json.loads(answer) == probe['expected']
            except json.JSONDecodeError:
                passed = False
        else:
            passed = answer.casefold().strip(' .\n') == probe['expected'].casefold()
        result['probes'][probe['id']][mode] = {'answer':answer,'passed':passed}
result['scores'] = {mode:sum(v[mode]['passed'] for v in result['probes'].values()) for mode in ['off','on']}
result['lost_successes'] = [key for key,v in result['probes'].items() if v['off']['passed'] and not v['on']['passed']]
result['qualified'] = not result['lost_successes'] and (a.probes_only or result['relative_ppl_change'] <= 0.01)
(a.out/'quality.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,ensure_ascii=False),flush=True)
if not result['qualified']:
    raise SystemExit(1)
