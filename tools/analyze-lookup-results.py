#!/usr/bin/env python3
"""Combine recorded decoding repetitions; distinguish token and block latency."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
manifest=json.loads((a.directory/'manifest.json').read_text())
exclude=manifest['excluded_runs_per_block']
groups={}
for command in manifest['commands']:
 label=command['label']; match=re.fullmatch(r'(.+)-block(\d+)-(.+)',label)
 path=a.directory/(label+'.json')
 if not path.exists(): continue
 data=json.loads(path.read_text()); entry=groups.setdefault((match[1],match[3]),{'runs':[],'all_runs':[],'gpu':{},'rss':[],'files':[],'texts':[],'host_windows':[]})
 entry['runs']+=data['runs'][exclude:];entry['all_runs']+=data['runs'];entry['files'].append(path.name)
 entry['texts'].append((a.directory/(label+'.txt')).read_text())
 stderr=(a.directory/(label+'-stderr.txt')).read_text()
 rss=re.search(r'Maximum resident set size \(kbytes\): (\d+)',stderr)
 if rss:entry['rss'].append(int(rss[1]))
 host_file=a.directory/(label+'-host-memory.csv')
 if host_file.exists():
  host=[[float(value) for value in line.split(',')] for line in host_file.read_text().splitlines()[1:]]
  if host:
   duration=max(x[0] for x in host)
   early=[x[2] for x in host if .2*duration<=x[0]<=.3*duration]
   late=[x[2] for x in host if x[0]>=.9*duration]
   entry['host_windows'].append({'file':host_file.name,'samples':len(host),'max_swap_kib':max(x[3] for x in host),'rss_median_20_to_30_percent_kib':statistics.median(early) if early else None,'rss_median_last_10_percent_kib':statistics.median(late) if late else None})
 for line in (a.directory/(label+'-telemetry.csv')).read_text().splitlines():
  fields=[x.strip() for x in line.split(',')]
  if len(fields)!=6 or not fields[1].startswith('GPU-'):continue
  values=entry['gpu'].setdefault(fields[1],{'memory':[],'temperature':[],'clock':[]})
  for name,index in [('memory',4),('temperature',2),('clock',3)]:
   try:values[name].append(float(fields[index].split()[0]))
   except ValueError:pass
def percentile(values):
 values=sorted(values);return values[int((len(values)-1)*.95)]
rows=[]
fixtures={x['id']:x for x in json.loads((Path(__file__).resolve().parent/'quality-probes.json').read_text())}
for (case,config),entry in sorted(groups.items()):
 runs=entry['runs'];tokens=runs[0]['tokens'];rates=[1000*len(x['tokens'])/x['decode_ms'] for x in runs]
 latencies=[v for x in runs for v in x['block_ms']]
 proposed=sum(x['proposed'] for x in runs);accepted=sum(x['accepted'] for x in runs)
 row={'case':case,'config':config,'samples':len(runs),'median_tps':statistics.median(rates),'min_tps':min(rates),'max_tps':max(rates),'spread_tps':max(rates)-min(rates),'generated_tokens':len(tokens),'p95_block_ms':percentile(latencies),'max_block_ms':max(latencies),'blocks_all_single_token':all(v==1 for x in runs for v in x['block_tokens']),'accepted':accepted,'proposed':proposed,'acceptance_fraction':accepted/proposed if proposed else None,'repeat_tokens_identical':all(x['tokens']==tokens for x in entry['all_runs']),'text_repeats_identical':len(set(entry['texts']))==1,'max_rss_kib':max(entry['rss']) if entry['rss'] else None,'gpu':{uuid:{name:{'min':min(v),'max':max(v)} for name,v in values.items() if v} for uuid,values in entry['gpu'].items()},'raw_files':entry['files']}
 row['host_memory_windows']=entry['host_windows']
 baseline=groups.get((case,'baseline'))
 if baseline:row['matches_baseline_tokens']=all(x['tokens']==baseline['runs'][0]['tokens'] for x in entry['all_runs'])
 if case in fixtures:
  fixture=fixtures[case]; successes=[];literal_successes=[]
  for raw_text in entry['texts']:
   value=raw_text.strip()
   if fixture['kind']=='json':
    try:value=json.loads(value)
    except ValueError:pass
   literal_successes.append(value==fixture['expected'])
   # Match the existing qualification scorer; retain literal equality too.
   answer=re.sub(r'<think>.*?</think>','',raw_text,flags=re.S).replace('[end of text]','').strip()
   answer=re.sub(r'^```(?:json)?\s*|\s*```$','',answer).strip()
   if fixture['kind']=='json':
    try:successes.append(json.loads(answer)==fixture['expected'])
    except ValueError:successes.append(False)
   else:successes.append(answer.casefold().strip(' .\n')==fixture['expected'].casefold())
  row['task_success_all_blocks']=all(successes)
  row['task_literal_success_all_blocks']=all(literal_successes)
 if case in ['code-copy','code-edit']:
  multiplier=2 if case=='code-copy' else 3
  expected='\n'.join(f'def transform_{i}(value):\n    adjusted = value + {i}\n    return adjusted * {multiplier}\n' for i in range(24))
  row['complete_code_exact']=all(value.strip()==expected.strip() for value in entry['texts'])
 rows.append(row)
for row in rows:
 baseline=next((x for x in rows if x['case']==row['case'] and x['config']=='baseline'),None)
 if baseline:row['speedup_vs_baseline']=row['median_tps']/baseline['median_tps']
report={'results':rows,'excluded_runs_per_block':exclude,'latency_definition':manifest['latency_definition'],'percentile_method':'sorted[floor((N-1)*0.95)]','task_scoring':'Existing qualification scorer normalizes case and trailing punctuation, strips thinking/fences, and parses JSON; literal matching is additionally reported. Complete code equality remains literal after outer whitespace trimming.','analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'note':'p95 block equals token latency only where every block contains one token; telemetry includes initialization, prefill and excluded runs.'}
a.out.write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print(json.dumps(row))
