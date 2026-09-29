#!/usr/bin/env python3
"""Summarize synchronous diagnostic events; never infer end-to-end throughput."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('log',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=8)
p.add_argument('--hidden',type=int,default=2560,help='Recorded Qwen3-4B hidden width; used to identify decode boundaries')
a=p.parse_args()
groups=defaultdict(list)
records=[]
for line in a.log.read_text().replace('\\n','\n').splitlines():
 if not line.startswith('LEGACY_PROFILE,'): continue
 fields=line.split(',')
 if len(fields)!=9: raise SystemExit('Unexpected profiling record')
 records.append(fields)
# Repetition alone cannot identify decoding: two prompt norms per microbatch
# can create eight records in a four-microbatch prefill. Use decode boundaries.
starts=[i for i,fields in enumerate(records) if fields[1:7]==['norm-0','RMS_NORM','f32',str(a.hidden),'1','1'] and i+1<len(records) and records[i+1][1]=='Qcur-0']
if len(starts)<a.steps: raise SystemExit('Not enough identifiable single-token decode steps')
for fields in records[starts[-a.steps]:]:
 groups[tuple(fields[1:7])].append((float(fields[7]),int(fields[8])))
by_op=defaultdict(lambda:{'gpu_ms':0.0,'wall_us':0,'events':0})
selected=[]
for key,events in groups.items():
 recent=events
 selected.append({'name':key[0],'op':key[1],'source_type':key[2],'source_ne0':key[3],'source_ne1':key[4],'output_ne1':key[5],'record_count':len(events),'selected_gpu_ms':[v[0] for v in recent]})
 row=by_op[(key[1],key[2])];row['gpu_ms']+=sum(v[0] for v in recent);row['wall_us']+=sum(v[1] for v in recent);row['events']+=len(recent)
if not selected: raise SystemExit('No repeated diagnostic events')
report={'steps':a.steps,'decode_boundary_count':len(starts),'method':'Select the final N complete single-token decode phases beginning at norm-0/RMS_NORM/hidden-width x 1 followed by Qcur-0; aggregate all subsequent events by name/op/type/dimensions. Profiling synchronizes every operation and is diagnostic, not throughput evidence.', 'operations':[{'op':key[0],'source_type':key[1],**row} for key,row in sorted(by_op.items(),key=lambda kv:-kv[1]['gpu_ms'])],'selected_series':selected}
a.out.write_text(json.dumps(report,indent=2)+'\n')
for row in report['operations'][:12]:print(json.dumps(row))
