#!/usr/bin/env python3
"""Test moving work from the thermally limited primary to the secondary GPU."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
root=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
rows=[]
for label,ratio in [('control-before','1/1'),('four-five','4/5'),('two-three','2/3'),('three-five','3/5'),('control-after','1/1')]:
 dest=a.out/label
 command=[sys.executable,str(root/'tools/port-benchmark.py'),'--source',str(root/'.deps/llama.cpp-port'),'--binary',str(root/'.deps/llama.cpp-port/build-port/bin/llama-bench'),'--out',str(dest),'--gpu','both','--split-mode','tensor','--allreduce','nccl','--nccl-proof','--tensor-split',ratio,'--flash-attn','on','--depth','2048','--prompt','0','--gen','128','--reps','5','--warmup-samples','5']
 print('Executing',label,ratio,flush=True)
 with (a.out/(label+'-runner.txt')).open('w') as log: result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 row={'label':label,'ratio':ratio,'exit_code':result.returncode}
 if (dest/'summary.json').exists(): row['median_tps']=json.loads((dest/'summary.json').read_text())[0]['median_tps']
 rows.append(row);(a.out/'report.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
controls=[row['median_tps'] for row in rows if row['label'].startswith('control') and 'median_tps' in row]
if len(controls)!=2: raise SystemExit('Missing bracketing control runs')
reference=statistics.mean(controls)
candidates=[row for row in rows if not row['label'].startswith('control') and 'median_tps' in row]
winner=max(candidates,key=lambda row:row['median_tps']) if candidates else None
selection={'reference_mean_of_control_medians':reference,'best_screen':winner,'requires_further_qualification':bool(winner and winner['median_tps']/reference>1.05),'criterion':'Screen gain >5%; still requires balanced final comparison and fixed quality qualification'}
(a.out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print(json.dumps(selection),flush=True)
