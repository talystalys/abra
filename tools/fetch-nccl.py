#!/usr/bin/env python3
"""Extract pinned NVIDIA packages locally; never install system packages."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
root = Path(__file__).resolve().parent.parent
prefix = root/'.deps/nccl-2.27.7-cuda12'
if prefix.exists(): raise SystemExit('Refusing to overwrite NCCL prefix')
prefix.mkdir(parents=True)
base = 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/'
index = urllib.request.urlopen(base+'Packages.gz',timeout=60).read()
text = gzip.decompress(index).decode()
records = []
for package in ['libnccl2','libnccl-dev']:
 selected = []
 for block in text.split('\n\n'):
  fields = dict(line.split(': ',1) for line in block.splitlines() if ': ' in line and not line.startswith(' '))
  if fields.get('Package')==package and fields.get('Version')=='2.27.7-1+cuda12.9': selected.append(fields)
 if len(selected)!=1: raise SystemExit(f'Pinned NVIDIA package missing or ambiguous: {package}')
 fields=selected[0]; url=base+fields['Filename']; dest=prefix/Path(fields['Filename']).name
 print('Downloading',url,flush=True)
 urllib.request.urlretrieve(url,dest)
 sha=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
 if sha!=fields['SHA256']: raise SystemExit('NVIDIA package SHA256 mismatch')
 subprocess.run(['dpkg-deb','-x',str(dest),str(prefix)],check=True)
 records.append({'package':package,'version':fields['Version'],'url':url,'sha256':sha,'bytes':dest.stat().st_size})
(prefix/'packages.json').write_text(json.dumps({'index_url':base+'Packages.gz','index_sha256':hashlib.sha256(index).hexdigest(),'packages':records},indent=2)+'\n')
print(json.dumps(records),flush=True)
