import argparse,hashlib,json,subprocess
from pathlib import Path
root=Path('/home/taly/abra-1080ti-port')
p=argparse.ArgumentParser(description='Record isolated runtime installation provenance')
p.add_argument('--out',type=Path,default=root/'bench/results/1080ti-comparison/setup/provenance.json')
a=p.parse_args()
if a.out.exists():raise SystemExit('Refusing to overwrite setup provenance; choose a fresh --out path')
a.out.parent.mkdir(parents=True,exist_ok=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(cmd):return subprocess.check_output(cmd,text=True).strip()
r={'sources':{},'archives':{},'headers':{},'scope':'Isolated build/install metadata. No driver, service, clock or power changes.'}
for name in ['llama.cpp-pinned-stock','llama.cpp-current','llama.cpp-port']:
 p=root/'.deps'/name
 r['sources'][name]={'head':run(['git','-C',str(p),'rev-parse','HEAD']),'status':run(['git','-C',str(p),'status','--short']),'build_caches':{}}
 for build in ['build-comparison','build-comparison-nccl','build-port']:
  cache=p/build/'CMakeCache.txt'
  if cache.exists():
   text=cache.read_text();r['sources'][name]['build_caches'][build]=[l for l in text.splitlines() if l.startswith(('CMAKE_BUILD_TYPE:','CMAKE_CUDA_COMPILER:','CMAKE_CUDA_ARCHITECTURES:','CMAKE_CUDA_FLAGS:','GGML_CUDA_NCCL:','NCCL_LIBRARY:','NCCL_INCLUDE_DIR:'))]
for folder,name in [('ollama-current','release.tar.zst'),('ollama-compatible','release.tgz')]:
 p=root/'.deps'/folder/name;r['archives'][folder]={'file':str(p),'bytes':p.stat().st_size,'sha256':digest(p)}
for p in (root/'.cuda_compat/include').rglob('*'):
 if p.is_file():r['headers'][str(p.relative_to(root))]=digest(p)
a.out.write_text(json.dumps(r,indent=2)+'\n')
