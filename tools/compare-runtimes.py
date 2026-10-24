#!/usr/bin/env python3
"""Reproducible local HTTP comparisons; run only on the identified BigBoy GPUs."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / 'models/Qwen3-4B-Q4_K_M.gguf'
MODEL_SHA = '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
DEVICES = [('GPU-3007f159-e0cf-9303-0a81-41ea8e945315', '03:00.0', 0),
           ('GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c', '81:00.0', 1)]
PORT = 18174
CTX = 10240


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.new')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def output(argv, **kwargs):
    return subprocess.check_output(list(map(str, argv)), text=True, **kwargs)


def api(path, data=None, timeout=300):
    conn = http.client.HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    try:
        conn.request('GET' if data is None else 'POST', path,
                     body=None if data is None else json.dumps(data).encode(),
                     headers={'Content-Type': 'application/json'})
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f'{path}: HTTP {response.status}: {body[:1500]!r}')
        return json.loads(body)
    finally:
        conn.close()


def chat(text):
    return '<|im_start|>user\n' + text + ' /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def identifier(serial):
    return '|'.join(f'{serial:06}'[::-1])


def fresh_prompt(workloads, depth, serial):
    # Delimit individual digits to avoid variable-length numeric merges.
    # Reverse digits so the first identifier token changes on adjacent requests.
    if not 0 <= serial < 1000000:
        raise ValueError('Unique document identifier space exhausted')
    label = identifier(serial)
    template = workloads[str(depth)][0]['prompt']
    if 'Document 0|0|0|0|0|0:' not in template:
        raise ValueError('Regenerate workloads with delimited document identifiers')
    return template.replace('Document 0|0|0|0|0|0:', 'Document ' + label + ':', 1)


def configurations():
    configs = {}
    for family, source in [('stock-pinned', 'llama.cpp-pinned-stock'),
                           ('stock-current', 'llama.cpp-current'),
                           ('legacy', 'llama.cpp-port'),
                           ('port-control', 'llama.cpp-port')]:
        for gpu in ['primary', 'secondary', 'both']:
            if family == 'port-control' and gpu != 'both':
                continue
            for split in (['layer', 'row', 'tensor'] if gpu == 'both' and family.startswith('stock') else
                          ['tensor'] if gpu == 'both' else ['none']):
                name = f'{family}-{gpu}' + ('-' + split if gpu == 'both' else '')
                configs[name] = {'name': name, 'kind': 'llama', 'family': family,
                                 'source': source, 'gpu': gpu, 'split': split}
    for family, folder in [('ollama-current', 'ollama-current'), ('ollama-compatible', 'ollama-compatible')]:
        for mode in ['default', 'tuned']:
            for gpu in ['primary', 'secondary', 'both']:
                name = f'{family}-{mode}-{gpu}'
                configs[name] = {'name': name, 'kind': 'ollama', 'family': family,
                                 'folder': folder, 'mode': mode, 'gpu': gpu}
    configs['ollama-current-cuda-secondary'] = {**configs['ollama-current-default-secondary'],
        'name': 'ollama-current-cuda-secondary', 'force_cuda': True}
    for revision in ['pinned', 'current']:
        name = f'stock-{revision}-nccl-both-tensor'
        configs[name] = {**configs[f'stock-{revision}-both-tensor'], 'name': name,
                         'family': f'stock-{revision}-nccl', 'build': 'build-comparison-nccl', 'nccl': True}
    return configs


def gpu_rows():
    return output(['nvidia-smi', '--query-gpu=name,uuid,pci.bus_id,compute_cap,driver_version,power.limit',
                   '--format=csv,noheader'])


def inventory():
    rows = gpu_rows()
    for uuid, pci, _ in DEVICES:
        found = [list(map(str.strip, x.split(','))) for x in rows.splitlines() if uuid in x]
        if len(found) != 1 or found[0][0] != 'NVIDIA GeForce GTX 1080 Ti' or found[0][3] != '6.1' or not found[0][2].endswith(pci):
            raise RuntimeError('Physical GPU identity mismatch')
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'host': socket.gethostname(),
            'gpus': rows, 'uname': output(['uname', '-a']), 'lscpu': output(['lscpu']),
            'nvcc': output(['/usr/local/cuda-12.9/bin/nvcc', '--version']),
            'driver': output(['nvidia-smi', '-q']), 'clocks': 'Dynamic; no clock, power, driver or service changes.'}


class Server:
    def __init__(self, config, directory, parallel=1):
        self.c = config
        self.directory = directory
        self.parallel = parallel
        self.process = None
        self.stop = threading.Event()
        self.telemetry = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        # Never accept a readiness response from an unrelated server.
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', PORT)) == 0:
                raise RuntimeError(f'Benchmark port {PORT} is already occupied')
        busy = output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name', '--format=csv,noheader']).strip()
        if busy:
            raise RuntimeError('GPU compute processes already active before server startup: ' + busy)
        c = self.c
        ids = [0, 1] if c['gpu'] == 'both' else [0 if c['gpu'] == 'primary' else 1]
        node = DEVICES[ids[0]][2]
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('LEGACY_AI_', 'NCCL_', 'OLLAMA_', 'GGML_'))}
        self.env['CUDA_VISIBLE_DEVICES'] = ','.join(DEVICES[i][0] for i in ids)
        # Vulkan enumerates independently of CUDA. Verify actual placement below.
        self.env['GGML_VK_VISIBLE_DEVICES'] = ','.join(str(i) for i in ids)
        if c['kind'] == 'llama':
            self.source = ROOT / '.deps' / c['source']
            bins = self.source / ('build-port/bin' if c['family'] in ['legacy', 'port-control'] else c.get('build', 'build-comparison') + '/bin')
            binary = bins / 'llama-server'
            if not binary.exists():
                raise RuntimeError('Missing server binary: ' + str(binary))
            self.env.update(LD_LIBRARY_PATH=str(bins) + ':/usr/local/cuda-12.9/lib64')
            if c['family'] in ['legacy', 'port-control']:
                self.env.update(LEGACY_AI_SPLIT_TIED_OUTPUT='1' if c['family'] == 'legacy' else '0',
                                LEGACY_AI_REUSE_Q8='1' if c['family'] == 'legacy' else '0',
                                GGML_CUDA_DISABLE_GRAPHS='1', GGML_CUDA_ALLREDUCE='nccl' if len(ids) == 2 else 'none')
            elif c.get('nccl'):
                self.env['GGML_CUDA_ALLREDUCE'] = 'nccl'
            self.argv = ['numactl', f'--cpunodebind={node}', f'--membind={node}', str(binary),
                         '-m', str(MODEL), '--host', '127.0.0.1', '--port', str(PORT),
                         '-ngl', '99', '-sm', c['split'], '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16',
                         '-c', str(CTX * self.parallel), '-np', str(self.parallel), '-t', '4', '-tb', '4',
                         '-b', '2048', '-ub', '512']
            if len(ids) == 2:
                self.argv += ['-ts', '1,1']
            metadata = {'source_commit': output(['git', '-C', self.source, 'rev-parse', 'HEAD']).strip(),
                        'binary_sha256': digest(binary),
                        'libraries': {x.name: digest(x) for x in bins.glob('*.so*') if x.is_file() and not x.is_symlink()}}
            (self.directory / 'source.diff').write_text(output(['git', '-C', self.source, 'diff', 'HEAD']))
            ldd = output(['ldd', binary], env=self.env)
            (self.directory / 'ldd.txt').write_text(ldd)
            for line in ldd.splitlines():
                if any(x in line for x in ['libllama', 'libggml']) and '=>' in line:
                    resolved = Path(line.split('=>')[1].split()[0]).resolve()
                    if not resolved.is_relative_to(bins.resolve()):
                        raise RuntimeError('Wrong runtime library: ' + str(resolved))
        else:
            self.source = ROOT / '.deps' / c['folder']
            binary = self.source / 'bin/ollama'
            if not binary.exists():
                raise RuntimeError('Missing Ollama binary: ' + str(binary))
            self.env.update(OLLAMA_HOST=f'127.0.0.1:{PORT}', OLLAMA_MODELS=str(self.source / 'models'),
                            OLLAMA_NO_CLOUD='1', OLLAMA_NUM_PARALLEL=str(self.parallel),
                            OLLAMA_CONTEXT_LENGTH=str(CTX), OLLAMA_KEEP_ALIVE='-1',
                            OLLAMA_LOAD_TIMEOUT='120s', OLLAMA_DEBUG='0')
            if c.get('force_cuda'):
                self.env.update(OLLAMA_VULKAN='0', GGML_VK_VISIBLE_DEVICES='-1')
            if c['mode'] == 'tuned':
                self.env.update(OLLAMA_FLASH_ATTENTION='1', OLLAMA_KV_CACHE_TYPE='f16')
                if len(ids) == 2:
                    self.env['OLLAMA_SCHED_SPREAD'] = '1'
            self.options = {'num_ctx': CTX, 'num_predict': 128, 'temperature': 0,
                            'seed': 0, 'repeat_penalty': 1}
            if c['mode'] == 'tuned':
                self.options.update(num_batch=2048, num_thread=4)
            self.argv = ['numactl', f'--cpunodebind={node}', f'--membind={node}', str(binary), 'serve']
            metadata = {'binary_sha256': digest(binary), 'release_folder': str(self.source)}
        self.manifest = {'configuration': c, 'parallel': self.parallel, 'argv': self.argv,
                         'environment': {k: v for k, v in self.env.items() if k.startswith(('LEGACY_AI_', 'GGML_', 'OLLAMA_', 'CUDA_VISIBLE', 'LD_LIBRARY'))},
                         'model_sha256': MODEL_SHA, **metadata}
        save(self.directory / 'manifest.json', self.manifest)
        self.log = (self.directory / 'server.log').open('w')
        began = time.monotonic()
        try:
            self.process = subprocess.Popen(self.argv, env=self.env, stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            self.telemetry = threading.Thread(target=self.sample, daemon=True)
            self.telemetry.start()
            ready_path = '/api/version' if c['kind'] == 'ollama' else '/health'
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError('Server exited: ' + str(self.process.returncode))
                try:
                    response = api(ready_path, timeout=2)
                    if response.get('status') != 'loading model':
                        break
                except (OSError, RuntimeError, ValueError):
                    pass
                if time.monotonic() - began > 180:
                    raise TimeoutError('Server readiness timeout')
                time.sleep(.2)
            if c['kind'] == 'ollama':
                self.manifest['version'] = response
                modelfile = self.directory / 'Modelfile'
                modelfile.write_text('FROM ' + str(MODEL) + '\n')
                with (self.directory / 'import.log').open('w') as log:
                    subprocess.run([str(binary), 'create', 'legacy-ai-benchmark', '-f', str(modelfile)],
                                   env=self.env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
                self.manifest['model_show'] = api('/api/show', {'model': 'legacy-ai-benchmark'})
                # Loading is excluded from warm measurements; retain the full load response.
                save(self.directory / 'load.json', api('/api/generate', {'model': 'legacy-ai-benchmark',
                      'prompt': '', 'stream': False, 'keep_alive': -1, 'options': self.options}))
                ps = api('/api/ps')
                self.manifest['loaded_models'] = ps
                models = ps.get('models', [])
                if not models or not models[0].get('size_vram', 0):
                    raise RuntimeError('Ollama CPU fallback: no GPU model allocation')
            self.manifest['startup_and_load_s'] = time.monotonic() - began
            self.manifest['process_mappings'] = self.mappings()
            self.manifest['active_gpu_processes'] = output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv'])
            own_pids = set(self.descendants())
            allocated = set()
            for line in self.manifest['active_gpu_processes'].splitlines()[1:]:
                fields = list(map(str.strip, line.split(',')))
                if fields and fields[0].isdigit() and int(fields[0]) in own_pids:
                    allocated.add(fields[1])
            self.manifest['observed_gpu_uuids'] = sorted(allocated)
            if not allocated:
                raise RuntimeError('No physical GPU allocation observed for the server process tree')
            if allocated - {DEVICES[i][0] for i in ids}:
                raise RuntimeError('Runtime allocated a GPU outside the requested physical device set')
            save(self.directory / 'manifest.json', self.manifest)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def descendants(self):
        if self.process is None:
            return []
        pids = [self.process.pid]
        seen = set(pids)
        for pid in pids:
            # Go and the HTTP server can fork from any thread. Linux exposes
            # children per task, so reading only the main task misses runners.
            for children in Path(f'/proc/{pid}/task').glob('*/children'):
                try:
                    for child in map(int, children.read_text().split()):
                        if child not in seen:
                            seen.add(child)
                            pids.append(child)
                except FileNotFoundError:
                    pass
        return pids

    def mappings(self):
        result = {}
        for pid in self.descendants():
            try:
                paths = sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines()
                                if '/' in line and any(x in line for x in ['libggml', 'libllama', 'libnccl', 'libcuda', 'libcublas', 'libvulkan', 'libnvidia'])})
                executable = Path(f'/proc/{pid}/exe').resolve()
                if executable.is_file():
                    paths.append(str(executable))
                result[str(pid)] = {path: digest(path) for path in paths if Path(path).is_file()}
            except FileNotFoundError:
                pass
        return result

    def sample(self):
        with (self.directory / 'telemetry.jsonl').open('w') as f:
            while not self.stop.is_set():
                try:
                    rows = output(['nvidia-smi', '--query-gpu=timestamp,uuid,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu', '--format=csv,noheader'], timeout=5)
                    rss = {}
                    for pid in self.descendants():
                        try:
                            status = Path(f'/proc/{pid}/status').read_text().splitlines()
                            rss[str(pid)] = {s.split(':')[0]: s.split(':', 1)[1].strip() for s in status if s.startswith(('Name:', 'VmRSS:', 'VmSwap:'))}
                        except FileNotFoundError:
                            pass
                    f.write(json.dumps({'monotonic_s': time.monotonic(), 'gpu_csv': rows, 'processes': rss,
                                        'loadavg': Path('/proc/loadavg').read_text().strip()}) + '\n')
                    f.flush()
                except Exception as e:
                    f.write(json.dumps({'error': str(e)}) + '\n')
                self.stop.wait(1)

    def __exit__(self, *args):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        self.stop.set()
        if self.telemetry:
            self.telemetry.join(timeout=10)
        if hasattr(self, 'log'):
            self.log.close()

    def generate(self, prompt, limit=128, cache=False):
        if self.c['kind'] == 'ollama':
            path = '/api/generate'
            request = {'model': 'legacy-ai-benchmark', 'prompt': prompt, 'raw': True, 'stream': True,
                       'keep_alive': -1, 'options': {**self.options, 'num_predict': limit}}
        else:
            path = '/completion'
            request = {'prompt': prompt, 'n_predict': limit, 'temperature': 0, 'seed': 0,
                       'repeat_penalty': 1, 'cache_prompt': cache, 'stream': True, 'return_tokens': True}
        conn = http.client.HTTPConnection('127.0.0.1', PORT, timeout=300)
        began = time.monotonic()
        chunks = []
        first = None
        final = None
        try:
            conn.request('POST', path, json.dumps(request).encode(), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status}: {response.read()[:1500]!r}')
            while line := response.readline():
                line = line.strip()
                if not line or line == b'data: [DONE]':
                    continue
                if line.startswith(b'data: '):
                    line = line[6:]
                obj = json.loads(line)
                now = time.monotonic()
                if obj.get('error'):
                    raise RuntimeError(str(obj['error']))
                content = obj.get('response', '') + obj.get('thinking', '') if self.c['kind'] == 'ollama' else obj.get('content', '')
                if content and first is None:
                    first = now - began
                chunks.append({'at_s': now - began, 'data': obj})
                if obj.get('done') or obj.get('stop'):
                    final = obj
            elapsed = time.monotonic() - began
        finally:
            conn.close()
        if final is None:
            raise RuntimeError('Stream ended without completion record')
        if self.c['kind'] == 'ollama':
            n = final['eval_count']
            decode_s = final['eval_duration'] / 1e9
            prompt_n = final.get('prompt_eval_count')
            prompt_s = final.get('prompt_eval_duration', 0) / 1e9
            cached = final.get('prompt_eval_cached_count')
            total_prompt_n = prompt_n
            if cached is not None:
                prompt_n -= cached
            text = ''.join(x['data'].get('response', '') + x['data'].get('thinking', '') for x in chunks)
        else:
            timings = final['timings']
            n, decode_s = timings['predicted_n'], timings['predicted_ms'] / 1000
            prompt_n, prompt_s = timings['prompt_n'], timings['prompt_ms'] / 1000
            cached = final.get('tokens_cached')
            total_prompt_n = None
            text = ''.join(x['data'].get('content', '') for x in chunks)
        return {'request': request, 'chunks': chunks, 'text': text, 'output_tokens': n,
                'started_monotonic_s': began, 'completed_monotonic_s': began + elapsed,
                'prompt_tokens_evaluated': prompt_n, 'cached_tokens_reported': cached,
                'prompt_tokens_total_reported': total_prompt_n,
                'decode_s': decode_s, 'prompt_s': prompt_s, 'ttft_s': first,
                'elapsed_s': elapsed, 'backend_decode_tps': n / decode_s if decode_s else None,
                'native_decode_tps': final.get('timings', {}).get('predicted_per_second'),
                'backend_prefill_tps': prompt_n / prompt_s if prompt_s and prompt_n else None,
                'client_output_tps': n / elapsed if elapsed else None}


def summarise(rows):
    result = {'requests': len(rows)}
    for name in ['backend_decode_tps', 'backend_prefill_tps', 'ttft_s', 'elapsed_s', 'output_tokens', 'prompt_tokens_evaluated']:
        v = sorted(x[name] for x in rows if x.get(name) is not None)
        if v:
            result[name] = {'median': statistics.median(v), 'min': min(v), 'max': max(v),
                            'spread': max(v) - min(v), 'p95': v[max(0, math.ceil(.95 * len(v)) - 1)]}
    return result


def make_workloads(reference):
    # Put the changing identifier at the start to prevent reuse of long prefixes.
    # All runtimes receive identical bytes. Sizes are verified using stock tokenization.
    suffix = '\nContinue with at least 250 numbered examples of everyday objects and their uses. Write a long list without introductory remarks.'
    workloads = {}
    for target in [512, 2048, 8192]:
        prompts = []
        for variant in range(2):
            base = 'Document ' + identifier(variant) + ':\n'
            words = ('A wooden box holds paper clips and plain cards. ' * 1600).split()
            low, high = 1, len(words)
            while low < high:
                mid = (low + high + 1) // 2
                prompt = chat(base + ' '.join(words[:mid]) + suffix)
                count = len(api('/tokenize', {'content': prompt, 'add_special': True})['tokens'])
                if count <= target:
                    low = mid
                else:
                    high = mid - 1
            prompt = chat(base + ' '.join(words[:low]) + suffix)
            token_ids = api('/tokenize', {'content': prompt, 'add_special': True})['tokens']
            prefix = prompt.split('Document ', 1)[0] + 'Document ' + identifier(variant) + ':'
            end = len(api('/tokenize', {'content': prefix, 'add_special': True})['tokens'])
            prompts.append({'prompt': prompt, 'reference_token_ids': token_ids, 'reference_token_count': len(token_ids),
                            'identifier_end_token_index': end})
        workloads[str(target)] = prompts
    return workloads


def execute(a):
    if socket.gethostname() != 'BigBoy':
        raise RuntimeError('Run on BigBoy')
    if digest(MODEL) != MODEL_SHA:
        raise RuntimeError('Model hash mismatch')
    configs = configurations()
    names = a.configs.split(',')
    if not set(names) <= configs.keys():
        raise RuntimeError('Unknown configuration names')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'runner-source.py').write_bytes(Path(__file__).read_bytes())
    save(a.out / 'inventory.json', inventory())
    save(a.out / 'protocol.json', {'runner_sha256': digest(__file__), 'stage': a.stage,
         'configs': [configs[n] for n in names], 'warmups_per_depth_block': a.warmups,
         'repetitions_per_depth_block': a.reps, 'blocks': a.blocks, 'context_per_slot': CTX,
         'target_output_tokens': 128, 'eos': 'honored, actual output count reported',
         'streaming': 'TTFT is first nonempty content chunk; chunk intervals are not token latency',
         'decode_rate_definition': 'API output token count / API generation duration. Native llama.cpp rate is retained separately because it may exclude the first token. Client total latency is independent of backend timing conventions.',
         'telemetry': 'Approximately 1 Hz; monotonic timestamps shared with request start/end records. GPU allocation, clocks, temperature, power, process-tree RSS/swap and host load.',
         'cache': 'Unique six-digit early document ID for every fresh request, including warmups; repeated-history requests separately labeled. Multiple cached prefixes must not turn the fresh table into cached prefill.',
         'identifier_domains': 'Matrix input-length positions use disjoint 1000-ID ranges. Concurrency uses range 1000 to overlap the 2048-token serial matrix for response comparisons.',
         'cached_field_semantics': 'Ollama prompt_eval_cached_count means reused input tokens when exposed; llama.cpp tokens_cached is final KV state length, not a cache-hit count. Inspect evaluated input counts and server logs.',
         'soak_seconds_per_configuration': a.seconds, 'model_sha256': MODEL_SHA})
    report = {}
    if a.stage == 'workloads':
        with Server(configs[names[0]], a.out / 'reference-server') as server:
            prepared = make_workloads(server)
            save(a.out / 'workloads.json', prepared)
            def check_label(i):
                ids = api('/tokenize', {'content': identifier(i), 'add_special': False})['tokens']
                return len(ids)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                label_lengths = list(pool.map(check_label, range(1000)))
            save(a.out / 'identifier-lengths.json', label_lengths)
            if len(set(label_lengths)) != 1:
                raise RuntimeError('Document identifiers do not have stable token lengths')
            counts = {depth: [len(api('/tokenize', {'content': fresh_prompt(prepared, depth, i), 'add_special': True})['tokens'])
                              for i in range(24)] for depth in prepared}
            if any(set(values) != {prepared[depth][0]['reference_token_count']} for depth, values in counts.items()):
                raise RuntimeError('Unique fresh prompts changed workload token length')
            save(a.out / 'tokenization-proof.json', {'numeric_labels_checked': 1000,
                 'numeric_label_token_lengths': sorted(set(label_lengths)), 'full_prompt_counts_first_24_ids': counts})
        return
    workloads = json.loads(a.workloads.read_text()) if a.workloads else None
    if workloads:
        shared = {}
        for depth, variants in workloads.items():
            n = 0
            for tokens in zip(*(v['reference_token_ids'] for v in variants)):
                if len(set(tokens)) != 1:
                    break
                n += 1
            shared[depth] = n
        protocol_path = a.out / 'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        protocol['shared_initial_tokens_between_reference_variants'] = shared
        protocol['workloads_sha256'] = digest(a.workloads)
        save(protocol_path, protocol)
    for block in range(a.blocks):
        order = names if block % 2 == 0 else list(reversed(names))
        for name in order:
            c = configs[name]
            label = f'block{block}-{name}'
            directory = a.out / label
            print('Executing', a.stage, label, flush=True)
            try:
                with Server(c, directory, parallel=a.parallel) as server:
                    rows = []
                    if a.stage in ['screen', 'matrix']:
                        for depth_index, depth in enumerate(a.depths.split(',')):
                            for i in range(a.warmups + a.reps):
                                prompt = fresh_prompt(workloads, depth, depth_index * 1000 + block * (a.warmups + a.reps) + i)
                                row = server.generate(prompt)
                                row['expected_input_tokens'] = workloads[depth][0]['reference_token_count']
                                reported_input = row['prompt_tokens_total_reported'] if c['kind'] == 'ollama' else row['prompt_tokens_evaluated']
                                if reported_input != row['expected_input_tokens']:
                                    save(directory / f'd{depth}-input-count-failure-{i}.json', row)
                                    raise RuntimeError('Runtime input count differs from the reference tokenizer')
                                if c['kind'] == 'ollama' and row['cached_tokens_reported'] is not None:
                                    if row['cached_tokens_reported'] > workloads[depth][0]['identifier_end_token_index']:
                                        save(directory / f'd{depth}-cache-failure-{i}.json', row)
                                        raise RuntimeError('Unexpected long-prefix reuse in a fresh-prompt request')
                                row.update(depth=int(depth), iteration=i, excluded=i < a.warmups)
                                save(directory / f'd{depth}-r{i:03}.json', row)
                                rows.append(row)
                            report[label + '-d' + depth] = summarise([x for x in rows if not x['excluded'] and x['depth'] == int(depth)])
                    elif a.stage == 'quality':
                        fixtures = json.loads((ROOT / 'tools/quality-probes.json').read_text())
                        for fixture in fixtures:
                            prompt = fixture['prompt']
                            if fixture.get('long_context'):
                                notes = [f'Record {i}: the shelf contains plain paper and a small box.' for i in range(90)]
                                notes.insert(43, 'The access word is cobalt-orchid-731.')
                                prompt += '\n' + '\n'.join(notes)
                            responses = []
                            for rep in range(a.reps):
                                row = server.generate(chat(prompt), limit=128)
                                text = row['text'].strip()
                                try:
                                    correct = json.loads(text) == fixture['expected'] if fixture['kind'] == 'json' else text.casefold() == fixture['expected'].casefold()
                                except ValueError:
                                    correct = False
                                row.update(fixture=fixture['id'], correct=correct)
                                save(directory / f'quality-{fixture["id"]}-{rep}.json', row)
                                responses.append(row)
                            rows.extend(responses)
                        for case, prompt in [('code', 'Write a Python function that merges two sorted lists. Return only complete code.'),
                                             ('reasoning', 'Explain why the sum of integers from 1 to n equals n(n+1)/2.'),
                                             ('prose', 'Explain CPU cache lines, locality and memory bandwidth in three short paragraphs.')]:
                            for rep in range(a.reps):
                                row = server.generate(chat(prompt), limit=1024)
                                save(directory / f'complete-{case}-{rep}.json', row)
                        report[label] = {'probe_responses': len(rows), 'correct': sum(x['correct'] for x in rows),
                                         'repeat_consistent': all(len({x['text'] for x in rows if x['fixture'] == f['id']}) == 1 for f in fixtures)}
                    elif a.stage in ['soak', 'cache', 'concurrency']:
                        prompt_variants = workloads['2048']
                        for i in range(a.warmups):
                            prompt = prompt_variants[0]['prompt'] if a.stage == 'cache' else fresh_prompt(workloads, '2048', i)
                            save(directory / f'warmup-{i}.json', server.generate(prompt, cache=a.stage == 'cache'))
                        began = time.monotonic()
                        if a.stage == 'soak':
                            i = 0
                            while time.monotonic() - began < a.seconds:
                                row = server.generate(fresh_prompt(workloads, '2048', i + a.warmups))
                                row['since_start_s'] = time.monotonic() - began
                                save(directory / f'soak-{i:05}.json', row)
                                rows.append(row)
                                i += 1
                        elif a.stage == 'cache':
                            for i in range(a.reps):
                                row = server.generate(prompt_variants[0]['prompt'], cache=True)
                                save(directory / f'cached-{i:03}.json', row)
                                rows.append(row)
                        else:
                            for batch in range(a.reps):
                                with concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel) as pool:
                                    futures = [pool.submit(server.generate, fresh_prompt(workloads, '2048', 1000 + a.warmups + batch * a.parallel + i)) for i in range(a.parallel)]
                                    for i, future in enumerate(futures):
                                        row = future.result()
                                        save(directory / f'batch-{batch:03}-request-{i}.json', row)
                                        rows.append(row)
                        elapsed = time.monotonic() - began
                        report[label] = {**summarise(rows), 'wall_s': elapsed,
                                         'aggregate_output_tps': sum(x['output_tokens'] for x in rows) / elapsed}
                        if a.stage == 'soak':
                            report[label]['windows'] = {str(start): summarise([x for x in rows if start <= x['since_start_s'] < start + 300]) for start in range(0, a.seconds, 300)}
                    save(a.out / 'report.json', report)
            except Exception as e:
                directory.mkdir(parents=True, exist_ok=True)
                error = {'error': str(e), 'traceback': traceback.format_exc()}
                save(directory / 'failure.json', error)
                report[label] = error
                save(a.out / 'report.json', report)
                print('FAILED', label, str(e), flush=True)
                if a.fail_fast:
                    raise
            print('Completed', label, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['workloads', 'screen', 'matrix', 'quality', 'soak', 'cache', 'concurrency'], required=True)
    p.add_argument('--configs', required=True, help='Comma-separated configuration names; see --list')
    p.add_argument('--workloads', type=Path)
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--depths', default='512,2048,8192')
    p.add_argument('--parallel', type=int, default=1)
    p.add_argument('--seconds', type=int, default=1800)
    p.add_argument('--fail-fast', action='store_true')
    p.add_argument('--list', action='store_true')
    import sys
    if '--list' in sys.argv:
        print('\n'.join(configurations()))
        return
    a = p.parse_args()
    with (ROOT / '.deps/comparison.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(a)


if __name__ == '__main__':
    main()
