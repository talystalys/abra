#!/usr/bin/env python3
"""Opt in to row splitting of Qwen3-4B's duplicated tied output projection."""
import argparse
from pathlib import Path
import subprocess

PIN = '169e4a7ff201e666c2325bcd35afb64573e39d09'
OLD = '''        // output
        if (std::regex_match(tensor_name, pattern_output_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }'''
NEW = '''        // output
        // Abra: a duplicated tied output keeps the token_embd.weight name,
        // so the upstream name-only classifier mirrors it. Identify the output
        // by its original tensor identity; leave token lookup embeddings mirrored.
        static const bool split_tied_output = [] {
            const char * value = std::getenv("ABRA_SPLIT_TIED_OUTPUT");
            return value && std::strcmp(value, "1") == 0;
        }();
        const bool canonical_tied_output = split_tied_output && ud->n_devices == 2 &&
            ud->model->arch == LLM_ARCH_QWEN3 && hparams.n_layer() == 36 &&
            tensor == ud->model->output && tensor != ud->model->tok_embd &&
            tensor->type == GGML_TYPE_Q6_K && tensor->ne[0] == 2560 && tensor->ne[1] == 151936;
        if (std::regex_match(tensor_name, pattern_output_weight) || canonical_tied_output) {
            if (canonical_tied_output) {
                static std::atomic_flag reported = ATOMIC_FLAG_INIT;
                if (!reported.test_and_set()) {
                    LLAMA_LOG_INFO("abra: tied Qwen output row splitting active\\n");
                }
            }
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }'''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    a = p.parse_args()
    if subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise SystemExit('Expected pinned source')
    path = a.source / 'src/llama-model.cpp'
    data = path.read_text()
    if NEW not in data:
        if data.count(OLD) != 1:
            raise SystemExit('Unexpected output classifier; refusing patch')
        data = data.replace(OLD, NEW)
    if '#include <atomic>' not in data:
        data = '#include <atomic>\n' + data
    path.write_text(data)
    print('Installed opt-in tied output row splitting:', path)

if __name__ == '__main__':
    main()
