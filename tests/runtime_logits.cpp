// Full-vocabulary, deterministic end-to-end validation of a runtime build.
// Run as separate processes for baseline/candidate so graph environment caches
// cannot leak. Output binary stays in ignored .tmp files; commit JSON analysis.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    try {
        if (argc != 6) {
            throw std::runtime_error("usage: runtime_logits MODEL OUTPUT.tmp TOKENS.json STEPS DEPTH");
        }
        const int steps = std::stoi(argv[4]);
        const int depth = std::stoi(argv[5]);
        if (steps <= 0 || depth < 0) throw std::runtime_error("invalid dimensions");
        ggml_backend_load_all();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        const char *split = std::getenv("LEGACY_AI_VALIDATE_SPLIT");
        mp.split_mode = split && std::strcmp(split, "row") == 0 ? LLAMA_SPLIT_MODE_ROW :
                        split && std::strcmp(split, "tensor") == 0 ? LLAMA_SPLIT_MODE_TENSOR :
                        split && std::strcmp(split, "layer") == 0 ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_NONE;
        llama_model *model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load failed");
        const llama_vocab *vocab = llama_model_get_vocab(model);
        const std::string prompt = "<|im_start|>user\nExplain why the sky is blue, then give a small Python example of a loop. /no_think<|im_end|>\n<|im_start|>assistant\n";
        int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        if (count <= 0) throw std::runtime_error("tokenization count failed");
        std::vector<llama_token> input(count);
        if (llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), input.size(), true, true) != count)
            throw std::runtime_error("tokenization failed");
        if (depth) {
            auto original = input;
            while (int(input.size()) < depth) input.push_back(original[input.size() % original.size()]);
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = input.size() + steps + 32;
        cp.n_batch = 2048;
        cp.n_ubatch = 512;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        const char *fa = std::getenv("LEGACY_AI_VALIDATE_FLASH_ATTN");
        cp.flash_attn_type = fa && std::strcmp(fa, "1") == 0 ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.no_perf = false;
        llama_context *ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context creation failed");
        std::ofstream raw(argv[2], std::ios::binary);
        std::ofstream meta(argv[3]);
        if (!raw || !meta) throw std::runtime_error("output open failed");
        uint32_t header[3] = {0x3149414c, uint32_t(llama_vocab_n_tokens(vocab)), uint32_t(steps)};
        raw.write(reinterpret_cast<const char *>(header), sizeof(header));
        for (size_t pos = 0; pos < input.size(); pos += cp.n_batch) {
            int n = std::min(size_t(cp.n_batch), input.size() - pos);
            auto batch = llama_batch_get_one(input.data() + pos, n);
            if (llama_decode(ctx, batch)) throw std::runtime_error("prompt decode failed");
        }
        std::vector<llama_token> generated;
        std::vector<double> timings;
        for (int step = 0; step < steps; ++step) {
            const float *logits = llama_get_logits_ith(ctx, -1);
            if (!logits) throw std::runtime_error("missing logits");
            for (uint32_t i = 0; i < header[1]; ++i)
                if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite output logit");
            raw.write(reinterpret_cast<const char *>(logits), header[1] * sizeof(float));
            llama_token token = std::max_element(logits, logits + header[1]) - logits;
            generated.push_back(token);
            auto batch = llama_batch_get_one(&token, 1);
            auto start = std::chrono::steady_clock::now();
            if (llama_decode(ctx, batch)) throw std::runtime_error("token decode failed");
            llama_synchronize(ctx);
            timings.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        meta << "{\"vocab\":" << header[1] << ",\"steps\":" << steps << ",\"prompt_tokens\":" << input.size() << ",\"tokens\":[";
        for (int i = 0; i < steps; ++i) meta << (i ? "," : "") << generated[i];
        meta << "],\"decode_ms\":[";
        for (int i = 0; i < steps; ++i) meta << (i ? "," : "") << timings[i];
        meta << "]}\n";
        if (!raw || !meta) throw std::runtime_error("output write failed");
        llama_perf_context_print(ctx);
        llama_free(ctx);
        llama_model_free(model);
        std::cout << "Validated finite logits for " << steps << " steps, " << header[1] << " vocabulary entries per step\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
