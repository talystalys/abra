#!/usr/bin/env bash
set -eu
test "$(hostname)" = BigBoy
cd /home/taly/legacy-ai-1080ti-port
comparison_out=${1:-bench/results/1080ti-comparison/setup}
mkdir -p "$comparison_out"
exec 9> .deps/comparison.lock
flock -n 9
mkdir -p "$comparison_out" .deps/ollama-current .deps/ollama-compatible
(
 if [ ! -f .deps/ollama-current/release.tar.zst ]; then
  curl -fL --retry 2 -o .deps/ollama-current/release.tar.zst https://github.com/ollama/ollama/releases/download/v0.40.6/ollama-linux-amd64.tar.zst > "${comparison_out}/ollama-download.log" 2>&1
 fi
 printf '%s\n' 'c164ab8b1b28e2a247c29719b8e404c423c5b83ae64c3ab4e59d50c7ff7fc5e0  .deps/ollama-current/release.tar.zst' | sha256sum --check
 tar --zstd -xf .deps/ollama-current/release.tar.zst -C .deps/ollama-current > "${comparison_out}/ollama-extract.log" 2>&1
 if [ ! -f .deps/ollama-compatible/release.tgz ]; then
  curl -fL --retry 2 -o .deps/ollama-compatible/release.tgz https://github.com/ollama/ollama/releases/download/v0.11.10/ollama-linux-amd64.tgz > "${comparison_out}/ollama-compatible-download.log" 2>&1
 fi
 printf '%s\n' 'da2be71e972752e3099ef1d1226050d2b3d1ce79a4fec3bdb6edc70a7601e060  .deps/ollama-compatible/release.tgz' | sha256sum --check
 tar -xzf .deps/ollama-compatible/release.tgz -C .deps/ollama-compatible
) &
ollama_download_pid=$!
if [ ! -d .deps/llama.cpp-current ]; then
 git clone --depth 1 --branch v0.7.0 https://github.com/ggml-org/llama.cpp.git .deps/llama.cpp-current > "${comparison_out}/current-clone.log" 2>&1
fi
test "$(git -C .deps/llama.cpp-current rev-parse HEAD)" = 620d2225cc436ab0b001e3200e590e3b286ef9ab
git -C .deps/llama.cpp-current diff --exit-code HEAD
cmake -S .deps/llama.cpp-current -B .deps/llama.cpp-current/build-comparison -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_FLAGS="-I/home/taly/legacy-ai-1080ti-port/.cuda_compat/include -Wno-deprecated-gpu-targets" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF > "${comparison_out}/current-configure.log" 2>&1
cmake --build .deps/llama.cpp-current/build-comparison --target llama-server llama-perplexity --parallel 8 > "${comparison_out}/current-build.log" 2>&1 &
current_build_pid=$!
cmake --build .deps/llama.cpp-port/build-port --target llama-server --parallel 8 > "${comparison_out}/optimized-server-build.log" 2>&1
if [ ! -d .deps/llama.cpp-pinned-stock ]; then
 git -C .deps/llama.cpp worktree add --detach /home/taly/legacy-ai-1080ti-port/.deps/llama.cpp-pinned-stock 169e4a7ff201e666c2325bcd35afb64573e39d09 > "${comparison_out}/pinned-worktree.log" 2>&1
fi
cmake -S .deps/llama.cpp-pinned-stock -B .deps/llama.cpp-pinned-stock/build-comparison -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_FLAGS="-I/home/taly/legacy-ai-1080ti-port/.cuda_compat/include -Wno-deprecated-gpu-targets" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF > "${comparison_out}/pinned-configure.log" 2>&1
cmake --build .deps/llama.cpp-pinned-stock/build-comparison --target llama-server llama-perplexity --parallel 8 > "${comparison_out}/pinned-build.log" 2>&1 &
pinned_build_pid=$!
wait "$ollama_download_pid"
wait "$current_build_pid"
wait "$pinned_build_pid"
printf 'Setup complete\n'
