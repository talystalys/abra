#!/usr/bin/env bash
set -eu
test "$(hostname)" = BigBoy
cd /home/taly/abra-1080ti-port
comparison_out=${1:-bench/results/1080ti-comparison/setup}
mkdir -p "$comparison_out"
exec 9> .deps/comparison.lock
flock -n 9
build_pids=()
for version in pinned current; do
 if [ "$version" = pinned ]; then source_dir=.deps/llama.cpp-pinned-stock; else source_dir=.deps/llama.cpp-current; fi
 (
 cmake -S "$source_dir" -B "$source_dir/build-comparison-nccl" -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_FLAGS="-I/home/taly/abra-1080ti-port/.cuda_compat/include -Wno-deprecated-gpu-targets" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR=/home/taly/abra-1080ti-port/.deps/nccl-2.27.7-cuda12/usr/include -DNCCL_LIBRARY=/home/taly/abra-1080ti-port/.deps/nccl-2.27.7-cuda12/usr/lib/x86_64-linux-gnu/libnccl.so > "${comparison_out}/$version-nccl-configure.log" 2>&1 &&
 cmake --build "$source_dir/build-comparison-nccl" --target llama-server llama-perplexity --parallel 12 > "${comparison_out}/$version-nccl-build.log" 2>&1
 ) &
 build_pids+=("$!")
done
build_status=0
for build_pid in "${build_pids[@]}"; do
 wait "$build_pid" || build_status=1
done
exit "$build_status"
