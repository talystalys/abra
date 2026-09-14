#include "q4k_family.cuh"
#include "q4k_vecdot.cuh"
#include "upstream_adapter.cuh"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <map>
#include <tuple>
#include <mutex>

// =============================================================================
// Milestone M4: Parameterized Q4_K x Q8_1 MMVQ Family Kernel Template
//
// Parameters:
//   WARPS_PER_ROW   : Number of warps cooperating on each output row (1, 2, or 4)
//   WARPS_PER_BLOCK : Total warps in the thread block (1, 2, 4, 8, 16)
//   UNROLL          : Unroll factor for K-superblock loop (1 or 2)
//
// Invariants:
//   ROWS_PER_BLOCK = WARPS_PER_BLOCK / WARPS_PER_ROW
//   THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32
//   Intra-warp shuffle reduction reduces 32 lanes -> lane 0
//   Inter-warp smem reduction used only when WARPS_PER_ROW > 1
//   Deadlock-free execution on arbitrary N through predicated row validation
// =============================================================================
template <int WARPS_PER_ROW, int WARPS_PER_BLOCK, int UNROLL = 1>
__global__ void q4k_q8_mmvq_family_kernel(
    const block_q4_K * __restrict__ d_w,
    const block_q8_1 * __restrict__ d_x,
    float * __restrict__ d_y,
    const int N,
    const int K
) {
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK / WARPS_PER_ROW;
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int group_id = warp_id / WARPS_PER_ROW;
    const int warp_in_group = warp_id % WARPS_PER_ROW;

    const int row = blockIdx.x * ROWS_PER_BLOCK + group_id;
    const bool is_valid_row = (row < N);

    const int tid_in_group = warp_in_group * 32 + lane;
    const int blocks_per_row = K / QK_K;
    const int blocks_per_iter = WARPS_PER_ROW * 2;
    const int kbx_start = tid_in_group / 16;
    const int kqs = 2 * (lane % 16);

    float sum = 0.0f;

    if (is_valid_row) {
        const block_q4_K * row_w = d_w + row * blocks_per_row;

        if constexpr (UNROLL == 2) {
            #pragma unroll 2
            for (int kbx = kbx_start; kbx < blocks_per_row; kbx += blocks_per_iter) {
                const int kby = kbx * (QK_K / QK8_1); // kbx * 8
                sum += vec_dot_q4_K_q8_1(row_w, &d_x[kby], kbx, kqs);
            }
        } else {
            for (int kbx = kbx_start; kbx < blocks_per_row; kbx += blocks_per_iter) {
                const int kby = kbx * (QK_K / QK8_1); // kbx * 8
                sum += vec_dot_q4_K_q8_1(row_w, &d_x[kby], kbx, kqs);
            }
        }

        // Intra-warp shuffle reduction (32 lanes -> lane 0)
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
        }
    }

    if constexpr (WARPS_PER_ROW == 1) {
        // Zero shared memory, zero syncthreads
        if (is_valid_row && lane == 0) {
            d_y[row] = sum;
        }
    } else if constexpr (WARPS_PER_ROW == 2) {
        __shared__ float smem_warps[WARPS_PER_BLOCK];
        if (lane == 0) {
            smem_warps[warp_id] = sum;
        }
        __syncthreads();

        if (is_valid_row && warp_in_group == 0 && lane == 0) {
            float total = smem_warps[group_id * 2 + 0] + smem_warps[group_id * 2 + 1];
            d_y[row] = total;
        }
    } else if constexpr (WARPS_PER_ROW == 4) {
        __shared__ float smem_warps[WARPS_PER_BLOCK];
        if (lane == 0) {
            smem_warps[warp_id] = sum;
        }
        __syncthreads();

        if (is_valid_row && warp_in_group == 0 && lane == 0) {
            float total = smem_warps[group_id * 4 + 0] +
                          smem_warps[group_id * 4 + 1] +
                          smem_warps[group_id * 4 + 2] +
                          smem_warps[group_id * 4 + 3];
            d_y[row] = total;
        }
    }
}

// Explicit instantiations of all candidate kernels
template __global__ void q4k_q8_mmvq_family_kernel<1, 1, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<1, 2, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<1, 4, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<1, 8, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<1, 16, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);

template __global__ void q4k_q8_mmvq_family_kernel<2, 2, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<2, 4, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<2, 8, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<2, 16, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);

template __global__ void q4k_q8_mmvq_family_kernel<4, 4, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<4, 8, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<4, 16, 1>(const block_q4_K *, const block_q8_1 *, float *, int, int);

template __global__ void q4k_q8_mmvq_family_kernel<1, 4, 2>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<2, 4, 2>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<1, 8, 2>(const block_q4_K *, const block_q8_1 *, float *, int, int);
template __global__ void q4k_q8_mmvq_family_kernel<2, 8, 2>(const block_q4_K *, const block_q8_1 *, float *, int, int);

static const std::vector<Q4KCandidateID> ALL_CANDIDATES = {
    CANDIDATE_UPSTREAM_MMVQ,
    CANDIDATE_WPR1_WPB1,
    CANDIDATE_WPR1_WPB2,
    CANDIDATE_WPR1_WPB4,
    CANDIDATE_WPR1_WPB8,
    CANDIDATE_WPR1_WPB16,
    CANDIDATE_WPR2_WPB2,
    CANDIDATE_WPR2_WPB4,
    CANDIDATE_WPR2_WPB8,
    CANDIDATE_WPR2_WPB16,
    CANDIDATE_WPR4_WPB4,
    CANDIDATE_WPR4_WPB8,
    CANDIDATE_WPR4_WPB16,
    CANDIDATE_WPR1_WPB4_U2,
    CANDIDATE_WPR2_WPB4_U2,
    CANDIDATE_WPR1_WPB8_U2,
    CANDIDATE_WPR2_WPB8_U2
};

const std::vector<Q4KCandidateID> & get_all_candidate_ids() {
    return ALL_CANDIDATES;
}

Q4KCandidateSpec get_candidate_spec(Q4KCandidateID id) {
    switch (id) {
        case CANDIDATE_UPSTREAM_MMVQ:
            return {CANDIDATE_UPSTREAM_MMVQ, "VARIANT_UPSTREAM_MMVQ", "ggml_cuda_op_mul_mat_vec_q", 4, 4, 128, 1, 1, "inter_warp_smem", 16, true};

        case CANDIDATE_WPR1_WPB1:
            return {CANDIDATE_WPR1_WPB1, "Q4K_PASCAL_WPR1_WPB1", "q4k_q8_mmvq_family_kernel<1, 1, 1>", 1, 1, 32, 1, 1, "warp_shuffle_zero_smem", 0, false};
        case CANDIDATE_WPR1_WPB2:
            return {CANDIDATE_WPR1_WPB2, "Q4K_PASCAL_WPR1_WPB2", "q4k_q8_mmvq_family_kernel<1, 2, 1>", 1, 2, 64, 2, 1, "warp_shuffle_zero_smem", 0, false};
        case CANDIDATE_WPR1_WPB4:
            return {CANDIDATE_WPR1_WPB4, "Q4K_PASCAL_WPR1_WPB4", "q4k_q8_mmvq_family_kernel<1, 4, 1>", 1, 4, 128, 4, 1, "warp_shuffle_zero_smem", 0, false};
        case CANDIDATE_WPR1_WPB8:
            return {CANDIDATE_WPR1_WPB8, "Q4K_PASCAL_WPR1_WPB8", "q4k_q8_mmvq_family_kernel<1, 8, 1>", 1, 8, 256, 8, 1, "warp_shuffle_zero_smem", 0, false};
        case CANDIDATE_WPR1_WPB16:
            return {CANDIDATE_WPR1_WPB16, "Q4K_PASCAL_WPR1_WPB16", "q4k_q8_mmvq_family_kernel<1, 16, 1>", 1, 16, 512, 16, 1, "warp_shuffle_zero_smem", 0, false};

        case CANDIDATE_WPR2_WPB2:
            return {CANDIDATE_WPR2_WPB2, "Q4K_PASCAL_WPR2_WPB2", "q4k_q8_mmvq_family_kernel<2, 2, 1>", 2, 2, 64, 1, 1, "inter_warp_smem", 8, false};
        case CANDIDATE_WPR2_WPB4:
            return {CANDIDATE_WPR2_WPB4, "Q4K_PASCAL_WPR2_WPB4", "q4k_q8_mmvq_family_kernel<2, 4, 1>", 2, 4, 128, 2, 1, "inter_warp_smem", 16, false};
        case CANDIDATE_WPR2_WPB8:
            return {CANDIDATE_WPR2_WPB8, "Q4K_PASCAL_WPR2_WPB8", "q4k_q8_mmvq_family_kernel<2, 8, 1>", 2, 8, 256, 4, 1, "inter_warp_smem", 32, false};
        case CANDIDATE_WPR2_WPB16:
            return {CANDIDATE_WPR2_WPB16, "Q4K_PASCAL_WPR2_WPB16", "q4k_q8_mmvq_family_kernel<2, 16, 1>", 2, 16, 512, 8, 1, "inter_warp_smem", 64, false};

        case CANDIDATE_WPR4_WPB4:
            return {CANDIDATE_WPR4_WPB4, "Q4K_PASCAL_WPR4_WPB4", "q4k_q8_mmvq_family_kernel<4, 4, 1>", 4, 4, 128, 1, 1, "inter_warp_smem", 16, false};
        case CANDIDATE_WPR4_WPB8:
            return {CANDIDATE_WPR4_WPB8, "Q4K_PASCAL_WPR4_WPB8", "q4k_q8_mmvq_family_kernel<4, 8, 1>", 4, 8, 256, 2, 1, "inter_warp_smem", 32, false};
        case CANDIDATE_WPR4_WPB16:
            return {CANDIDATE_WPR4_WPB16, "Q4K_PASCAL_WPR4_WPB16", "q4k_q8_mmvq_family_kernel<4, 16, 1>", 4, 16, 512, 4, 1, "inter_warp_smem", 64, false};

        case CANDIDATE_WPR1_WPB4_U2:
            return {CANDIDATE_WPR1_WPB4_U2, "Q4K_PASCAL_WPR1_WPB4_U2", "q4k_q8_mmvq_family_kernel<1, 4, 2>", 1, 4, 128, 4, 2, "warp_shuffle_zero_smem", 0, false};
        case CANDIDATE_WPR2_WPB4_U2:
            return {CANDIDATE_WPR2_WPB4_U2, "Q4K_PASCAL_WPR2_WPB4_U2", "q4k_q8_mmvq_family_kernel<2, 4, 2>", 2, 4, 128, 2, 2, "inter_warp_smem", 16, false};
        case CANDIDATE_WPR1_WPB8_U2:
            return {CANDIDATE_WPR1_WPB8_U2, "Q4K_PASCAL_WPR1_WPB8_U2", "q4k_q8_mmvq_family_kernel<1, 8, 2>", 1, 8, 256, 8, 2, "warp_shuffle_zero_smem", 0, false};
        case CANDIDATE_WPR2_WPB8_U2:
            return {CANDIDATE_WPR2_WPB8_U2, "Q4K_PASCAL_WPR2_WPB8_U2", "q4k_q8_mmvq_family_kernel<2, 8, 2>", 2, 8, 256, 4, 2, "inter_warp_smem", 32, false};

        case CANDIDATE_AUTOTUNED_DISPATCH:
            return {CANDIDATE_AUTOTUNED_DISPATCH, "Q4K_AUTOTUNED_DISPATCH", "launch_q4k_autotuned", 0, 0, 0, 0, 0, "autotuned_dispatch", 0, false};

        default:
            return {id, "UNKNOWN_CANDIDATE", "unknown", 0, 0, 0, 0, 0, "unknown", 0, false};
    }
}

const char * get_candidate_name(Q4KCandidateID id) {
    static std::string unknown = "UNKNOWN";
    Q4KCandidateSpec spec = get_candidate_spec(id);
    return spec.name;
}

void launch_q4k_candidate(
    Q4KCandidateID candidate,
    const block_q4_K * d_w,
    const block_q8_1 * d_x,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (K % QK_K != 0) {
        throw std::invalid_argument("K must be a multiple of QK_K=256");
    }

    if (candidate == CANDIDATE_UPSTREAM_MMVQ) {
        launch_pinned_upstream_mmvq(
            (const void *)d_w,
            (const void *)d_x,
            d_y,
            N,
            K,
            stream
        );
        return;
    }

    if (candidate == CANDIDATE_AUTOTUNED_DISPATCH) {
        launch_q4k_autotuned(d_w, d_x, d_y, N, K, 61, stream);
        return;
    }

    Q4KCandidateSpec spec = get_candidate_spec(candidate);
    dim3 block_dims(32, spec.warps_per_block);
    dim3 grid_dims((N + spec.rows_per_block - 1) / spec.rows_per_block, 1, 1);

    switch (candidate) {
        case CANDIDATE_WPR1_WPB1:
            q4k_q8_mmvq_family_kernel<1, 1, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR1_WPB2:
            q4k_q8_mmvq_family_kernel<1, 2, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR1_WPB4:
            q4k_q8_mmvq_family_kernel<1, 4, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR1_WPB8:
            q4k_q8_mmvq_family_kernel<1, 8, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR1_WPB16:
            q4k_q8_mmvq_family_kernel<1, 16, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;

        case CANDIDATE_WPR2_WPB2:
            q4k_q8_mmvq_family_kernel<2, 2, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR2_WPB4:
            q4k_q8_mmvq_family_kernel<2, 4, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR2_WPB8:
            q4k_q8_mmvq_family_kernel<2, 8, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR2_WPB16:
            q4k_q8_mmvq_family_kernel<2, 16, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;

        case CANDIDATE_WPR4_WPB4:
            q4k_q8_mmvq_family_kernel<4, 4, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR4_WPB8:
            q4k_q8_mmvq_family_kernel<4, 8, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR4_WPB16:
            q4k_q8_mmvq_family_kernel<4, 16, 1><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;

        case CANDIDATE_WPR1_WPB4_U2:
            q4k_q8_mmvq_family_kernel<1, 4, 2><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR2_WPB4_U2:
            q4k_q8_mmvq_family_kernel<2, 4, 2><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR1_WPB8_U2:
            q4k_q8_mmvq_family_kernel<1, 8, 2><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        case CANDIDATE_WPR2_WPB8_U2:
            q4k_q8_mmvq_family_kernel<2, 8, 2><<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;

        default:
            throw std::invalid_argument("Unknown Q4KCandidateID");
    }
}

void launch_q4k_candidate_combined_pipeline(
    Q4KCandidateID candidate,
    const block_q4_K * d_w,
    const float * d_x_f32,
    block_q8_1 * d_x_q8_scratch,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream
) {
    launch_quantize_f32_to_q8_1(d_x_f32, d_x_q8_scratch, K, stream);
    launch_q4k_candidate(candidate, d_w, d_x_q8_scratch, d_y, N, K, stream);
}

static const void * get_kernel_func_pointer(Q4KCandidateID candidate) {
    switch (candidate) {
        case CANDIDATE_UPSTREAM_MMVQ:
        case CANDIDATE_WPR4_WPB4:
            return (const void *)q4k_q8_mmvq_family_kernel<4, 4, 1>;
        case CANDIDATE_WPR1_WPB1:
            return (const void *)q4k_q8_mmvq_family_kernel<1, 1, 1>;
        case CANDIDATE_WPR1_WPB2:
            return (const void *)q4k_q8_mmvq_family_kernel<1, 2, 1>;
        case CANDIDATE_WPR1_WPB4:
            return (const void *)q4k_q8_mmvq_family_kernel<1, 4, 1>;
        case CANDIDATE_WPR1_WPB8:
            return (const void *)q4k_q8_mmvq_family_kernel<1, 8, 1>;
        case CANDIDATE_WPR1_WPB16:
            return (const void *)q4k_q8_mmvq_family_kernel<1, 16, 1>;
        case CANDIDATE_WPR2_WPB2:
            return (const void *)q4k_q8_mmvq_family_kernel<2, 2, 1>;
        case CANDIDATE_WPR2_WPB4:
            return (const void *)q4k_q8_mmvq_family_kernel<2, 4, 1>;
        case CANDIDATE_WPR2_WPB8:
            return (const void *)q4k_q8_mmvq_family_kernel<2, 8, 1>;
        case CANDIDATE_WPR2_WPB16:
            return (const void *)q4k_q8_mmvq_family_kernel<2, 16, 1>;
        case CANDIDATE_WPR4_WPB8:
            return (const void *)q4k_q8_mmvq_family_kernel<4, 8, 1>;
        case CANDIDATE_WPR4_WPB16:
            return (const void *)q4k_q8_mmvq_family_kernel<4, 16, 1>;
        case CANDIDATE_WPR1_WPB4_U2:
            return (const void *)q4k_q8_mmvq_family_kernel<1, 4, 2>;
        case CANDIDATE_WPR2_WPB4_U2:
            return (const void *)q4k_q8_mmvq_family_kernel<2, 4, 2>;
        case CANDIDATE_WPR1_WPB8_U2:
            return (const void *)q4k_q8_mmvq_family_kernel<1, 8, 2>;
        case CANDIDATE_WPR2_WPB8_U2:
            return (const void *)q4k_q8_mmvq_family_kernel<2, 8, 2>;
        default:
            return (const void *)q4k_q8_mmvq_family_kernel<2, 4, 1>;
    }
}

Q4KOccupancyInfo get_q4k_candidate_occupancy_info(Q4KCandidateID candidate, int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    const void * kernel_func = get_kernel_func_pointer(candidate);
    Q4KCandidateSpec spec = get_candidate_spec(candidate);
    int block_dim_x = 32;
    int block_dim_y = spec.warps_per_block;
    size_t dynamic_smem = 0;

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel_func);

    int block_size = block_dim_x * block_dim_y;
    int num_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel_func, block_size, dynamic_smem);

    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / 32;
    int resident_warps_per_sm = (num_blocks * block_size) / 32;
    double theoretical_occupancy_pct = (double)resident_warps_per_sm / (double)max_warps_per_sm * 100.0;

    Q4KOccupancyInfo info;
    info.variant_name = spec.name;
    info.block_dim_x = block_dim_x;
    info.block_dim_y = block_dim_y;
    info.threads_per_block = block_size;
    info.regs_per_thread = attr.numRegs;
    info.static_shared_mem_bytes = attr.sharedSizeBytes;
    info.max_active_blocks_per_sm = num_blocks;
    info.resident_warps_per_sm = resident_warps_per_sm;
    info.max_warps_per_sm = max_warps_per_sm;
    info.theoretical_occupancy_pct = theoretical_occupancy_pct;
    info.spill_stores_bytes = attr.localSizeBytes;
    info.spill_loads_bytes = 0;

    return info;
}

// =============================================================================
// Offline Autotuning Database & Shape-Aware Selector Table
// =============================================================================
static std::mutex g_tuning_mutex;
static std::map<std::tuple<int, int, int>, Q4KTuningEntry> g_tuning_table;

// Built-in verified baseline tuning entries (safe fallback table)
static void init_default_tuning_table() {
    std::lock_guard<std::mutex> lock(g_tuning_mutex);
    if (!g_tuning_table.empty()) return;

    // Pascal SM 6.1 initial baseline defaults
    // attn_q (4096 x 2560): 2-warp split
    g_tuning_table[{4096, 2560, 61}] = {4096, 2560, 61, CANDIDATE_WPR2_WPB4, "Q4K_PASCAL_WPR2_WPB4", 54.27, 60.42, 1.113};
    // attn_kv (1024 x 2560): 1-warp row (zero smem)
    g_tuning_table[{1024, 2560, 61}] = {1024, 2560, 61, CANDIDATE_WPR1_WPB4, "Q4K_PASCAL_WPR1_WPB4", 13.31, 17.41, 1.308};
    // attn_out (2560 x 4096): 2-warp split
    g_tuning_table[{2560, 4096, 61}] = {2560, 4096, 61, CANDIDATE_WPR2_WPB4, "Q4K_PASCAL_WPR2_WPB4", 55.30, 59.07, 1.068};
    // ffn_gate_up (9728 x 2560): 2-warp split
    g_tuning_table[{9728, 2560, 61}] = {9728, 2560, 61, CANDIDATE_WPR2_WPB4, "Q4K_PASCAL_WPR2_WPB4", 122.88, 135.09, 1.099};
    // ffn_down (2560 x 9728): 4-warp control
    g_tuning_table[{2560, 9728, 61}] = {2560, 9728, 61, CANDIDATE_WPR4_WPB4, "Q4K_PASCAL_WPR4_WPB4", 126.91, 132.10, 1.041};

    // Turing SM 7.5 initial baseline defaults
    g_tuning_table[{4096, 2560, 75}] = {4096, 2560, 75, CANDIDATE_WPR1_WPB4, "Q4K_PASCAL_WPR1_WPB4", 33.84, 35.07, 1.036};
    g_tuning_table[{1024, 2560, 75}] = {1024, 2560, 75, CANDIDATE_WPR1_WPB8, "Q4K_PASCAL_WPR1_WPB8", 11.30, 12.29, 1.088};
    g_tuning_table[{2560, 4096, 75}] = {2560, 4096, 75, CANDIDATE_WPR1_WPB4_U2, "Q4K_PASCAL_WPR1_WPB4_U2", 32.08, 33.46, 1.043};
    g_tuning_table[{9728, 2560, 75}] = {9728, 2560, 75, CANDIDATE_WPR1_WPB4_U2, "Q4K_PASCAL_WPR1_WPB4_U2", 72.08, 76.78, 1.065};
    g_tuning_table[{2560, 9728, 75}] = {2560, 9728, 75, CANDIDATE_UPSTREAM_MMVQ, "VARIANT_UPSTREAM_MMVQ", 72.80, 72.80, 1.000};
}

void register_autotuned_table_entry(
    int N,
    int K,
    int arch,
    Q4KCandidateID winning_candidate,
    double winning_med_us,
    double upstream_med_us
) {
    std::lock_guard<std::mutex> lock(g_tuning_mutex);
    double speedup = (upstream_med_us > 0.0 && winning_med_us > 0.0) ? upstream_med_us / winning_med_us : 1.0;
    g_tuning_table[{N, K, arch}] = {
        N,
        K,
        arch,
        winning_candidate,
        get_candidate_name(winning_candidate),
        winning_med_us,
        upstream_med_us,
        speedup
    };
}

Q4KCandidateID get_autotuned_candidate_for_shape(int N, int K, int arch) {
    init_default_tuning_table();
    std::lock_guard<std::mutex> lock(g_tuning_mutex);
    auto it = g_tuning_table.find({N, K, arch});
    if (it != g_tuning_table.end()) {
        return it->second.winning_candidate;
    }

    // Deterministic safe fallback heuristic:
    // If shape is not in table:
    // If K % 256 == 0:
    //   If K == 2560 -> WPR2_WPB4 (addresses K=2560 tail)
    //   If N >= 8192 -> WPR1_WPB4 (high thread-block count)
    //   Else -> WPR2_WPB4
    // Otherwise fallback to exact upstream MMVQ
    if (K % QK_K == 0) {
        if (K == 2560) {
            return (N >= 8192) ? CANDIDATE_WPR1_WPB4 : CANDIDATE_WPR2_WPB4;
        }
        return CANDIDATE_WPR2_WPB4;
    }
    return CANDIDATE_UPSTREAM_MMVQ;
}

void launch_q4k_autotuned(
    const block_q4_K * d_w,
    const block_q8_1 * d_x,
    float * d_y,
    int N,
    int K,
    int arch,
    cudaStream_t stream
) {
    Q4KCandidateID selected = get_autotuned_candidate_for_shape(N, K, arch);
    launch_q4k_candidate(selected, d_w, d_x, d_y, N, K, stream);
}

void launch_q4k_autotuned_combined_pipeline(
    const block_q4_K * d_w,
    const float * d_x_f32,
    block_q8_1 * d_x_q8_scratch,
    float * d_y,
    int N,
    int K,
    int arch,
    cudaStream_t stream
) {
    launch_quantize_f32_to_q8_1(d_x_f32, d_x_q8_scratch, K, stream);
    launch_q4k_autotuned(d_w, d_x_q8_scratch, d_y, N, K, arch, stream);
}
