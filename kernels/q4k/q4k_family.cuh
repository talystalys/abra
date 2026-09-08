#pragma once

#include "q4k_q8_mmvq.cuh"
#include <vector>
#include <string>
#include <map>

// Milestone M4: Parameterized Q4_K x Q8_1 Kernel Family Candidate Enumeration
enum Q4KCandidateID {
    CANDIDATE_UPSTREAM_MMVQ      = 0,     // Exact pinned llama.cpp MMVQ (libggml-cuda.so)

    // WPR = 1 (1 warp per row -> 0 smem, intra-warp shuffle reduction only)
    CANDIDATE_WPR1_WPB1          = 101,   // 1 warp/row, 1 warp/block (32 thds, 1 row/blk)
    CANDIDATE_WPR1_WPB2          = 102,   // 1 warp/row, 2 warps/block (64 thds, 2 rows/blk)
    CANDIDATE_WPR1_WPB4          = 104,   // 1 warp/row, 4 warps/block (128 thds, 4 rows/blk) [M3 Variant C]
    CANDIDATE_WPR1_WPB8          = 108,   // 1 warp/row, 8 warps/block (256 thds, 8 rows/blk)
    CANDIDATE_WPR1_WPB16         = 116,   // 1 warp/row, 16 warps/block (512 thds, 16 rows/blk)

    // WPR = 2 (2 warps per row -> smem reduction across 2-warp pairs)
    CANDIDATE_WPR2_WPB2          = 202,   // 2 warps/row, 2 warps/block (64 thds, 1 row/blk)
    CANDIDATE_WPR2_WPB4          = 204,   // 2 warps/row, 4 warps/block (128 thds, 2 rows/blk) [M3 Variant B]
    CANDIDATE_WPR2_WPB8          = 208,   // 2 warps/row, 8 warps/block (256 thds, 4 rows/blk)
    CANDIDATE_WPR2_WPB16         = 216,   // 2 warps/row, 16 warps/block (512 thds, 8 rows/blk)

    // WPR = 4 (4 warps per row -> smem reduction across 4-warp quads)
    CANDIDATE_WPR4_WPB4          = 404,   // 4 warps/row, 4 warps/block (128 thds, 1 row/blk) [M3 Variant A]
    CANDIDATE_WPR4_WPB8          = 408,   // 4 warps/row, 8 warps/block (256 thds, 2 rows/blk)
    CANDIDATE_WPR4_WPB16         = 416,   // 4 warps/row, 16 warps/block (512 thds, 4 rows/blk)

    // Unrolled variants (U=2)
    CANDIDATE_WPR1_WPB4_U2       = 1042,  // 1 warp/row, 4 warps/block, Unroll 2
    CANDIDATE_WPR2_WPB4_U2       = 2042,  // 2 warps/row, 4 warps/block, Unroll 2
    CANDIDATE_WPR1_WPB8_U2       = 1082,  // 1 warp/row, 8 warps/block, Unroll 2
    CANDIDATE_WPR2_WPB8_U2       = 2082,  // 2 warps/row, 8 warps/block, Unroll 2

    // Dynamic autotuned shape-aware dispatch selector
    CANDIDATE_AUTOTUNED_DISPATCH = 999
};

struct Q4KCandidateSpec {
    Q4KCandidateID id;
    const char * name;
    const char * symbol;
    int warps_per_row;
    int warps_per_block;
    int threads_per_block;
    int rows_per_block;
    int unroll_factor;
    const char * reduction_type;
    size_t smem_bytes;
    bool is_upstream;
};

struct Q4KTuningEntry {
    int N;
    int K;
    int arch; // 61 (Pascal) or 75 (Turing)
    Q4KCandidateID winning_candidate;
    const char * winning_name;
    double winning_median_us;
    double upstream_median_us;
    double speedup_vs_upstream;
};

// Candidate inspection utilities
const std::vector<Q4KCandidateID> & get_all_candidate_ids();
Q4KCandidateSpec get_candidate_spec(Q4KCandidateID id);
const char * get_candidate_name(Q4KCandidateID id);

// Launch specific candidate kernel
void launch_q4k_candidate(
    Q4KCandidateID candidate,
    const block_q4_K * d_w,
    const block_q8_1 * d_x,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream = 0
);

// Combined pipeline: F32 activation -> Q8_1 quantization (exact upstream) + Q4_K multiply with candidate
void launch_q4k_candidate_combined_pipeline(
    Q4KCandidateID candidate,
    const block_q4_K * d_w,
    const float * d_x_f32,
    block_q8_1 * d_x_q8_scratch,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream = 0
);

// Theoretical occupancy for candidate
Q4KOccupancyInfo get_q4k_candidate_occupancy_info(Q4KCandidateID candidate, int device_id);

// Shape-aware autotuner selector interface
Q4KCandidateID get_autotuned_candidate_for_shape(int N, int K, int arch = 61);
void register_autotuned_table_entry(int N, int K, int arch, Q4KCandidateID winning_candidate, double winning_med_us, double upstream_med_us);

// Launch autotuned kernel for given shape
void launch_q4k_autotuned(
    const block_q4_K * d_w,
    const block_q8_1 * d_x,
    float * d_y,
    int N,
    int K,
    int arch = 61,
    cudaStream_t stream = 0
);

void launch_q4k_autotuned_combined_pipeline(
    const block_q4_K * d_w,
    const float * d_x_f32,
    block_q8_1 * d_x_q8_scratch,
    float * d_y,
    int N,
    int K,
    int arch = 61,
    cudaStream_t stream = 0
);
