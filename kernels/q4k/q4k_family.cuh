#pragma once

#include "q4k_q8_mmvq.cuh"
#include <vector>
#include <string>
#include <map>

enum Q4KCandidateID {
    CANDIDATE_UPSTREAM_MMVQ      = 0,

    CANDIDATE_WPR1_WPB1          = 101,
    CANDIDATE_WPR1_WPB2          = 102,
    CANDIDATE_WPR1_WPB4          = 104,
    CANDIDATE_WPR1_WPB8          = 108,
    CANDIDATE_WPR1_WPB16         = 116,

    CANDIDATE_WPR2_WPB2          = 202,
    CANDIDATE_WPR2_WPB4          = 204,
    CANDIDATE_WPR2_WPB8          = 208,
    CANDIDATE_WPR2_WPB16         = 216,

    CANDIDATE_WPR4_WPB4          = 404,
    CANDIDATE_WPR4_WPB8          = 408,
    CANDIDATE_WPR4_WPB16         = 416,

    CANDIDATE_WPR1_WPB4_U2       = 1042,
    CANDIDATE_WPR2_WPB4_U2       = 2042,
    CANDIDATE_WPR1_WPB8_U2       = 1082,
    CANDIDATE_WPR2_WPB8_U2       = 2082,

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
    int arch;
    Q4KCandidateID winning_candidate;
    const char * winning_name;
    double winning_median_us;
    double upstream_median_us;
    double speedup_vs_upstream;
};

const std::vector<Q4KCandidateID> & get_all_candidate_ids();
Q4KCandidateSpec get_candidate_spec(Q4KCandidateID id);
const char * get_candidate_name(Q4KCandidateID id);

void launch_q4k_candidate(
    Q4KCandidateID candidate,
    const block_q4_K * d_w,
    const block_q8_1 * d_x,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream = 0
);

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

Q4KOccupancyInfo get_q4k_candidate_occupancy_info(Q4KCandidateID candidate, int device_id);

Q4KCandidateID get_autotuned_candidate_for_shape(int N, int K, int arch = 61);
void register_autotuned_table_entry(int N, int K, int arch, Q4KCandidateID winning_candidate, double winning_med_us, double upstream_med_us);

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
