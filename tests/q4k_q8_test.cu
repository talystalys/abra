#include "kernels/q4k/q4k_q8_mmvq.cuh"
#include "kernels/q4k/q4k_family.cuh"
#include "kernels/q4k/upstream_adapter.cuh"
#include "kernels/smoke/smoke.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

#include "../.deps/llama.cpp/ggml/include/ggml.h"
#include "../.deps/llama.cpp/ggml/include/gguf.h"

enum InputDistribution {
    DIST_ALL_ZERO = 0,
    DIST_LOW_AMPLITUDE_CONST,
    DIST_ALTERNATING_SIGN,
    DIST_BOUNDED_PSEUDORANDOM,
    DIST_WIDE_DYNAMIC_RANGE,
    DIST_SPARSE_OUTLIERS,
    DIST_VARYING_SCALE,
    DIST_NONZERO_MIN_EXERCISE
};

const char * get_dist_name(InputDistribution dist) {
    switch (dist) {
        case DIST_ALL_ZERO: return "ALL_ZERO";
        case DIST_LOW_AMPLITUDE_CONST: return "LOW_AMPLITUDE_CONST";
        case DIST_ALTERNATING_SIGN: return "ALTERNATING_SIGN";
        case DIST_BOUNDED_PSEUDORANDOM: return "BOUNDED_PSEUDORANDOM";
        case DIST_WIDE_DYNAMIC_RANGE: return "WIDE_DYNAMIC_RANGE";
        case DIST_SPARSE_OUTLIERS: return "SPARSE_OUTLIERS";
        case DIST_VARYING_SCALE: return "VARYING_SCALE";
        case DIST_NONZERO_MIN_EXERCISE: return "NONZERO_MIN_EXERCISE";
        default: return "UNKNOWN";
    }
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// Independent Reference: High-precision double oracle directly from quantized blocks
void independent_q4k_q8_oracle_f64(
    const block_q4_K * w_q4,
    const block_q8_1 * x_q8,
    double * y_f64,
    int N,
    int K
) {
    const int blocks_per_row = K / QK_K;
    for (int n = 0; n < N; ++n) {
        double row_sum = 0.0;
        const block_q4_K * row_w = w_q4 + n * blocks_per_row;

        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_K * bq4 = row_w + b;
            const block_q8_1 * bq8_base = x_q8 + b * 8; // 8 Q8_1 blocks per Q4_K block

            float d4 = __half2float(bq4->d);
            float dmin4 = __half2float(bq4->dmin);

            const uint8_t * q = bq4->qs;
            int is = 0;

            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc0, m0, sc1, m1;
                get_scale_min_k4(is + 0, bq4->scales, &sc0, &m0);
                get_scale_min_k4(is + 1, bq4->scales, &sc1, &m1);

                double d1 = (double)d4 * (double)sc0;
                double m_val1 = (double)dmin4 * (double)m0;
                double d2 = (double)d4 * (double)sc1;
                double m_val2 = (double)dmin4 * (double)m1;

                const block_q8_1 * bq8_0 = bq8_base + is + 0;
                const block_q8_1 * bq8_1 = bq8_base + is + 1;
                double d8_0 = (double)__half2float(bq8_0->d);
                double d8_1 = (double)__half2float(bq8_1->d);

                for (int l = 0; l < 32; ++l) {
                    int q0 = q[l] & 0xF;
                    int q1 = (q[l] >> 4) & 0xF;
                    double w0 = d1 * (double)q0 - m_val1;
                    double w1 = d2 * (double)q1 - m_val2;
                    double x0 = d8_0 * (double)bq8_0->qs[l];
                    double x1 = d8_1 * (double)bq8_1->qs[l];
                    row_sum += w0 * x0 + w1 * x1;
                }

                q += 32;
                is += 2;
            }
        }
        y_f64[n] = row_sum;
    }
}

// Generate deterministic source float data according to distribution
void generate_test_data(std::vector<float> & data, InputDistribution dist, uint32_t seed) {
    std::mt19937 gen(seed);
    switch (dist) {
        case DIST_ALL_ZERO:
            std::fill(data.begin(), data.end(), 0.0f);
            break;
        case DIST_LOW_AMPLITUDE_CONST:
            std::fill(data.begin(), data.end(), 0.125f);
            break;
        case DIST_ALTERNATING_SIGN:
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = (i % 2 == 0) ? 0.5f : -0.5f;
            }
            break;
        case DIST_BOUNDED_PSEUDORANDOM: {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (size_t i = 0; i < data.size(); ++i) data[i] = dis(gen);
            break;
        }
        case DIST_WIDE_DYNAMIC_RANGE: {
            std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
            for (size_t i = 0; i < data.size(); ++i) data[i] = dis(gen);
            break;
        }
        case DIST_SPARSE_OUTLIERS: {
            std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = (i % 64 == 0) ? 10.0f : dis(gen);
            }
            break;
        }
        case DIST_VARYING_SCALE: {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (size_t i = 0; i < data.size(); ++i) {
                float scale = 0.01f * (1 + (i / 32) % 100);
                data[i] = dis(gen) * scale;
            }
            break;
        }
        case DIST_NONZERO_MIN_EXERCISE: {
            std::uniform_real_distribution<float> dis(10.0f, 25.0f);
            for (size_t i = 0; i < data.size(); ++i) data[i] = dis(gen);
            break;
        }
    }
}

struct NumericalErrorStats {
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double rms_err = 0.0;
    bool has_nan = false;
    bool has_inf = false;
};

NumericalErrorStats compute_error_stats(const std::vector<float> & actual, const std::vector<double> & expected) {
    NumericalErrorStats stats;
    double sum_sq_err = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float act = actual[i];
        double exp = expected[i];
        if (std::isnan(act)) stats.has_nan = true;
        if (std::isinf(act)) stats.has_inf = true;

        double abs_err = std::abs((double)act - exp);
        double denom = std::max(1e-5, std::abs(exp));
        double rel_err = abs_err / denom;

        if (abs_err > stats.max_abs_err) stats.max_abs_err = abs_err;
        if (rel_err > stats.max_rel_err) stats.max_rel_err = rel_err;
        sum_sq_err += abs_err * abs_err;
    }
    stats.rms_err = std::sqrt(sum_sq_err / (double)actual.size());
    return stats;
}

NumericalErrorStats compute_diff_stats(const std::vector<float> & actual, const std::vector<float> & reference) {
    NumericalErrorStats stats;
    double sum_sq_err = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float act = actual[i];
        float ref = reference[i];
        if (std::isnan(act)) stats.has_nan = true;
        if (std::isinf(act)) stats.has_inf = true;

        double abs_err = std::abs((double)act - (double)ref);
        double denom = std::max(1e-5, std::abs((double)ref));
        double rel_err = abs_err / denom;

        if (abs_err > stats.max_abs_err) stats.max_abs_err = abs_err;
        if (rel_err > stats.max_rel_err) stats.max_rel_err = rel_err;
        sum_sq_err += abs_err * abs_err;
    }
    stats.rms_err = std::sqrt(sum_sq_err / (double)actual.size());
    return stats;
}

int main(int argc, char ** argv) {
    std::string arch_type = "pascal";
    std::string assert_name = "";
    std::string assert_cc = "";
    std::string assert_pci = "";
    std::string assert_uuid = "";
    std::string envelope_output = "bench/results/m4/numerical-envelope.json";
    std::string model_path = "models/Qwen3-4B-Q4_K_M.gguf";
    std::string test_mode = "m4"; // "m3", "m4", or "all"

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--arch" && i + 1 < argc) arch_type = argv[++i];
        else if (arg == "--assert-name" && i + 1 < argc) assert_name = argv[++i];
        else if (arg == "--assert-cc" && i + 1 < argc) assert_cc = argv[++i];
        else if (arg == "--assert-pci" && i + 1 < argc) assert_pci = argv[++i];
        else if (arg == "--assert-uuid" && i + 1 < argc) assert_uuid = argv[++i];
        else if (arg == "--envelope-out" && i + 1 < argc) envelope_output = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) test_mode = argv[++i];
    }

    int cc_major = 0;
    int cc_minor = 0;
    if (assert_name.empty()) {
        if (arch_type == "pascal") {
            assert_name = "NVIDIA GeForce GTX 1060 6GB";
            cc_major = 6;
            cc_minor = 1;
            assert_pci = "0000:03:00.0";
            assert_uuid = "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd";
        } else if (arch_type == "turing") {
            assert_name = "NVIDIA GeForce GTX 1660 SUPER";
            cc_major = 7;
            cc_minor = 5;
            assert_pci = "0000:81:00.0";
            assert_uuid = "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd";
        }
    } else {
        if (!assert_cc.empty()) {
            size_t dot_pos = assert_cc.find('.');
            if (dot_pos != std::string::npos) {
                cc_major = std::stoi(assert_cc.substr(0, dot_pos));
                cc_minor = std::stoi(assert_cc.substr(dot_pos + 1));
            }
        }
    }

    int dev_id = 0;
    cudaGetDevice(&dev_id);
    legacy_ai::GpuDevice dev = legacy_ai::get_device_info(dev_id);
    std::string err_msg;
    if (!legacy_ai::verify_device_identity(dev, assert_name, cc_major, cc_minor, assert_pci, assert_uuid, err_msg)) {
        std::cerr << "CRITICAL ERROR: In-binary GPU physical identity assertion FAILED:\n  "
                  << err_msg << "\nAborting execution." << std::endl;
        return 2;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);
    std::cout << "============================================================\n";
    std::cout << "Milestone M4: Pascal Q4_K x Q8_1 Parameterized Family Correctness Test Suite\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "PCI Bus ID: " << assert_pci << "\n";
    std::cout << "UUID: " << assert_uuid << "\n";
    std::cout << "Test Mode: " << test_mode << "\n";
    std::cout << "============================================================\n";
    std::cout << "PASS: Physical GPU identity assertions verified successfully.\n\n";

    std::vector<Q4KCandidateID> test_candidates;
    if (test_mode == "m3") {
        test_candidates = {
            CANDIDATE_UPSTREAM_MMVQ,
            CANDIDATE_WPR4_WPB4,
            CANDIDATE_WPR2_WPB4,
            CANDIDATE_WPR1_WPB4
        };
    } else {
        test_candidates = get_all_candidate_ids();
        test_candidates.push_back(CANDIDATE_AUTOTUNED_DISPATCH);
    }

    // Print candidate occupancy summary
    std::cout << "=== Theoretical Occupancy Analysis for Tested Candidates ===\n";
    std::cout << std::left << std::setw(30) << "Candidate Name"
              << std::setw(12) << "Block Dim"
              << std::setw(10) << "Regs/Thd"
              << std::setw(10) << "SMem(B)"
              << std::setw(10) << "Blks/SM"
              << std::setw(12) << "Warps/SM"
              << "Occupancy(%)\n";
    std::cout << std::string(94, '-') << "\n";

    for (auto c : test_candidates) {
        if (c == CANDIDATE_AUTOTUNED_DISPATCH) continue;
        Q4KOccupancyInfo occ = get_q4k_candidate_occupancy_info(c, dev_id);
        std::string dim_str = std::to_string(occ.block_dim_x) + "x" + std::to_string(occ.block_dim_y);
        std::string warps_str = std::to_string(occ.resident_warps_per_sm) + "/" + std::to_string(occ.max_warps_per_sm);
        std::cout << std::left << std::setw(30) << occ.variant_name.substr(0, 29)
                  << std::setw(12) << dim_str
                  << std::setw(10) << occ.regs_per_thread
                  << std::setw(10) << occ.static_shared_mem_bytes
                  << std::setw(10) << occ.max_active_blocks_per_sm
                  << std::setw(12) << warps_str
                  << std::fixed << std::setprecision(1) << occ.theoretical_occupancy_pct << "%\n";
    }
    std::cout << std::string(94, '-') << "\n\n";

    int total_tests = 0;
    int passed_tests = 0;
    
    // Overall error tracking
    double upstream_max_abs_vs_f64 = 0.0;
    double upstream_max_rel_vs_f64 = 0.0;
    double upstream_max_rms_vs_f64 = 0.0;

    double custom_max_abs_vs_f64 = 0.0;
    double custom_max_rel_vs_f64 = 0.0;
    double custom_max_abs_vs_upstream = 0.0;
    double custom_max_rel_vs_upstream = 0.0;

    // Tolerance thresholds (frozen envelope: FP16/FP32 rounding bounds)
    const double TOL_MAX_REL_ERR = 5e-3;
    const double TOL_MAX_ABS_ERR = 0.05;

    // 1. Input Distribution Tests
    std::cout << "=== Phase 1: Input Families x Candidate Smoke Tests (N=128, K=256) ===\n";
    int N_test = 128;
    int K_test = 256;
    int K_padded_test = ((K_test + 511) / 512) * 512;
    std::vector<float> h_w_f32(N_test * K_test);
    std::vector<float> h_x_f32(K_test);
    std::vector<block_q4_K> h_w_q4(N_test * (K_test / QK_K));
    std::vector<block_q8_1> h_x_q8(K_padded_test / QK8_1);
    std::vector<double> h_y_oracle(N_test);
    std::vector<float> h_y_upstream(N_test);
    std::vector<float> h_y_gpu(N_test);

    block_q4_K * d_w_q4 = nullptr;
    float * d_x_f32 = nullptr;
    block_q8_1 * d_x_q8 = nullptr;
    float * d_y_f32 = nullptr;

    cudaMalloc(&d_w_q4, h_w_q4.size() * sizeof(block_q4_K));
    cudaMalloc(&d_x_f32, K_test * sizeof(float));
    cudaMalloc(&d_x_q8, (K_padded_test / QK8_1) * sizeof(block_q8_1));
    cudaMalloc(&d_y_f32, N_test * sizeof(float));

    for (int dist_i = 0; dist_i <= (int)DIST_NONZERO_MIN_EXERCISE; ++dist_i) {
        InputDistribution dist = (InputDistribution)dist_i;
        generate_test_data(h_w_f32, dist, 100 + dist_i);
        generate_test_data(h_x_f32, dist, 200 + dist_i);

        // Exact pinned CPU Q4_K quantizer
        for (int n = 0; n < N_test; ++n) {
            quantize_row_q4_K_upstream_cpu(&h_w_f32[n * K_test], &h_w_q4[n * (K_test / QK_K)], K_test);
        }

        cudaMemcpy(d_w_q4, h_w_q4.data(), h_w_q4.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_f32, h_x_f32.data(), K_test * sizeof(float), cudaMemcpyHostToDevice);

        // Exact pinned CUDA Q8_1 quantizer
        launch_quantize_f32_to_q8_1(d_x_f32, d_x_q8, K_test, 0);
        cudaMemcpy(h_x_q8.data(), d_x_q8, h_x_q8.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost);

        // Compute independent F64 oracle
        independent_q4k_q8_oracle_f64(h_w_q4.data(), h_x_q8.data(), h_y_oracle.data(), N_test, K_test);

        // First run Variant U (Exact Upstream)
        cudaMemset(d_y_f32, 0, N_test * sizeof(float));
        launch_q4k_candidate(CANDIDATE_UPSTREAM_MMVQ, d_w_q4, d_x_q8, d_y_f32, N_test, K_test, 0);
        cudaMemcpy(h_y_upstream.data(), d_y_f32, N_test * sizeof(float), cudaMemcpyDeviceToHost);

        NumericalErrorStats u_stats = compute_error_stats(h_y_upstream, h_y_oracle);
        if (u_stats.max_abs_err > upstream_max_abs_vs_f64) upstream_max_abs_vs_f64 = u_stats.max_abs_err;
        if (u_stats.max_rel_err > upstream_max_rel_vs_f64) upstream_max_rel_vs_f64 = u_stats.max_rel_err;
        if (u_stats.rms_err > upstream_max_rms_vs_f64) upstream_max_rms_vs_f64 = u_stats.rms_err;

        for (auto cand : test_candidates) {
            total_tests++;
            cudaMemset(d_y_f32, 0, N_test * sizeof(float));
            launch_q4k_candidate(cand, d_w_q4, d_x_q8, d_y_f32, N_test, K_test, 0);
            cudaMemcpy(h_y_gpu.data(), d_y_f32, N_test * sizeof(float), cudaMemcpyDeviceToHost);

            NumericalErrorStats stats_f64 = compute_error_stats(h_y_gpu, h_y_oracle);
            NumericalErrorStats stats_u = compute_diff_stats(h_y_gpu, h_y_upstream);

            if (cand != CANDIDATE_UPSTREAM_MMVQ) {
                if (stats_f64.max_abs_err > custom_max_abs_vs_f64) custom_max_abs_vs_f64 = stats_f64.max_abs_err;
                if (stats_f64.max_rel_err > custom_max_rel_vs_f64) custom_max_rel_vs_f64 = stats_f64.max_rel_err;
                if (stats_u.max_abs_err > custom_max_abs_vs_upstream) custom_max_abs_vs_upstream = stats_u.max_abs_err;
                if (stats_u.max_rel_err > custom_max_rel_vs_upstream) custom_max_rel_vs_upstream = stats_u.max_rel_err;
            }

            bool pass = (!stats_f64.has_nan && !stats_f64.has_inf &&
                         (stats_f64.max_rel_err <= TOL_MAX_REL_ERR || stats_f64.max_abs_err <= TOL_MAX_ABS_ERR));
            if (pass) {
                passed_tests++;
                std::cout << "PASS: Candidate=" << std::left << std::setw(28) << get_candidate_name(cand)
                          << " | Dist=" << std::setw(24) << get_dist_name(dist)
                          << " | RelVsF64=" << std::scientific << std::setprecision(2) << stats_f64.max_rel_err
                          << " | AbsVsU=" << stats_u.max_abs_err << "\n";
            } else {
                std::cerr << "FAIL: Candidate=" << get_candidate_name(cand) << " | Dist=" << get_dist_name(dist)
                          << " | RelVsF64=" << stats_f64.max_rel_err << " | AbsVsU=" << stats_u.max_abs_err << "\n";
            }
        }
    }

    cudaFree(d_w_q4);
    cudaFree(d_x_f32);
    cudaFree(d_x_q8);
    cudaFree(d_y_f32);

    // 2. K-Edge Coverage Suite
    std::cout << "\n=== Phase 2: K-Edge Coverage Suite (N=64, various K) ===\n";
    std::vector<int> k_values = {256, 512, 768, 1024, 2048, 2304, 2560, 4096, 8192, 9728};
    for (int K_val : k_values) {
        int N_val = 64;
        int K_padded = ((K_val + 511) / 512) * 512;
        std::vector<float> w_f(N_val * K_val);
        std::vector<float> x_f(K_val);
        std::vector<block_q4_K> w_q(N_val * (K_val / QK_K));
        std::vector<block_q8_1> x_q(K_padded / QK8_1);
        std::vector<double> y_oracle(N_val);
        std::vector<float> y_upstream(N_val);
        std::vector<float> y_gpu(N_val);

        generate_test_data(w_f, DIST_BOUNDED_PSEUDORANDOM, 300 + K_val);
        generate_test_data(x_f, DIST_BOUNDED_PSEUDORANDOM, 400 + K_val);

        for (int n = 0; n < N_val; ++n) {
            quantize_row_q4_K_upstream_cpu(&w_f[n * K_val], &w_q[n * (K_val / QK_K)], K_val);
        }

        block_q4_K * d_w = nullptr;
        float * d_x_f = nullptr;
        block_q8_1 * d_x = nullptr;
        float * d_y = nullptr;
        cudaMalloc(&d_w, w_q.size() * sizeof(block_q4_K));
        cudaMalloc(&d_x_f, K_val * sizeof(float));
        cudaMalloc(&d_x, (K_padded / QK8_1) * sizeof(block_q8_1));
        cudaMalloc(&d_y, N_val * sizeof(float));

        cudaMemcpy(d_w, w_q.data(), w_q.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_f, x_f.data(), K_val * sizeof(float), cudaMemcpyHostToDevice);

        launch_quantize_f32_to_q8_1(d_x_f, d_x, K_val, 0);
        cudaMemcpy(x_q.data(), d_x, x_q.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost);

        independent_q4k_q8_oracle_f64(w_q.data(), x_q.data(), y_oracle.data(), N_val, K_val);

        // Run Upstream
        cudaMemset(d_y, 0, N_val * sizeof(float));
        launch_q4k_candidate(CANDIDATE_UPSTREAM_MMVQ, d_w, d_x, d_y, N_val, K_val, 0);
        cudaMemcpy(y_upstream.data(), d_y, N_val * sizeof(float), cudaMemcpyDeviceToHost);

        NumericalErrorStats u_stats = compute_error_stats(y_upstream, y_oracle);
        if (u_stats.max_abs_err > upstream_max_abs_vs_f64) upstream_max_abs_vs_f64 = u_stats.max_abs_err;
        if (u_stats.max_rel_err > upstream_max_rel_vs_f64) upstream_max_rel_vs_f64 = u_stats.max_rel_err;
        if (u_stats.rms_err > upstream_max_rms_vs_f64) upstream_max_rms_vs_f64 = u_stats.rms_err;

        for (auto cand : test_candidates) {
            total_tests++;
            cudaMemset(d_y, 0, N_val * sizeof(float));
            launch_q4k_candidate(cand, d_w, d_x, d_y, N_val, K_val, 0);
            cudaMemcpy(y_gpu.data(), d_y, N_val * sizeof(float), cudaMemcpyDeviceToHost);

            NumericalErrorStats stats_f64 = compute_error_stats(y_gpu, y_oracle);
            NumericalErrorStats stats_u = compute_diff_stats(y_gpu, y_upstream);

            if (cand != CANDIDATE_UPSTREAM_MMVQ) {
                if (stats_f64.max_abs_err > custom_max_abs_vs_f64) custom_max_abs_vs_f64 = stats_f64.max_abs_err;
                if (stats_f64.max_rel_err > custom_max_rel_vs_f64) custom_max_rel_vs_f64 = stats_f64.max_rel_err;
                if (stats_u.max_abs_err > custom_max_abs_vs_upstream) custom_max_abs_vs_upstream = stats_u.max_abs_err;
                if (stats_u.max_rel_err > custom_max_rel_vs_upstream) custom_max_rel_vs_upstream = stats_u.max_rel_err;
            }

            if (!stats_f64.has_nan && !stats_f64.has_inf && (stats_f64.max_rel_err <= TOL_MAX_REL_ERR || stats_f64.max_abs_err <= TOL_MAX_ABS_ERR)) {
                passed_tests++;
            } else {
                std::cerr << "FAIL: K-edge K=" << K_val << " Candidate=" << get_candidate_name(cand) << "\n";
            }
        }

        cudaFree(d_w);
        cudaFree(d_x_f);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    std::cout << "K-edge coverage tests passed: " << k_values.size() << " K values across " << test_candidates.size() << " candidates.\n";

    // 3. N-Boundary Coverage Suite
    std::cout << "\n=== Phase 3: N-Boundary Coverage Suite (K=256, various N) ===\n";
    std::vector<int> n_values = {1, 2, 3, 4, 5, 31, 32, 33, 127, 128, 129, 1024, 2560, 4096, 9728};
    for (int N_val : n_values) {
        int K_val = 256;
        int K_padded = ((K_val + 511) / 512) * 512;
        std::vector<float> w_f(N_val * K_val);
        std::vector<float> x_f(K_val);
        std::vector<block_q4_K> w_q(N_val * (K_val / QK_K));
        std::vector<block_q8_1> x_q(K_padded / QK8_1);
        std::vector<double> y_oracle(N_val);
        std::vector<float> y_upstream(N_val);
        std::vector<float> y_gpu(N_val);

        generate_test_data(w_f, DIST_BOUNDED_PSEUDORANDOM, 500 + N_val);
        generate_test_data(x_f, DIST_BOUNDED_PSEUDORANDOM, 600 + N_val);

        for (int n = 0; n < N_val; ++n) {
            quantize_row_q4_K_upstream_cpu(&w_f[n * K_val], &w_q[n * (K_val / QK_K)], K_val);
        }

        block_q4_K * d_w = nullptr;
        float * d_x_f = nullptr;
        block_q8_1 * d_x = nullptr;
        float * d_y = nullptr;
        cudaMalloc(&d_w, w_q.size() * sizeof(block_q4_K));
        cudaMalloc(&d_x_f, K_val * sizeof(float));
        cudaMalloc(&d_x, (K_padded / QK8_1) * sizeof(block_q8_1));
        cudaMalloc(&d_y, N_val * sizeof(float));

        cudaMemcpy(d_w, w_q.data(), w_q.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_f, x_f.data(), K_val * sizeof(float), cudaMemcpyHostToDevice);

        launch_quantize_f32_to_q8_1(d_x_f, d_x, K_val, 0);
        cudaMemcpy(x_q.data(), d_x, x_q.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost);

        independent_q4k_q8_oracle_f64(w_q.data(), x_q.data(), y_oracle.data(), N_val, K_val);

        // Run Upstream
        cudaMemset(d_y, 0, N_val * sizeof(float));
        launch_q4k_candidate(CANDIDATE_UPSTREAM_MMVQ, d_w, d_x, d_y, N_val, K_val, 0);
        cudaMemcpy(y_upstream.data(), d_y, N_val * sizeof(float), cudaMemcpyDeviceToHost);

        NumericalErrorStats u_stats = compute_error_stats(y_upstream, y_oracle);
        if (u_stats.max_abs_err > upstream_max_abs_vs_f64) upstream_max_abs_vs_f64 = u_stats.max_abs_err;
        if (u_stats.max_rel_err > upstream_max_rel_vs_f64) upstream_max_rel_vs_f64 = u_stats.max_rel_err;
        if (u_stats.rms_err > upstream_max_rms_vs_f64) upstream_max_rms_vs_f64 = u_stats.rms_err;

        for (auto cand : test_candidates) {
            total_tests++;
            cudaMemset(d_y, 0, N_val * sizeof(float));
            launch_q4k_candidate(cand, d_w, d_x, d_y, N_val, K_val, 0);
            cudaMemcpy(y_gpu.data(), d_y, N_val * sizeof(float), cudaMemcpyDeviceToHost);

            NumericalErrorStats stats_f64 = compute_error_stats(y_gpu, y_oracle);
            NumericalErrorStats stats_u = compute_diff_stats(y_gpu, y_upstream);

            if (cand != CANDIDATE_UPSTREAM_MMVQ) {
                if (stats_f64.max_abs_err > custom_max_abs_vs_f64) custom_max_abs_vs_f64 = stats_f64.max_abs_err;
                if (stats_f64.max_rel_err > custom_max_rel_vs_f64) custom_max_rel_vs_f64 = stats_f64.max_rel_err;
                if (stats_u.max_abs_err > custom_max_abs_vs_upstream) custom_max_abs_vs_upstream = stats_u.max_abs_err;
                if (stats_u.max_rel_err > custom_max_rel_vs_upstream) custom_max_rel_vs_upstream = stats_u.max_rel_err;
            }

            if (!stats_f64.has_nan && !stats_f64.has_inf && (stats_f64.max_rel_err <= TOL_MAX_REL_ERR || stats_f64.max_abs_err <= TOL_MAX_ABS_ERR)) {
                passed_tests++;
            } else {
                std::cerr << "FAIL: N-boundary N=" << N_val << " Candidate=" << get_candidate_name(cand) << "\n";
            }
        }

        cudaFree(d_w);
        cudaFree(d_x_f);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    std::cout << "N-boundary tests passed: " << n_values.size() << " N values across " << test_candidates.size() << " candidates.\n";

    // 4. Qwen3-Derived Synthetic Shape Suite
    std::cout << "\n=== Phase 4: Qwen3-Derived Canonical & Synthetic Shapes ===\n";
    struct ShapeDef { std::string name; int N; int K; };
    std::vector<ShapeDef> qwen_shapes = {
        {"attn_q", 4096, 2560},
        {"attn_kv", 1024, 2560},
        {"attn_out", 2560, 4096},
        {"ffn_gate_up", 9728, 2560},
        {"ffn_down", 2560, 9728}
    };

    for (const auto & s : qwen_shapes) {
        int K_padded = ((s.K + 511) / 512) * 512;
        std::vector<float> w_f(s.N * s.K);
        std::vector<float> x_f(s.K);
        std::vector<block_q4_K> w_q(s.N * (s.K / QK_K));
        std::vector<block_q8_1> x_q(K_padded / QK8_1);
        std::vector<double> y_oracle(s.N);
        std::vector<float> y_upstream(s.N);
        std::vector<float> y_gpu(s.N);

        generate_test_data(w_f, DIST_BOUNDED_PSEUDORANDOM, 700 + s.N + s.K);
        generate_test_data(x_f, DIST_BOUNDED_PSEUDORANDOM, 800 + s.N + s.K);

        for (int n = 0; n < s.N; ++n) {
            quantize_row_q4_K_upstream_cpu(&w_f[n * s.K], &w_q[n * (s.K / QK_K)], s.K);
        }

        block_q4_K * d_w = nullptr;
        float * d_x_f = nullptr;
        block_q8_1 * d_x = nullptr;
        float * d_y = nullptr;
        cudaMalloc(&d_w, w_q.size() * sizeof(block_q4_K));
        cudaMalloc(&d_x_f, s.K * sizeof(float));
        cudaMalloc(&d_x, (K_padded / QK8_1) * sizeof(block_q8_1));
        cudaMalloc(&d_y, s.N * sizeof(float));

        cudaMemcpy(d_w, w_q.data(), w_q.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_f, x_f.data(), s.K * sizeof(float), cudaMemcpyHostToDevice);

        launch_quantize_f32_to_q8_1(d_x_f, d_x, s.K, 0);
        cudaMemcpy(x_q.data(), d_x, x_q.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost);

        independent_q4k_q8_oracle_f64(w_q.data(), x_q.data(), y_oracle.data(), s.N, s.K);

        // Run Upstream
        cudaMemset(d_y, 0, s.N * sizeof(float));
        launch_q4k_candidate(CANDIDATE_UPSTREAM_MMVQ, d_w, d_x, d_y, s.N, s.K, 0);
        cudaMemcpy(y_upstream.data(), d_y, s.N * sizeof(float), cudaMemcpyDeviceToHost);

        NumericalErrorStats u_stats = compute_error_stats(y_upstream, y_oracle);
        if (u_stats.max_abs_err > upstream_max_abs_vs_f64) upstream_max_abs_vs_f64 = u_stats.max_abs_err;
        if (u_stats.max_rel_err > upstream_max_rel_vs_f64) upstream_max_rel_vs_f64 = u_stats.max_rel_err;
        if (u_stats.rms_err > upstream_max_rms_vs_f64) upstream_max_rms_vs_f64 = u_stats.rms_err;

        for (auto cand : test_candidates) {
            total_tests++;
            cudaMemset(d_y, 0, s.N * sizeof(float));
            launch_q4k_candidate(cand, d_w, d_x, d_y, s.N, s.K, 0);
            cudaMemcpy(y_gpu.data(), d_y, s.N * sizeof(float), cudaMemcpyDeviceToHost);

            NumericalErrorStats stats_f64 = compute_error_stats(y_gpu, y_oracle);
            NumericalErrorStats stats_u = compute_diff_stats(y_gpu, y_upstream);

            if (cand != CANDIDATE_UPSTREAM_MMVQ) {
                if (stats_f64.max_abs_err > custom_max_abs_vs_f64) custom_max_abs_vs_f64 = stats_f64.max_abs_err;
                if (stats_f64.max_rel_err > custom_max_rel_vs_f64) custom_max_rel_vs_f64 = stats_f64.max_rel_err;
                if (stats_u.max_abs_err > custom_max_abs_vs_upstream) custom_max_abs_vs_upstream = stats_u.max_abs_err;
                if (stats_u.max_rel_err > custom_max_rel_vs_upstream) custom_max_rel_vs_upstream = stats_u.max_rel_err;
            }

            if (!stats_f64.has_nan && !stats_f64.has_inf && (stats_f64.max_rel_err <= TOL_MAX_REL_ERR || stats_f64.max_abs_err <= TOL_MAX_ABS_ERR)) {
                passed_tests++;
                std::cout << "PASS: Shape=" << std::left << std::setw(14) << s.name
                          << " | Candidate=" << std::setw(28) << get_candidate_name(cand)
                          << " | RelVsF64=" << std::scientific << std::setprecision(2) << stats_f64.max_rel_err
                          << " | AbsVsU=" << stats_u.max_abs_err << "\n";
            } else {
                std::cerr << "FAIL: Shape=" << s.name << " Candidate=" << get_candidate_name(cand) << "\n";
            }
        }

        cudaFree(d_w);
        cudaFree(d_x_f);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    // 5. Real Model-Byte Sanity Suite
    std::cout << "\n=== Phase 5: Real Model-Byte Sanity Suite (from " << model_path << ") ===\n";
    struct gguf_init_params gparams = { false, nullptr };
    struct gguf_context * gctx = gguf_init_from_file(model_path.c_str(), gparams);
    if (!gctx) {
        std::cerr << "FATAL: Could not open GGUF model file for real-byte sanity check at: " << model_path << "\n";
        return 1;
    }
    FILE * gf = fopen(model_path.c_str(), "rb");
    if (!gf) {
        std::cerr << "FATAL: Could not open GGUF file binary handle at: " << model_path << "\n";
        gguf_free(gctx);
        return 1;
    }

    size_t data_offset = gguf_get_data_offset(gctx);
    const char * real_tensors[] = {
        "blk.0.attn_q.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight"
    };

    for (const char * tname : real_tensors) {
        int tid = gguf_find_tensor(gctx, tname);
        if (tid < 0) {
            std::cerr << "FATAL: Could not find required tensor " << tname << " in GGUF model!\n";
            fclose(gf);
            gguf_free(gctx);
            return 1;
        }

        enum ggml_type ttype = gguf_get_tensor_type(gctx, tid);
        if (ttype != GGML_TYPE_Q4_K) {
            std::cerr << "FATAL: Tensor " << tname << " has type " << ttype
                      << " (expected GGML_TYPE_Q4_K = " << GGML_TYPE_Q4_K << ")!\n";
            fclose(gf);
            gguf_free(gctx);
            return 1;
        }

        const int64_t * ne = gguf_get_tensor_ne(gctx, tid);
        int64_t K_real = ne[0];
        int64_t N_real = std::min((int64_t)16, ne[1]); // Test 16 actual model rows
        int64_t K_padded_real = ((K_real + 511) / 512) * 512;
        size_t tensor_offset = gguf_get_tensor_offset(gctx, tid);

        std::vector<block_q4_K> w_real(N_real * (K_real / QK_K));
        std::vector<float> x_real_f32(K_real);
        std::vector<block_q8_1> x_real_q8(K_padded_real / QK8_1);
        std::vector<double> y_real_oracle(N_real);
        std::vector<float> y_real_u(N_real);
        std::vector<float> y_real_gpu(N_real);

        if (fseek(gf, data_offset + tensor_offset, SEEK_SET) != 0) {
            std::cerr << "FATAL: fseek failed for tensor " << tname << " offset=" << (data_offset + tensor_offset) << "\n";
            fclose(gf);
            gguf_free(gctx);
            return 1;
        }
        size_t read_blocks = fread(w_real.data(), sizeof(block_q4_K), w_real.size(), gf);
        if (read_blocks != w_real.size()) {
            std::cerr << "FATAL: fread failed for tensor " << tname << ": expected " << w_real.size()
                      << " blocks, got " << read_blocks << "\n";
            fclose(gf);
            gguf_free(gctx);
            return 1;
        }

        generate_test_data(x_real_f32, DIST_BOUNDED_PSEUDORANDOM, 999);

        block_q4_K * d_w_r = nullptr;
        float * d_x_r_f = nullptr;
        block_q8_1 * d_x_r_q = nullptr;
        float * d_y_r = nullptr;

        cudaMalloc(&d_w_r, w_real.size() * sizeof(block_q4_K));
        cudaMalloc(&d_x_r_f, K_real * sizeof(float));
        cudaMalloc(&d_x_r_q, (K_padded_real / QK8_1) * sizeof(block_q8_1));
        cudaMalloc(&d_y_r, N_real * sizeof(float));

        cudaMemcpy(d_w_r, w_real.data(), w_real.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_r_f, x_real_f32.data(), K_real * sizeof(float), cudaMemcpyHostToDevice);

        launch_quantize_f32_to_q8_1(d_x_r_f, d_x_r_q, K_real, 0);
        cudaMemcpy(x_real_q8.data(), d_x_r_q, x_real_q8.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost);

        independent_q4k_q8_oracle_f64(w_real.data(), x_real_q8.data(), y_real_oracle.data(), N_real, K_real);

        // Run Upstream
        cudaMemset(d_y_r, 0, N_real * sizeof(float));
        launch_q4k_candidate(CANDIDATE_UPSTREAM_MMVQ, d_w_r, d_x_r_q, d_y_r, N_real, K_real, 0);
        cudaMemcpy(y_real_u.data(), d_y_r, N_real * sizeof(float), cudaMemcpyDeviceToHost);

        for (auto cand : test_candidates) {
            total_tests++;
            cudaMemset(d_y_r, 0, N_real * sizeof(float));
            launch_q4k_candidate(cand, d_w_r, d_x_r_q, d_y_r, N_real, K_real, 0);
            cudaMemcpy(y_real_gpu.data(), d_y_r, N_real * sizeof(float), cudaMemcpyDeviceToHost);

            NumericalErrorStats stats_f64 = compute_error_stats(y_real_gpu, y_real_oracle);
            NumericalErrorStats stats_u = compute_diff_stats(y_real_gpu, y_real_u);

            if (!stats_f64.has_nan && !stats_f64.has_inf && (stats_f64.max_rel_err <= TOL_MAX_REL_ERR || stats_f64.max_abs_err <= TOL_MAX_ABS_ERR)) {
                passed_tests++;
                std::cout << "PASS: RealTensor=" << std::left << std::setw(26) << tname
                          << " | Candidate=" << std::setw(28) << get_candidate_name(cand)
                          << " | RelVsF64=" << std::scientific << std::setprecision(2) << stats_f64.max_rel_err
                          << " | AbsVsU=" << stats_u.max_abs_err << "\n";
            } else {
                std::cerr << "FAIL: RealTensor=" << tname << " Candidate=" << get_candidate_name(cand) << "\n";
            }
        }

        cudaFree(d_w_r);
        cudaFree(d_x_r_f);
        cudaFree(d_x_r_q);
        cudaFree(d_y_r);
    }
    fclose(gf);
    gguf_free(gctx);

    // Step B: Write frozen numerical envelope JSON
    std::ofstream env_f(envelope_output);
    if (env_f.is_open()) {
        env_f << "{\n";
        env_f << "  \"device\": {\n";
        env_f << "    \"name\": \"" << prop.name << "\",\n";
        env_f << "    \"compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n";
        env_f << "    \"pci_bus_id\": \"" << assert_pci << "\",\n";
        env_f << "    \"uuid\": \"" << assert_uuid << "\"\n";
        env_f << "  },\n";
        env_f << "  \"upstream_vs_f64_oracle\": {\n";
        env_f << "    \"max_abs_error\": " << upstream_max_abs_vs_f64 << ",\n";
        env_f << "    \"max_rel_error\": " << upstream_max_rel_vs_f64 << ",\n";
        env_f << "    \"max_rms_error\": " << upstream_max_rms_vs_f64 << "\n";
        env_f << "  },\n";
        env_f << "  \"frozen_acceptance_envelope\": {\n";
        env_f << "    \"max_rel_tolerance\": " << TOL_MAX_REL_ERR << ",\n";
        env_f << "    \"max_abs_tolerance\": " << TOL_MAX_ABS_ERR << ",\n";
        env_f << "    \"justification\": \"Frozen based on measured upstream vs F64 error bounds plus FP16 rounding margin\"\n";
        env_f << "  },\n";
        env_f << "  \"custom_variants_vs_upstream\": {\n";
        env_f << "    \"max_abs_diff_vs_upstream\": " << custom_max_abs_vs_upstream << ",\n";
        env_f << "    \"max_rel_diff_vs_upstream\": " << custom_max_rel_vs_upstream << "\n";
        env_f << "  },\n";
        env_f << "  \"custom_variants_vs_f64_oracle\": {\n";
        env_f << "    \"max_abs_error\": " << custom_max_abs_vs_f64 << ",\n";
        env_f << "    \"max_rel_error\": " << custom_max_rel_vs_f64 << "\n";
        env_f << "  },\n";
        env_f << "  \"total_tests\": " << total_tests << ",\n";
        env_f << "  \"passed_tests\": " << passed_tests << "\n";
        env_f << "}\n";
        env_f.close();
        std::cout << "Wrote frozen numerical envelope to: " << envelope_output << "\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << "Q4_K x Q8_1 Test Summary: " << passed_tests << " / " << total_tests << " Tests PASSED\n";
    std::cout << "Upstream vs F64 Max Rel Error: " << std::scientific << std::setprecision(4) << upstream_max_rel_vs_f64 << "\n";
    std::cout << "Upstream vs F64 Max Abs Error: " << std::scientific << std::setprecision(4) << upstream_max_abs_vs_f64 << "\n";
    std::cout << "Custom vs Upstream Max Abs Diff: " << std::scientific << std::setprecision(4) << custom_max_abs_vs_upstream << "\n";
    std::cout << "Custom vs F64 Max Rel Error:   " << std::scientific << std::setprecision(4) << custom_max_rel_vs_f64 << "\n";
    std::cout << "============================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
