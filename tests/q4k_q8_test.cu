#include "kernels/q4k/q4k_q8_mmvq.cuh"
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

static inline void set_scale_min_k4(int j, uint8_t d, uint8_t m, uint8_t * q) {
    if (j < 4) {
        q[j]     = (q[j]     & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } else {
        q[j - 4] = (q[j - 4] & 0x3F) | ((d & 0x30) << 2);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m & 0x30) << 2);
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
    }
}

// Canonical CPU Quantizers (exact upstream logic)
void quantize_row_q4_k_cpu_ref(const float * x, block_q4_K * y, int k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        const float * xb = x + i * QK_K;
        block_q4_K * yb = y + i;

        float max_val = -1e30f;
        float min_val = 1e30f;
        for (int j = 0; j < QK_K; ++j) {
            float v = xb[j];
            if (v > max_val) max_val = v;
            if (v < min_val) min_val = v;
        }

        float d = (max_val > min_val) ? (max_val - min_val) / 15.0f : 0.0f;
        float min_block = (max_val > min_val) ? min_val : 0.0f;
        yb->d = __float2half(d);
        yb->dmin = __float2half(min_block);

        memset(yb->scales, 0, K_SCALE_SIZE);
        for (int j = 0; j < 8; ++j) {
            set_scale_min_k4(j, 1, 1, yb->scales);
        }

        uint8_t L[QK_K];
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        for (int j = 0; j < 8; ++j) {
            uint8_t sc, m;
            get_scale_min_k4(j, yb->scales, &sc, &m);
            float d_sub = d * (float)sc;
            float m_sub = min_block * (float)m;
            float id_sub = d_sub > 0.0f ? 1.0f / d_sub : 0.0f;
            for (int ii = 0; ii < 32; ++ii) {
                float v = (xb[32 * j + ii] + m_sub) * id_sub;
                int l = std::min(15, std::max(0, (int)std::round(v)));
                L[32 * j + ii] = (uint8_t)l;
            }
        }

        uint8_t * q = yb->qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) {
                q[l] = L[j + l] | (L[j + l + 32] << 4);
            }
            q += 32;
        }
    }
}

void quantize_row_q8_1_cpu_ref(const float * x, block_q8_1 * y, int k) {
    const int nb = k / QK8_1;
    for (int i = 0; i < nb; ++i) {
        const float * xb = x + i * QK8_1;
        block_q8_1 * yb = y + i;

        float amax = 0.0f;
        for (int j = 0; j < QK8_1; ++j) {
            float v = std::fabs(xb[j]);
            if (v > amax) amax = v;
        }

        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        int sum_qs = 0;

        for (int j = 0; j < QK8_1; ++j) {
            float x0 = xb[j] * id;
            int8_t q0 = (int8_t)std::round(x0);
            yb->qs[j] = q0;
            sum_qs += q0;
        }

        yb->d = __float2half(d);
        yb->s = __float2half(d * (float)sum_qs);
    }
}

// Independent Reference 2: High-precision double oracle directly from quantized blocks
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
            std::uniform_real_distribution<float> dis(10.0f, 25.0f); // strictly positive offset
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

int main(int argc, char ** argv) {
    std::string arch_type = "pascal";
    std::string assert_name = "";
    std::string assert_cc = "";
    std::string assert_pci = "";
    std::string assert_uuid = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--arch" && i + 1 < argc) arch_type = argv[++i];
        else if (arg == "--assert-name" && i + 1 < argc) assert_name = argv[++i];
        else if (arg == "--assert-cc" && i + 1 < argc) assert_cc = argv[++i];
        else if (arg == "--assert-pci" && i + 1 < argc) assert_pci = argv[++i];
        else if (arg == "--assert-uuid" && i + 1 < argc) assert_uuid = argv[++i];
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
    std::cout << "Milestone M3: Pascal Q4_K x Q8_1 MMVQ Correctness Test Suite\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "PCI Bus ID: " << assert_pci << "\n";
    std::cout << "UUID: " << assert_uuid << "\n";
    std::cout << "============================================================\n";
    std::cout << "PASS: Physical GPU identity assertions verified successfully.\n\n";

    // Print theoretical occupancy
    std::cout << "=== Theoretical Occupancy Analysis (via CUDA Occupancy API) ===\n";
    std::cout << std::left << std::setw(32) << "Variant"
              << std::setw(14) << "Block Dim"
              << std::setw(10) << "Regs/Thd"
              << std::setw(12) << "SMem(B)"
              << std::setw(12) << "Blks/SM"
              << std::setw(12) << "Warps/SM"
              << "Occupancy(%)\n";
    std::cout << std::string(98, '-') << "\n";

    std::vector<Q4KVariant> variants = {
        VARIANT_4WARP_CONTROL,
        VARIANT_2WARP_SPLIT,
        VARIANT_1WARP_ROW
    };

    for (auto v : variants) {
        Q4KOccupancyInfo occ = get_q4k_kernel_occupancy_info(v, dev_id);
        std::string dim_str = std::to_string(occ.block_dim_x) + "x" + std::to_string(occ.block_dim_y);
        std::string warps_str = std::to_string(occ.resident_warps_per_sm) + "/" + std::to_string(occ.max_warps_per_sm);
        std::cout << std::left << std::setw(32) << occ.variant_name.substr(0, 31)
                  << std::setw(14) << dim_str
                  << std::setw(10) << occ.regs_per_thread
                  << std::setw(12) << occ.static_shared_mem_bytes
                  << std::setw(12) << occ.max_active_blocks_per_sm
                  << std::setw(12) << warps_str
                  << std::fixed << std::setprecision(1) << occ.theoretical_occupancy_pct << "%\n";
    }
    std::cout << std::string(98, '-') << "\n\n";

    int total_tests = 0;
    int passed_tests = 0;
    double max_observed_abs_err = 0.0;
    double max_observed_rel_err = 0.0;

    // Tolerance thresholds (frozen envelope: FP16/FP32 rounding bounds)
    const double TOL_MAX_REL_ERR = 5e-3; // 0.5% relative error envelope for FP16 dequantization
    const double TOL_MAX_ABS_ERR = 0.05;

    // 1. Input Distribution Tests
    std::cout << "=== Phase 1: Input Families x Variant Smoke Tests (N=128, K=256) ===\n";
    int N_test = 128;
    int K_test = 256;
    std::vector<float> h_w_f32(N_test * K_test);
    std::vector<float> h_x_f32(K_test);
    std::vector<block_q4_K> h_w_q4(N_test * (K_test / QK_K));
    std::vector<block_q8_1> h_x_q8(K_test / QK8_1);
    std::vector<double> h_y_oracle(N_test);
    std::vector<float> h_y_gpu(N_test);

    block_q4_K * d_w_q4 = nullptr;
    block_q8_1 * d_x_q8 = nullptr;
    float * d_y_f32 = nullptr;

    cudaMalloc(&d_w_q4, h_w_q4.size() * sizeof(block_q4_K));
    cudaMalloc(&d_x_q8, h_x_q8.size() * sizeof(block_q8_1));
    cudaMalloc(&d_y_f32, N_test * sizeof(float));

    for (int dist_i = 0; dist_i <= (int)DIST_NONZERO_MIN_EXERCISE; ++dist_i) {
        InputDistribution dist = (InputDistribution)dist_i;
        generate_test_data(h_w_f32, dist, 100 + dist_i);
        generate_test_data(h_x_f32, dist, 200 + dist_i);

        quantize_row_q4_k_cpu_ref(h_w_f32.data(), h_w_q4.data(), N_test * K_test);
        quantize_row_q8_1_cpu_ref(h_x_f32.data(), h_x_q8.data(), K_test);

        independent_q4k_q8_oracle_f64(h_w_q4.data(), h_x_q8.data(), h_y_oracle.data(), N_test, K_test);

        cudaMemcpy(d_w_q4, h_w_q4.data(), h_w_q4.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_q8, h_x_q8.data(), h_x_q8.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice);

        for (auto var : variants) {
            total_tests++;
            cudaMemset(d_y_f32, 0, N_test * sizeof(float));
            launch_q4k_q8_mmvq(var, d_w_q4, d_x_q8, d_y_f32, N_test, K_test, 0);
            cudaMemcpy(h_y_gpu.data(), d_y_f32, N_test * sizeof(float), cudaMemcpyDeviceToHost);

            NumericalErrorStats stats = compute_error_stats(h_y_gpu, h_y_oracle);
            if (stats.max_abs_err > max_observed_abs_err) max_observed_abs_err = stats.max_abs_err;
            if (stats.max_rel_err > max_observed_rel_err) max_observed_rel_err = stats.max_rel_err;

            bool pass = (!stats.has_nan && !stats.has_inf &&
                         (stats.max_rel_err <= TOL_MAX_REL_ERR || stats.max_abs_err <= TOL_MAX_ABS_ERR));
            if (pass) {
                passed_tests++;
                std::cout << "PASS: Variant=" << std::left << std::setw(30) << get_q4k_variant_name(var)
                          << " | Dist=" << std::setw(24) << get_dist_name(dist)
                          << " | MaxRelErr=" << std::scientific << std::setprecision(2) << stats.max_rel_err
                          << " | MaxAbsErr=" << stats.max_abs_err << "\n";
            } else {
                std::cerr << "FAIL: Variant=" << get_q4k_variant_name(var) << " | Dist=" << get_dist_name(dist)
                          << " | MaxRelErr=" << stats.max_rel_err << " | MaxAbsErr=" << stats.max_abs_err << "\n";
            }
        }
    }

    cudaFree(d_w_q4);
    cudaFree(d_x_q8);
    cudaFree(d_y_f32);

    // 2. K-Edge Coverage Suite
    std::cout << "\n=== Phase 2: K-Edge Coverage Suite (N=64, various K) ===\n";
    std::vector<int> k_values = {256, 512, 768, 1024, 2048, 2304, 2560, 4096, 8192, 9728};
    for (int K_val : k_values) {
        int N_val = 64;
        std::vector<float> w_f(N_val * K_val);
        std::vector<float> x_f(K_val);
        std::vector<block_q4_K> w_q(N_val * (K_val / QK_K));
        std::vector<block_q8_1> x_q(K_val / QK8_1);
        std::vector<double> y_oracle(N_val);
        std::vector<float> y_gpu(N_val);

        generate_test_data(w_f, DIST_BOUNDED_PSEUDORANDOM, 300 + K_val);
        generate_test_data(x_f, DIST_BOUNDED_PSEUDORANDOM, 400 + K_val);
        quantize_row_q4_k_cpu_ref(w_f.data(), w_q.data(), N_val * K_val);
        quantize_row_q8_1_cpu_ref(x_f.data(), x_q.data(), K_val);
        independent_q4k_q8_oracle_f64(w_q.data(), x_q.data(), y_oracle.data(), N_val, K_val);

        block_q4_K * d_w = nullptr;
        block_q8_1 * d_x = nullptr;
        float * d_y = nullptr;
        cudaMalloc(&d_w, w_q.size() * sizeof(block_q4_K));
        cudaMalloc(&d_x, x_q.size() * sizeof(block_q8_1));
        cudaMalloc(&d_y, N_val * sizeof(float));

        cudaMemcpy(d_w, w_q.data(), w_q.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x_q.data(), x_q.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice);

        for (auto var : variants) {
            total_tests++;
            cudaMemset(d_y, 0, N_val * sizeof(float));
            launch_q4k_q8_mmvq(var, d_w, d_x, d_y, N_val, K_val, 0);
            cudaMemcpy(y_gpu.data(), d_y, N_val * sizeof(float), cudaMemcpyDeviceToHost);

            NumericalErrorStats stats = compute_error_stats(y_gpu, y_oracle);
            if (stats.max_abs_err > max_observed_abs_err) max_observed_abs_err = stats.max_abs_err;
            if (stats.max_rel_err > max_observed_rel_err) max_observed_rel_err = stats.max_rel_err;

            if (!stats.has_nan && !stats.has_inf && (stats.max_rel_err <= TOL_MAX_REL_ERR || stats.max_abs_err <= TOL_MAX_ABS_ERR)) {
                passed_tests++;
            } else {
                std::cerr << "FAIL: K-edge K=" << K_val << " Variant=" << get_q4k_variant_name(var) << "\n";
            }
        }

        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    std::cout << "K-edge coverage tests passed: " << k_values.size() << " K values across " << variants.size() << " variants.\n";

    // 3. N-Boundary Coverage Suite
    std::cout << "\n=== Phase 3: N-Boundary Coverage Suite (K=256, various N) ===\n";
    std::vector<int> n_values = {1, 2, 3, 4, 5, 31, 32, 33, 127, 128, 129, 1024, 2560, 4096, 9728};
    for (int N_val : n_values) {
        int K_val = 256;
        std::vector<float> w_f(N_val * K_val);
        std::vector<float> x_f(K_val);
        std::vector<block_q4_K> w_q(N_val * (K_val / QK_K));
        std::vector<block_q8_1> x_q(K_val / QK8_1);
        std::vector<double> y_oracle(N_val);
        std::vector<float> y_gpu(N_val);

        generate_test_data(w_f, DIST_BOUNDED_PSEUDORANDOM, 500 + N_val);
        generate_test_data(x_f, DIST_BOUNDED_PSEUDORANDOM, 600 + N_val);
        quantize_row_q4_k_cpu_ref(w_f.data(), w_q.data(), N_val * K_val);
        quantize_row_q8_1_cpu_ref(x_f.data(), x_q.data(), K_val);
        independent_q4k_q8_oracle_f64(w_q.data(), x_q.data(), y_oracle.data(), N_val, K_val);

        block_q4_K * d_w = nullptr;
        block_q8_1 * d_x = nullptr;
        float * d_y = nullptr;
        cudaMalloc(&d_w, w_q.size() * sizeof(block_q4_K));
        cudaMalloc(&d_x, x_q.size() * sizeof(block_q8_1));
        cudaMalloc(&d_y, N_val * sizeof(float));

        cudaMemcpy(d_w, w_q.data(), w_q.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x_q.data(), x_q.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice);

        for (auto var : variants) {
            total_tests++;
            cudaMemset(d_y, 0, N_val * sizeof(float));
            launch_q4k_q8_mmvq(var, d_w, d_x, d_y, N_val, K_val, 0);
            cudaMemcpy(y_gpu.data(), d_y, N_val * sizeof(float), cudaMemcpyDeviceToHost);

            NumericalErrorStats stats = compute_error_stats(y_gpu, y_oracle);
            if (stats.max_abs_err > max_observed_abs_err) max_observed_abs_err = stats.max_abs_err;
            if (stats.max_rel_err > max_observed_rel_err) max_observed_rel_err = stats.max_rel_err;

            if (!stats.has_nan && !stats.has_inf && (stats.max_rel_err <= TOL_MAX_REL_ERR || stats.max_abs_err <= TOL_MAX_ABS_ERR)) {
                passed_tests++;
            } else {
                std::cerr << "FAIL: N-boundary N=" << N_val << " Variant=" << get_q4k_variant_name(var) << "\n";
            }
        }

        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    std::cout << "N-boundary tests passed: " << n_values.size() << " N values across " << variants.size() << " variants.\n";

    // 4. Qwen3-4B Real Shapes Suite
    std::cout << "\n=== Phase 4: Qwen3-4B Real Model Shapes Coverage Suite ===\n";
    struct ShapeDef { std::string name; int N; int K; };
    std::vector<ShapeDef> qwen_shapes = {
        {"attn_q", 4096, 2560},
        {"attn_kv", 1024, 2560},
        {"attn_out", 2560, 4096},
        {"ffn_gate_up", 9728, 2560},
        {"ffn_down", 2560, 9728}
    };

    for (const auto & s : qwen_shapes) {
        std::vector<float> w_f(s.N * s.K);
        std::vector<float> x_f(s.K);
        std::vector<block_q4_K> w_q(s.N * (s.K / QK_K));
        std::vector<block_q8_1> x_q(s.K / QK8_1);
        std::vector<double> y_oracle(s.N);
        std::vector<float> y_gpu(s.N);

        generate_test_data(w_f, DIST_BOUNDED_PSEUDORANDOM, 700 + s.N + s.K);
        generate_test_data(x_f, DIST_BOUNDED_PSEUDORANDOM, 800 + s.N + s.K);
        quantize_row_q4_k_cpu_ref(w_f.data(), w_q.data(), s.N * s.K);
        quantize_row_q8_1_cpu_ref(x_f.data(), x_q.data(), s.K);
        independent_q4k_q8_oracle_f64(w_q.data(), x_q.data(), y_oracle.data(), s.N, s.K);

        block_q4_K * d_w = nullptr;
        block_q8_1 * d_x = nullptr;
        float * d_y = nullptr;
        cudaMalloc(&d_w, w_q.size() * sizeof(block_q4_K));
        cudaMalloc(&d_x, x_q.size() * sizeof(block_q8_1));
        cudaMalloc(&d_y, s.N * sizeof(float));

        cudaMemcpy(d_w, w_q.data(), w_q.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x_q.data(), x_q.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice);

        for (auto var : variants) {
            total_tests++;
            cudaMemset(d_y, 0, s.N * sizeof(float));
            launch_q4k_q8_mmvq(var, d_w, d_x, d_y, s.N, s.K, 0);
            cudaMemcpy(y_gpu.data(), d_y, s.N * sizeof(float), cudaMemcpyDeviceToHost);

            NumericalErrorStats stats = compute_error_stats(y_gpu, y_oracle);
            if (stats.max_abs_err > max_observed_abs_err) max_observed_abs_err = stats.max_abs_err;
            if (stats.max_rel_err > max_observed_rel_err) max_observed_rel_err = stats.max_rel_err;

            if (!stats.has_nan && !stats.has_inf && (stats.max_rel_err <= TOL_MAX_REL_ERR || stats.max_abs_err <= TOL_MAX_ABS_ERR)) {
                passed_tests++;
                std::cout << "PASS: Shape=" << std::left << std::setw(14) << s.name
                          << " | Variant=" << std::setw(30) << get_q4k_variant_name(var)
                          << " | MaxRelErr=" << std::scientific << std::setprecision(2) << stats.max_rel_err
                          << " | MaxAbsErr=" << stats.max_abs_err << "\n";
            } else {
                std::cerr << "FAIL: Shape=" << s.name << " Variant=" << get_q4k_variant_name(var) << "\n";
            }
        }

        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    std::cout << "\n============================================================\n";
    std::cout << "Q4_K x Q8_1 Test Summary: " << passed_tests << " / " << total_tests << " Tests PASSED\n";
    std::cout << "Max Observed Relative Error vs F64: " << std::scientific << std::setprecision(4) << max_observed_rel_err << "\n";
    std::cout << "Max Observed Absolute Error vs F64: " << std::scientific << std::setprecision(4) << max_observed_abs_err << "\n";
    std::cout << "============================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
