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
#include <map>

struct BenchConfig {
    std::string shape_name;
    int N;
    int K;
    std::string classification;
    int occurrence_count;
};

struct TimingStats {
    std::vector<float> latencies_us;
    double median_us;
    double mean_us;
    double min_us;
    double max_us;
    double stddev_us;
    double spread_us;
};

TimingStats compute_timing_stats(std::vector<float> & lats) {
    std::sort(lats.begin(), lats.end());
    size_t count = lats.size();
    double med = (count % 2 == 1) ? lats[count / 2] : 0.5 * (lats[count / 2 - 1] + lats[count / 2]);

    double sum = 0.0;
    for (float v : lats) sum += v;
    double mean = sum / (double)count;

    double var = 0.0;
    for (float v : lats) var += ((double)v - mean) * ((double)v - mean);
    double sd = (count > 1) ? std::sqrt(var / (double)(count - 1)) : 0.0;

    TimingStats s;
    s.latencies_us = lats;
    s.median_us = med;
    s.mean_us = mean;
    s.min_us = lats.front();
    s.max_us = lats.back();
    s.stddev_us = sd;
    s.spread_us = lats.back() - lats.front();
    return s;
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
            const block_q8_1 * bq8_base = x_q8 + b * 8;

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

int main(int argc, char ** argv) {
    std::string arch_type = "pascal";
    std::string bench_mode = "m4-final";
    std::string output_json_path = "";
    std::string assert_name = "";
    std::string assert_cc = "";
    std::string assert_pci = "";
    std::string assert_uuid = "";
    int warmup_iters = 20;
    int timed_iters = 100;
    int arch_cc_int = 61;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--arch" && i + 1 < argc) arch_type = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) bench_mode = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_json_path = argv[++i];
        else if (arg == "--assert-name" && i + 1 < argc) assert_name = argv[++i];
        else if (arg == "--assert-cc" && i + 1 < argc) assert_cc = argv[++i];
        else if (arg == "--assert-pci" && i + 1 < argc) assert_pci = argv[++i];
        else if (arg == "--assert-uuid" && i + 1 < argc) assert_uuid = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc) warmup_iters = std::stoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) timed_iters = std::stoi(argv[++i]);
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
            arch_cc_int = 61;
        } else if (arch_type == "turing") {
            assert_name = "NVIDIA GeForce GTX 1660 SUPER";
            cc_major = 7;
            cc_minor = 5;
            assert_pci = "0000:81:00.0";
            assert_uuid = "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd";
            arch_cc_int = 75;
        }
    } else {
        if (!assert_cc.empty()) {
            size_t dot_pos = assert_cc.find('.');
            if (dot_pos != std::string::npos) {
                cc_major = std::stoi(assert_cc.substr(0, dot_pos));
                cc_minor = std::stoi(assert_cc.substr(dot_pos + 1));
                arch_cc_int = cc_major * 10 + cc_minor;
            }
        }
    }

    int dev_id = 0;
    cudaGetDevice(&dev_id);
    abra::GpuDevice dev = abra::get_device_info(dev_id);
    std::string err_msg;
    if (!abra::verify_device_identity(dev, assert_name, cc_major, cc_minor, assert_pci, assert_uuid, err_msg)) {
        std::cerr << "CRITICAL ERROR: In-binary GPU physical identity assertion FAILED:\n  "
                  << err_msg << "\nAborting execution." << std::endl;
        return 2;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);

    std::cout << "============================================================\n";
    std::cout << "Milestone M4: Pascal Q4_K x Q8_1 Autotune & Microbenchmark Suite\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "PCI Bus ID: " << assert_pci << "\n";
    std::cout << "UUID: " << assert_uuid << "\n";
    std::cout << "Benchmark Mode: " << bench_mode << "\n";
    std::cout << "Warmup iterations: " << warmup_iters << "\n";
    std::cout << "Measured iterations: " << timed_iters << "\n";
    std::cout << "============================================================\n";
    std::cout << "PASS: Physical GPU identity assertions verified successfully.\n\n";

    std::vector<BenchConfig> shapes;
    if (bench_mode == "m3") {
        shapes = {
            {"attn_q", 4096, 2560, "canonical_q4k_workload", 36},
            {"attn_kv", 1024, 2560, "canonical_q4k_workload", 54},
            {"attn_out", 2560, 4096, "canonical_q4k_workload", 36},
            {"ffn_gate_up", 9728, 2560, "canonical_q4k_workload", 72},
            {"ffn_down", 2560, 9728, "canonical_q4k_workload", 18},
            {"synth_256", 256, 256, "synthetic", 0},
            {"synth_1024", 1024, 1024, "synthetic", 0},
            {"synth_4096", 4096, 4096, "synthetic", 0}
        };
    } else {

        shapes = {
            {"attn_q", 4096, 2560, "canonical_q4k_workload", 36},
            {"attn_kv", 1024, 2560, "canonical_q4k_workload", 54},
            {"attn_out", 2560, 4096, "canonical_q4k_workload", 36},
            {"ffn_gate_up", 9728, 2560, "canonical_q4k_workload", 72},
            {"ffn_down", 2560, 9728, "canonical_q4k_workload", 18},
            {"ffn_down_synth36", 2560, 9728, "synthetic_m3_comparison", 36},
            {"guardrail_small_tail_k", 64, 256, "generalization_guardrail", 0},
            {"guardrail_mid_k_tail", 128, 768, "generalization_guardrail", 0},
            {"guardrail_large_n_mid_k", 8192, 1024, "generalization_guardrail", 0},
            {"guardrail_odd_n_boundary", 33, 2560, "generalization_guardrail", 0}
        };
    }

    std::vector<Q4KCandidateID> candidates_to_bench;
    if (bench_mode == "m3") {
        candidates_to_bench = {
            CANDIDATE_UPSTREAM_MMVQ,
            CANDIDATE_WPR4_WPB4,
            CANDIDATE_WPR2_WPB4,
            CANDIDATE_WPR1_WPB4
        };
    } else if (bench_mode == "autotune") {
        candidates_to_bench = get_all_candidate_ids();
    } else if (bench_mode == "m4-final") {
        candidates_to_bench = {
            CANDIDATE_UPSTREAM_MMVQ,
            CANDIDATE_WPR4_WPB4,
            CANDIDATE_WPR2_WPB4,
            CANDIDATE_WPR1_WPB4,
            CANDIDATE_WPR1_WPB8,
            CANDIDATE_WPR2_WPB8,
            CANDIDATE_WPR1_WPB4_U2,
            CANDIDATE_WPR2_WPB4_U2,
            CANDIDATE_AUTOTUNED_DISPATCH
        };
    }

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    std::stringstream json_results;
    json_results << "[\n";
    bool first_json_entry = true;

    std::map<std::string, std::pair<Q4KCandidateID, double>> shape_winners;
    std::map<std::string, double> shape_upstream_meds;

    for (const auto & s : shapes) {
        std::cout << "============================================================\n";
        std::cout << "Benchmarking Shape: " << s.shape_name << " (N=" << s.N << ", K=" << s.K
                  << ", Class=" << s.classification << ", Weight=" << s.occurrence_count << ")\n";
        std::cout << "============================================================\n";

        int K_padded = ((s.K + 511) / 512) * 512;
        std::vector<float> h_w_f(s.N * s.K);
        std::vector<float> h_x_f(s.K);
        std::vector<block_q4_K> h_w_q4(s.N * (s.K / QK_K));
        std::vector<block_q8_1> h_x_q8(K_padded / QK8_1);
        std::vector<double> h_y_oracle(s.N);
        std::vector<float> h_y_upstream(s.N);
        std::vector<float> h_y_check(s.N);

        std::mt19937 rng(42 + s.N + s.K);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (size_t i = 0; i < h_w_f.size(); ++i) h_w_f[i] = dis(rng);
        for (size_t i = 0; i < h_x_f.size(); ++i) h_x_f[i] = dis(rng);

        for (int n = 0; n < s.N; ++n) {
            quantize_row_q4_K_upstream_cpu(&h_w_f[n * s.K], &h_w_q4[n * (s.K / QK_K)], s.K);
        }

        block_q4_K * d_w_q4 = nullptr;
        block_q8_1 * d_x_q8 = nullptr;
        float * d_x_f32 = nullptr;
        float * d_y = nullptr;

        cudaMalloc(&d_w_q4, h_w_q4.size() * sizeof(block_q4_K));
        cudaMalloc(&d_x_q8, (K_padded / QK8_1) * sizeof(block_q8_1));
        cudaMalloc(&d_x_f32, s.K * sizeof(float));
        cudaMalloc(&d_y, s.N * sizeof(float));

        cudaMemcpy(d_w_q4, h_w_q4.data(), h_w_q4.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_f32, h_x_f.data(), s.K * sizeof(float), cudaMemcpyHostToDevice);

        launch_quantize_f32_to_q8_1(d_x_f32, d_x_q8, s.K, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(h_x_q8.data(), d_x_q8, h_x_q8.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost);

        independent_q4k_q8_oracle_f64(h_w_q4.data(), h_x_q8.data(), h_y_oracle.data(), s.N, s.K);

        cudaMemset(d_y, 0, s.N * sizeof(float));
        launch_q4k_candidate(CANDIDATE_UPSTREAM_MMVQ, d_w_q4, d_x_q8, d_y, s.N, s.K, 0);
        cudaMemcpy(h_y_upstream.data(), d_y, s.N * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> quantize_only_lats(timed_iters);
        for (int w = 0; w < warmup_iters; ++w) {
            launch_quantize_f32_to_q8_1(d_x_f32, d_x_q8, s.K, 0);
        }
        cudaDeviceSynchronize();

        for (int it = 0; it < timed_iters; ++it) {
            cudaEventRecord(start_event, 0);
            launch_quantize_f32_to_q8_1(d_x_f32, d_x_q8, s.K, 0);
            cudaEventRecord(stop_event, 0);
            cudaEventSynchronize(stop_event);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start_event, stop_event);
            quantize_only_lats[it] = ms * 1000.0f;
        }
        TimingStats q_stats = compute_timing_stats(quantize_only_lats);

        double q4_bytes = (double)s.N * ((double)s.K / 256.0) * (double)sizeof(block_q4_K);
        double q8_bytes = ((double)s.K / 32.0) * (double)sizeof(block_q8_1);
        double out_bytes = (double)s.N * sizeof(float);
        double total_tensor_bytes = q4_bytes + q8_bytes + out_bytes;

        double best_cand_med = 1e9;
        Q4KCandidateID best_cand_id = CANDIDATE_UPSTREAM_MMVQ;

        for (auto cand : candidates_to_bench) {

            for (int w = 0; w < warmup_iters; ++w) {
                launch_q4k_candidate(cand, d_w_q4, d_x_q8, d_y, s.N, s.K, 0);
            }
            cudaDeviceSynchronize();

            std::vector<float> mul_lats(timed_iters);
            for (int it = 0; it < timed_iters; ++it) {
                cudaEventRecord(start_event, 0);
                launch_q4k_candidate(cand, d_w_q4, d_x_q8, d_y, s.N, s.K, 0);
                cudaEventRecord(stop_event, 0);
                cudaEventSynchronize(stop_event);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start_event, stop_event);
                mul_lats[it] = ms * 1000.0f;
            }
            TimingStats mul_stats = compute_timing_stats(mul_lats);

            if (cand != CANDIDATE_AUTOTUNED_DISPATCH) {
                if (mul_stats.median_us < best_cand_med) {
                    best_cand_med = mul_stats.median_us;
                    best_cand_id = cand;
                }
            }

            if (cand == CANDIDATE_UPSTREAM_MMVQ) {
                shape_upstream_meds[s.shape_name] = mul_stats.median_us;
            }

            cudaMemcpy(h_y_check.data(), d_y, s.N * sizeof(float), cudaMemcpyDeviceToHost);

            double max_abs_vs_oracle = 0.0;
            double max_rel_vs_oracle = 0.0;
            double max_abs_vs_upstream = 0.0;
            bool has_nan = false;
            bool has_inf = false;

            for (int n = 0; n < s.N; ++n) {
                float act = h_y_check[n];
                double exp = h_y_oracle[n];
                float u_val = h_y_upstream[n];

                if (std::isnan(act)) has_nan = true;
                if (std::isinf(act)) has_inf = true;

                double abs_err_orc = std::abs((double)act - exp);
                double rel_err_orc = abs_err_orc / std::max(1e-5, std::abs(exp));
                double abs_err_u = std::abs((double)act - (double)u_val);

                if (abs_err_orc > max_abs_vs_oracle) max_abs_vs_oracle = abs_err_orc;
                if (rel_err_orc > max_rel_vs_oracle) max_rel_vs_oracle = rel_err_orc;
                if (abs_err_u > max_abs_vs_upstream) max_abs_vs_upstream = abs_err_u;
            }

            bool is_verified = (!has_nan && !has_inf && (max_rel_vs_oracle <= 5e-3 || max_abs_vs_oracle <= 0.05));

            for (int w = 0; w < warmup_iters; ++w) {
                launch_q4k_candidate_combined_pipeline(cand, d_w_q4, d_x_f32, d_x_q8, d_y, s.N, s.K, 0);
            }
            cudaDeviceSynchronize();

            std::vector<float> comb_lats(timed_iters);
            for (int it = 0; it < timed_iters; ++it) {
                cudaEventRecord(start_event, 0);
                launch_q4k_candidate_combined_pipeline(cand, d_w_q4, d_x_f32, d_x_q8, d_y, s.N, s.K, 0);
                cudaEventRecord(stop_event, 0);
                cudaEventSynchronize(stop_event);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start_event, stop_event);
                comb_lats[it] = ms * 1000.0f;
            }
            TimingStats comb_stats = compute_timing_stats(comb_lats);

            double gmac_s = ((double)s.N * (double)s.K) / (mul_stats.median_us * 1e-6) / 1e9;
            double eff_gbps = total_tensor_bytes / (mul_stats.median_us * 1e-6) / (1024.0 * 1024.0 * 1024.0);

            std::cout << "  " << std::left << std::setw(28) << get_candidate_name(cand)
                      << " | Med=" << std::fixed << std::setprecision(2) << std::setw(7) << mul_stats.median_us << " us"
                      << " | Mean=" << std::setw(7) << mul_stats.mean_us << " us"
                      << " | Min=" << std::setw(7) << mul_stats.min_us << " us"
                      << " | Max=" << std::setw(7) << mul_stats.max_us << " us"
                      << " | Spread=" << std::setw(6) << mul_stats.spread_us << " us"
                      << " | GMAC/s=" << std::setw(7) << gmac_s
                      << " | Eff=" << std::setw(6) << eff_gbps << " GiB/s"
                      << " | CombMed=" << std::setw(7) << comb_stats.median_us << " us"
                      << " | " << (is_verified ? "VERIFIED" : "FAIL") << "\n";

            if (!first_json_entry) json_results << ",\n";
            first_json_entry = false;

            json_results << "    {\n";
            json_results << "      \"shape_name\": \"" << s.shape_name << "\",\n";
            json_results << "      \"N\": " << s.N << ",\n";
            json_results << "      \"K\": " << s.K << ",\n";
            json_results << "      \"classification\": \"" << s.classification << "\",\n";
            json_results << "      \"occurrence_count\": " << s.occurrence_count << ",\n";
            json_results << "      \"candidate_id\": " << (int)cand << ",\n";
            json_results << "      \"variant_id\": " << (int)cand << ",\n";
            json_results << "      \"variant_name\": \"" << get_candidate_name(cand) << "\",\n";
            json_results << "      \"median_us\": " << mul_stats.median_us << ",\n";
            json_results << "      \"mean_us\": " << mul_stats.mean_us << ",\n";
            json_results << "      \"min_us\": " << mul_stats.min_us << ",\n";
            json_results << "      \"max_us\": " << mul_stats.max_us << ",\n";
            json_results << "      \"stddev_us\": " << mul_stats.stddev_us << ",\n";
            json_results << "      \"spread_us\": " << mul_stats.spread_us << ",\n";
            json_results << "      \"gmac_per_sec\": " << gmac_s << ",\n";
            json_results << "      \"effective_throughput_gbps\": " << eff_gbps << ",\n";
            json_results << "      \"quantize_only_median_us\": " << q_stats.median_us << ",\n";
            json_results << "      \"combined_median_us\": " << comb_stats.median_us << ",\n";
            json_results << "      \"combined_mean_us\": " << comb_stats.mean_us << ",\n";
            json_results << "      \"combined_min_us\": " << comb_stats.min_us << ",\n";
            json_results << "      \"combined_max_us\": " << comb_stats.max_us << ",\n";
            json_results << "      \"combined_spread_us\": " << comb_stats.spread_us << ",\n";
            json_results << "      \"verified_correct\": " << (is_verified ? "true" : "false") << ",\n";
            json_results << "      \"verification\": {\n";
            json_results << "        \"verified\": " << (is_verified ? "true" : "false") << ",\n";
            json_results << "        \"max_abs_diff_vs_oracle\": " << max_abs_vs_oracle << ",\n";
            json_results << "        \"max_rel_diff_vs_oracle\": " << max_rel_vs_oracle << ",\n";
            json_results << "        \"max_abs_diff_vs_upstream\": " << max_abs_vs_upstream << ",\n";
            json_results << "        \"has_nan\": " << (has_nan ? "true" : "false") << ",\n";
            json_results << "        \"has_inf\": " << (has_inf ? "true" : "false") << "\n";
            json_results << "      },\n";
            json_results << "      \"latencies_us\": [";
            for (size_t i = 0; i < mul_stats.latencies_us.size(); ++i) {
                json_results << mul_stats.latencies_us[i] << (i + 1 < mul_stats.latencies_us.size() ? ", " : "");
            }
            json_results << "],\n";
            json_results << "      \"combined_latencies_us\": [";
            for (size_t i = 0; i < comb_stats.latencies_us.size(); ++i) {
                json_results << comb_stats.latencies_us[i] << (i + 1 < comb_stats.latencies_us.size() ? ", " : "");
            }
            json_results << "]\n";
            json_results << "    }";
        }

        shape_winners[s.shape_name] = {best_cand_id, best_cand_med};
        double u_med = shape_upstream_meds[s.shape_name];
        double sp = (u_med > 0 && best_cand_med > 0) ? u_med / best_cand_med : 1.0;
        std::cout << ">>> Winner for " << s.shape_name << ": " << get_candidate_name(best_cand_id)
                  << " (" << best_cand_med << " us, " << sp << "x vs Upstream)\n\n";

        register_autotuned_table_entry(s.N, s.K, arch_cc_int, best_cand_id, best_cand_med, u_med);

        cudaFree(d_w_q4);
        cudaFree(d_x_q8);
        cudaFree(d_x_f32);
        cudaFree(d_y);
    }
    json_results << "\n  ]";

    if (!output_json_path.empty()) {
        std::ofstream out_f(output_json_path);
        if (out_f.is_open()) {
            out_f << "{\n";
            out_f << "  \"milestone\": \"M4\",\n";
            out_f << "  \"bench_mode\": \"" << bench_mode << "\",\n";
            out_f << "  \"device\": {\n";
            out_f << "    \"name\": \"" << prop.name << "\",\n";
            out_f << "    \"compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n";
            out_f << "    \"pci_bus_id\": \"" << assert_pci << "\",\n";
            out_f << "    \"uuid\": \"" << assert_uuid << "\"\n";
            out_f << "  },\n";
            out_f << "  \"results\": " << json_results.str() << "\n";
            out_f << "}\n";
            out_f.close();
            std::cout << "Wrote benchmark results to: " << output_json_path << "\n";
        }
    }

    return 0;
}
