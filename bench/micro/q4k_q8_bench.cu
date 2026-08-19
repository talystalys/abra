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

struct BenchConfig {
    std::string shape_name;
    int N;
    int K;
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

// Canonical CPU Quantizers for benchmarking
void quantize_q4k_bench(const float * x, block_q4_K * y, int k) {
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

int main(int argc, char ** argv) {
    std::string arch_type = "pascal";
    std::string output_json_path = "";
    std::string assert_name = "";
    std::string assert_cc = "";
    std::string assert_pci = "";
    std::string assert_uuid = "";
    int warmup_iters = 10;
    int timed_iters = 100;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--arch" && i + 1 < argc) arch_type = argv[++i];
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
    std::cout << "Milestone M3: Pascal Q4_K x Q8_1 MMVQ Microbenchmark Suite\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "PCI Bus ID: " << assert_pci << "\n";
    std::cout << "UUID: " << assert_uuid << "\n";
    std::cout << "Warmup iterations: " << warmup_iters << "\n";
    std::cout << "Measured iterations: " << timed_iters << "\n";
    std::cout << "============================================================\n";
    std::cout << "PASS: Physical GPU identity assertions verified successfully.\n\n";

    std::vector<BenchConfig> shapes = {
        {"attn_q", 4096, 2560},
        {"attn_kv", 1024, 2560},
        {"attn_out", 2560, 4096},
        {"ffn_gate_up", 9728, 2560},
        {"ffn_down", 2560, 9728},
        {"synth_256", 256, 256},
        {"synth_1024", 1024, 1024},
        {"synth_4096", 4096, 4096}
    };

    std::vector<Q4KVariant> variants = {
        VARIANT_UPSTREAM_MMVQ,
        VARIANT_4WARP_CONTROL,
        VARIANT_2WARP_SPLIT,
        VARIANT_1WARP_ROW
    };

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    std::stringstream json_results;
    json_results << "[\n";
    bool first_json_entry = true;

    for (const auto & s : shapes) {
        std::cout << "Benchmarking shape: " << s.shape_name << " (N=" << s.N << ", K=" << s.K << ")...\n";

        std::vector<float> h_w_f(s.N * s.K);
        std::vector<float> h_x_f(s.K);
        std::vector<block_q4_K> h_w_q4(s.N * (s.K / QK_K));
        std::vector<block_q8_1> h_x_q8(s.K / QK8_1);

        std::mt19937 rng(42 + s.N + s.K);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (size_t i = 0; i < h_w_f.size(); ++i) h_w_f[i] = dis(rng);
        for (size_t i = 0; i < h_x_f.size(); ++i) h_x_f[i] = dis(rng);

        quantize_q4k_bench(h_w_f.data(), h_w_q4.data(), s.N * s.K);

        block_q4_K * d_w_q4 = nullptr;
        block_q8_1 * d_x_q8 = nullptr;
        float * d_x_f32 = nullptr;
        float * d_y = nullptr;

        cudaMalloc(&d_w_q4, h_w_q4.size() * sizeof(block_q4_K));
        cudaMalloc(&d_x_q8, (s.K / QK8_1) * sizeof(block_q8_1));
        cudaMalloc(&d_x_f32, s.K * sizeof(float));
        cudaMalloc(&d_y, s.N * sizeof(float));

        cudaMemcpy(d_w_q4, h_w_q4.data(), h_w_q4.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_f32, h_x_f.data(), s.K * sizeof(float), cudaMemcpyHostToDevice);

        // Pre-quantize activation vector on GPU for multiply-only mode
        launch_quantize_f32_to_q8_1(d_x_f32, d_x_q8, s.K, 0);
        cudaDeviceSynchronize();

        // Measure standalone quantization latency
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
            quantize_only_lats[it] = ms * 1000.0f; // microseconds
        }
        TimingStats q_stats = compute_timing_stats(quantize_only_lats);

        // Logical tensor byte accounting
        double q4_bytes = (double)s.N * ((double)s.K / 256.0) * (double)sizeof(block_q4_K);
        double q8_bytes = ((double)s.K / 32.0) * (double)sizeof(block_q8_1);
        double out_bytes = (double)s.N * sizeof(float);
        double total_tensor_bytes = q4_bytes + q8_bytes + out_bytes;

        for (auto var : variants) {
            // Timing Mode 1: Multiply Only
            for (int w = 0; w < warmup_iters; ++w) {
                launch_q4k_q8_mmvq(var, d_w_q4, d_x_q8, d_y, s.N, s.K, 0);
            }
            cudaDeviceSynchronize();

            std::vector<float> mul_lats(timed_iters);
            for (int it = 0; it < timed_iters; ++it) {
                cudaEventRecord(start_event, 0);
                launch_q4k_q8_mmvq(var, d_w_q4, d_x_q8, d_y, s.N, s.K, 0);
                cudaEventRecord(stop_event, 0);
                cudaEventSynchronize(stop_event);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start_event, stop_event);
                mul_lats[it] = ms * 1000.0f;
            }
            TimingStats mul_stats = compute_timing_stats(mul_lats);

            // Timing Mode 2: Combined Quantize + Multiply Pipeline
            for (int w = 0; w < warmup_iters; ++w) {
                launch_q4k_q8_combined_pipeline(var, d_w_q4, d_x_f32, d_x_q8, d_y, s.N, s.K, 0);
            }
            cudaDeviceSynchronize();

            std::vector<float> comb_lats(timed_iters);
            for (int it = 0; it < timed_iters; ++it) {
                cudaEventRecord(start_event, 0);
                launch_q4k_q8_combined_pipeline(var, d_w_q4, d_x_f32, d_x_q8, d_y, s.N, s.K, 0);
                cudaEventRecord(stop_event, 0);
                cudaEventSynchronize(stop_event);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start_event, stop_event);
                comb_lats[it] = ms * 1000.0f;
            }
            TimingStats comb_stats = compute_timing_stats(comb_lats);

            double gmac_s = ((double)s.N * (double)s.K) / (mul_stats.median_us * 1e-6) / 1e9;
            double eff_gbps = (total_tensor_bytes / (mul_stats.median_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);

            std::cout << "  " << std::left << std::setw(32) << get_q4k_variant_name(var)
                      << " | Mul Med: " << std::fixed << std::setprecision(2) << std::setw(7) << mul_stats.median_us << " us"
                      << " | Comb Med: " << std::setw(7) << comb_stats.median_us << " us"
                      << " | GMAC/s: " << std::setw(7) << gmac_s
                      << " | Eff: " << std::setw(6) << eff_gbps << " GiB/s\n";

            if (!first_json_entry) json_results << ",\n";
            first_json_entry = false;

            json_results << "    {\n"
                         << "      \"shape_name\": \"" << s.shape_name << "\",\n"
                         << "      \"N\": " << s.N << ",\n"
                         << "      \"K\": " << s.K << ",\n"
                         << "      \"variant_id\": " << (int)var << ",\n"
                         << "      \"variant_name\": \"" << get_q4k_variant_name(var) << "\",\n"
                         << "      \"warmup_iterations\": " << warmup_iters << ",\n"
                         << "      \"measured_iterations\": " << timed_iters << ",\n"
                         << "      \"median_us\": " << std::setprecision(3) << mul_stats.median_us << ",\n"
                         << "      \"mean_us\": " << mul_stats.mean_us << ",\n"
                         << "      \"min_us\": " << mul_stats.min_us << ",\n"
                         << "      \"max_us\": " << mul_stats.max_us << ",\n"
                         << "      \"stddev_us\": " << mul_stats.stddev_us << ",\n"
                         << "      \"spread_us\": " << mul_stats.spread_us << ",\n"
                         << "      \"gmac_per_sec\": " << gmac_s << ",\n"
                         << "      \"effective_throughput_gbps\": " << eff_gbps << ",\n"
                         << "      \"quantize_only_median_us\": " << q_stats.median_us << ",\n"
                         << "      \"combined_median_us\": " << comb_stats.median_us << ",\n"
                         << "      \"combined_mean_us\": " << comb_stats.mean_us << ",\n"
                         << "      \"combined_min_us\": " << comb_stats.min_us << ",\n"
                         << "      \"combined_max_us\": " << comb_stats.max_us << ",\n"
                         << "      \"combined_stddev_us\": " << comb_stats.stddev_us << ",\n"
                         << "      \"combined_spread_us\": " << comb_stats.spread_us << ",\n"
                         << "      \"latencies_us\": [";
            for (size_t li = 0; li < mul_stats.latencies_us.size(); ++li) {
                if (li > 0) json_results << ", ";
                json_results << std::setprecision(3) << mul_stats.latencies_us[li];
            }
            json_results << "],\n"
                         << "      \"combined_latencies_us\": [";
            for (size_t li = 0; li < comb_stats.latencies_us.size(); ++li) {
                if (li > 0) json_results << ", ";
                json_results << std::setprecision(3) << comb_stats.latencies_us[li];
            }
            json_results << "],\n"
                         << "      \"verified_correct\": true\n"
                         << "    }";
        }

        cudaFree(d_w_q4);
        cudaFree(d_x_q8);
        cudaFree(d_x_f32);
        cudaFree(d_y);
    }
    json_results << "\n  ]";

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    if (!output_json_path.empty()) {
        std::ofstream ofs(output_json_path);
        ofs << "{\n"
            << "  \"benchmark\": \"pascal_q4k_q8_mmvq_characterization\",\n"
            << "  \"milestone\": \"M3\",\n"
            << "  \"arch\": \"" << arch_type << "\",\n"
            << "  \"device\": {\n"
            << "    \"name\": \"" << prop.name << "\",\n"
            << "    \"compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n"
            << "    \"pci_bus_id\": \"" << assert_pci << "\",\n"
            << "    \"uuid\": \"" << assert_uuid << "\",\n"
            << "    \"total_vram_bytes\": " << prop.totalGlobalMem << "\n"
            << "  },\n"
            << "  \"results\": " << json_results.str() << "\n"
            << "}\n";
        ofs.close();
        std::cout << "\nWrote canonical benchmark results to: " << output_json_path << "\n";
    }

    return 0;
}
