#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>

// Function prototypes for all implementations
extern "C" int Kernel_CUDA(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec,
                          struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                          const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                          const float dt, const float h_x, const float h_y, const float h_z,
                          const float o_x, const float o_y, const float o_z,
                          const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                          const int deviceid, const int devicerm, struct profiler *timers);

extern "C" int Kernel_Mixed_Precision(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec,
                                     struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                                     const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                                     const float dt, const float h_x, const float h_y, const float h_z,
                                     const float o_x, const float o_y, const float o_z,
                                     const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                                     const int deviceid, const int devicerm, struct profiler *timers, int use_vectorized);

extern "C" int Kernel_Optimized(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec,
                                struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                                const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                                const float dt, const float h_x, const float h_y, const float h_z,
                                const float o_x, const float o_y, const float o_z,
                                const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                                const int deviceid, const int devicerm, struct profiler *timers, int use_half2);

extern "C" int Kernel_OpenACC(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec,
                              struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                              const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                              const float dt, const float h_x, const float h_y, const float h_z,
                              const float o_x, const float o_y, const float o_z,
                              const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                              const int deviceid, const int devicerm, struct profiler *timers);

extern "C" int Kernel_Temporal_Blocking(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec,
                                        struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                                        const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                                        const float dt, const float h_x, const float h_y, const float h_z,
                                        const float o_x, const float o_y, const float o_z,
                                        const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                                        const int deviceid, const int devicerm, struct profiler *timers);

struct dataobj {
    void *__restrict data;
    int *size;
    unsigned long nbytes;
    unsigned long *npsize;
    unsigned long *dsize;
    int *hsize;
    int *hofs;
    int *oofs;
    void *dmap;
};

struct profiler {
    double section0;
    double section1;
    double cuda_malloc;
    double cuda_memcpy;
    double kernel_time;
    double conversion_time;
};

class ComprehensiveBenchmark {
private:
    std::vector<int> grid_sizes = {64, 96, 128, 160, 192};
    int time_steps = 100;
    int num_sources = 10;
    int num_runs = 3; // Multiple runs for statistical reliability
    std::mt19937 rng{42};

    struct BenchmarkResult {
        std::string implementation;
        int grid_size;
        double total_time;
        double kernel_time;
        double memory_time;
        double conversion_time;
        double memory_gb;
        double gflops;
        double throughput_gb_s;
        double speedup_vs_baseline;
        int grid_points;
    };

    struct ValidationResult {
        double relL2_error{};
        double relLinf_error{};
        double max_abs_error{};
        bool passes_threshold{};
    };

    ValidationResult compute_validation_error(const float* candidate, const float* reference, int nx, int ny, int nz) {
        ValidationResult result;

        double relL2 = 0, relL2den = 0, relInf = 0, refInf = 0;
        int halo = 4;

        for (int i = halo; i < nx - halo; i++) {
            for (int j = halo; j < ny - halo; j++) {
                for (int k = halo; k < nz - halo; k++) {
                    size_t idx = ((size_t)i * ny + j) * nz + k;
                    double d = double(candidate[idx]) - double(reference[idx]);
                    relL2 += d * d;
                    relL2den += double(reference[idx]) * double(reference[idx]);
                    relInf = std::max(relInf, std::abs(d));
                    refInf = std::max(refInf, std::abs(double(reference[idx])));
                    result.max_abs_error = std::max(result.max_abs_error, std::abs(d));
                }
            }
        }

        result.relL2_error = std::sqrt(relL2) / (std::sqrt(relL2den) + 1e-30);
        result.relLinf_error = relInf / (refInf + 1e-30);

        // Validation thresholds
        bool fp32_precision = result.relL2_error <= 1e-6;
        bool fp16_precision = result.relL2_error <= 1e-3; // FP16 storage vs FP32 reference
        result.passes_threshold = fp32_precision || fp16_precision;

        return result;
    }

    void initialize_arrays(dataobj* m_vec, dataobj* src_vec, dataobj* src_coords_vec, dataobj* u_vec, int nx, int ny, int nz) {
        std::uniform_real_distribution<float> dist(0.5f, 2.0f);
        std::uniform_real_distribution<float> coord_dist(-1.0f, 1.0f);

        // Initialize velocity model (m = 1/c^2)
        float* m_data = (float*)m_vec->data;
        for (int i = 0; i < nx * ny * nz; i++) {
            m_data[i] = 1.0f / (dist(rng) * dist(rng));
        }

        // Initialize source data with Ricker wavelet
        float* src_data = (float*)src_vec->data;
        for (int t = 0; t < time_steps; t++) {
            for (int s = 0; s < num_sources; s++) {
                float t_shift = 0.1f;
                float freq = 10.0f;
                float time_val = (t - time_steps/4) * 0.001f - t_shift;
                float pi_freq_t = M_PI * freq * time_val;
                src_data[t * num_sources + s] = (1.0f - 2.0f * pi_freq_t * pi_freq_t) *
                                               expf(-pi_freq_t * pi_freq_t);
            }
        }

        // Initialize source coordinates
        float* src_coords_data = (float*)src_coords_vec->data;
        for (int s = 0; s < num_sources; s++) {
            src_coords_data[s * 3 + 0] = coord_dist(rng) * nx * 0.01f;
            src_coords_data[s * 3 + 1] = coord_dist(rng) * ny * 0.01f;
            src_coords_data[s * 3 + 2] = coord_dist(rng) * nz * 0.01f;
        }

        // Initialize wavefield to zero
        float* u_data = (float*)u_vec->data;
        memset(u_data, 0, 3 * nx * ny * nz * sizeof(float));
    }

    void copy_arrays(const dataobj* src_m, const dataobj* src_src, const dataobj* src_coords, const dataobj* src_u,
                    dataobj* dst_m, dataobj* dst_src, dataobj* dst_coords, dataobj* dst_u) {
        // Deep copy all arrays to ensure identical initial conditions
        memcpy(dst_m->data, src_m->data, src_m->size[0] * src_m->size[1] * src_m->size[2] * sizeof(float));
        memcpy(dst_src->data, src_src->data, src_src->size[0] * src_src->size[1] * sizeof(float));
        memcpy(dst_coords->data, src_coords->data, src_coords->size[0] * src_coords->size[1] * sizeof(float));
        memcpy(dst_u->data, src_u->data, 3 * src_u->size[1] * src_u->size[2] * src_u->size[3] * sizeof(float));
    }

    BenchmarkResult benchmark_implementation(const std::string& impl_name, int grid_size,
                                           dataobj* m_vec, dataobj* src_vec, dataobj* src_coords_vec, dataobj* u_vec) {
        int nx = grid_size + 8;
        int ny = grid_size + 8;
        int nz = grid_size + 8;

        profiler timers{}; // value-init to zeros
        float dt = 0.001f;
        float h_x = 0.01f, h_y = 0.01f, h_z = 0.01f;
        float o_x = 0.0f, o_y = 0.0f, o_z = 0.0f;

        std::vector<double> run_times;

        for (int run = 0; run < num_runs; run++) {
            // Reset timers
            timers = {0};

            auto start = std::chrono::high_resolution_clock::now();

            int ret = 0;
            if (impl_name == "OpenACC_Original") {
                ret = Kernel_OpenACC(m_vec, src_vec, src_coords_vec, u_vec,
                                    grid_size-1, 0, grid_size-1, 0, grid_size-1, 0,
                                    dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                    num_sources-1, 0, time_steps-1, 0, 0, 1, &timers);
            } else if (impl_name == "CUDA_Baseline") {
                ret = Kernel_CUDA(m_vec, src_vec, src_coords_vec, u_vec,
                                 grid_size-1, 0, grid_size-1, 0, grid_size-1, 0,
                                 dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                 num_sources-1, 0, time_steps-1, 0, 0, 1, &timers);
            } else if (impl_name == "Mixed_Precision_Shared") {
                ret = Kernel_Mixed_Precision(m_vec, src_vec, src_coords_vec, u_vec,
                                           grid_size-1, 0, grid_size-1, 0, grid_size-1, 0,
                                           dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                           num_sources-1, 0, time_steps-1, 0, 0, 1, &timers, 0);
            } else if (impl_name == "Mixed_Precision_Half2") {
                ret = Kernel_Mixed_Precision(m_vec, src_vec, src_coords_vec, u_vec,
                                           grid_size-1, 0, grid_size-1, 0, grid_size-1, 0,
                                           dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                           num_sources-1, 0, time_steps-1, 0, 0, 1, &timers, 1);
            } else if (impl_name == "Optimized_Convert_Once") {
                ret = Kernel_Optimized(m_vec, src_vec, src_coords_vec, u_vec,
                                     grid_size-1, 0, grid_size-1, 0, grid_size-1, 0,
                                     dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                     num_sources-1, 0, time_steps-1, 0, 0, 1, &timers, 0);
            } else if (impl_name == "Optimized_Half2") {
                ret = Kernel_Optimized(m_vec, src_vec, src_coords_vec, u_vec,
                                     grid_size-1, 0, grid_size-1, 0, grid_size-1, 0,
                                     dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                     num_sources-1, 0, time_steps-1, 0, 0, 1, &timers, 1);
            } else if (impl_name == "Temporal_Blocking") {
                ret = Kernel_Temporal_Blocking(m_vec, src_vec, src_coords_vec, u_vec,
                                             grid_size-1, 0, grid_size-1, 0, grid_size-1, 0,
                                             dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                             num_sources-1, 0, time_steps-1, 0, 0, 1, &timers);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            run_times.push_back(duration.count() / 1e6);

            if (ret != 0) {
                std::cerr << "Error in " << impl_name << " kernel" << std::endl;
                break;
            }
        }

        // Use minimum time for performance measurement (most reliable)
        double total_time = *std::min_element(run_times.begin(), run_times.end());

        BenchmarkResult result;
        result.implementation = impl_name;
        result.grid_size = grid_size;
        result.total_time = total_time;
        result.kernel_time = timers.kernel_time;
        result.memory_time = timers.cuda_malloc + timers.cuda_memcpy;
        result.conversion_time = timers.conversion_time;
        result.grid_points = grid_size * grid_size * grid_size;

        // Memory usage calculation
        size_t u_bytes = 3 * nx * ny * nz * sizeof(float);
        size_t m_bytes = nx * ny * nz * sizeof(float);
        size_t src_bytes = time_steps * num_sources * sizeof(float);
        size_t coords_bytes = num_sources * 3 * sizeof(float);

        if (impl_name.find("Mixed_Precision") != std::string::npos) {
            u_bytes = 3 * nx * ny * nz * sizeof(__half); // Half precision for wavefield
        }

        result.memory_gb = (u_bytes + m_bytes + src_bytes + coords_bytes) / (1024.0 * 1024.0 * 1024.0);

        // GFLOPS calculation (37 FLOPs per grid point for 4th order stencil)
        long long ops_per_point = 37;
        long long total_ops = (long long)result.grid_points * time_steps * ops_per_point;
        result.gflops = (total_ops / 1e9) / result.kernel_time;

        // Throughput calculation (model):
        // FP32 u: ~ (2 reads + 1 write)*4 + prev read*4 + m read*4 ≈ 20B per cell (conservative model)
        const double bytes_per_cell = 20.0; // Conservative estimate with cache effects
        double bytes_total = bytes_per_cell * result.grid_points * time_steps;
        result.throughput_gb_s = (bytes_total / (1024.0 * 1024.0 * 1024.0)) / result.kernel_time;

        return result;
    }

    void verify_accuracy(const dataobj* baseline, const dataobj* test, const std::string& test_name) {
        float* baseline_data = (float*)baseline->data;
        float* test_data = (float*)test->data;

        int nx = baseline->size[1];
        int ny = baseline->size[2];
        int nz = baseline->size[3];
        int total_points = 3 * nx * ny * nz;

        // Use improved validation with proper halo handling
        ValidationResult validation = compute_validation_error(test_data, baseline_data, nx, ny, nz);

        std::cout << "\nAccuracy verification for " << test_name << ":" << std::endl;
        std::cout << "  RMS Error: " << std::scientific << std::setprecision(2) << validation.relL2_error << std::endl;
        std::cout << "  Max Error: " << std::scientific << std::setprecision(2) << validation.max_abs_error << std::endl;
        std::cout << "  Relative Error: " << std::scientific << std::setprecision(2) << validation.relLinf_error
                  << " (" << std::fixed << std::setprecision(4) << validation.relLinf_error * 100 << "%)" << std::endl;
    }

public:
    void run_comprehensive_benchmark() {
        std::cout << "=== Comprehensive FDTD Benchmark Suite ===" << std::endl;
        std::cout << "Testing implementations: CUDA Baseline, Mixed Precision (Shared), Mixed Precision (Half2), Temporal Blocking" << std::endl;
        std::cout << "Grid sizes: ";
        for (int size : grid_sizes) std::cout << size << "³ ";
        std::cout << std::endl;
        std::cout << "Time steps: " << time_steps << ", Runs per test: " << num_runs << std::endl << std::endl;

        std::vector<BenchmarkResult> all_results;
        std::vector<std::string> implementations = {"OpenACC_Original", "CUDA_Baseline", "Mixed_Precision_Shared", "Optimized_Convert_Once", "Temporal_Blocking"};

        for (int grid_size : grid_sizes) {
            std::cout << "Testing grid size: " << grid_size << "³" << std::endl;

            // Setup data structures
            int nx = grid_size + 8;
            int ny = grid_size + 8;
            int nz = grid_size + 8;

            // Master arrays (baseline)
            dataobj m_master, src_master, src_coords_master, u_master;

            m_master.data = new float[nx * ny * nz];
            int m_size[] = {nx, ny, nz};
            m_master.size = m_size;

            src_master.data = new float[time_steps * num_sources];
            int src_size[] = {time_steps, num_sources};
            src_master.size = src_size;

            src_coords_master.data = new float[num_sources * 3];
            int src_coords_size[] = {num_sources, 3};
            src_coords_master.size = src_coords_size;

            u_master.data = new float[3 * nx * ny * nz];
            int u_size[] = {3, nx, ny, nz};
            u_master.size = u_size;

            initialize_arrays(&m_master, &src_master, &src_coords_master, &u_master, nx, ny, nz);

            // Store baseline result for accuracy comparison
            BenchmarkResult baseline_result;
            dataobj u_baseline = u_master; // Copy for baseline
            u_baseline.data = new float[3 * nx * ny * nz];
            memcpy(u_baseline.data, u_master.data, 3 * nx * ny * nz * sizeof(float));

            for (const auto& impl : implementations) {
                // Create working copies
                dataobj m_work = m_master;
                dataobj src_work = src_master;
                dataobj src_coords_work = src_coords_master;
                dataobj u_work = u_master;

                m_work.data = new float[nx * ny * nz];
                src_work.data = new float[time_steps * num_sources];
                src_coords_work.data = new float[num_sources * 3];
                u_work.data = new float[3 * nx * ny * nz];

                copy_arrays(&m_master, &src_master, &src_coords_master, &u_master,
                           &m_work, &src_work, &src_coords_work, &u_work);

                auto result = benchmark_implementation(impl, grid_size, &m_work, &src_work, &src_coords_work, &u_work);

                if (impl == "OpenACC_Original") {
                    baseline_result = result;
                    result.speedup_vs_baseline = 1.0;
                    // Save baseline result for accuracy comparison
                    memcpy(u_baseline.data, u_work.data, 3 * nx * ny * nz * sizeof(float));
                } else {
                    result.speedup_vs_baseline = baseline_result.kernel_time / result.kernel_time;
                    // Verify accuracy
                    verify_accuracy(&u_baseline, &u_work, impl);
                }

                all_results.push_back(result);
                print_result(result);

                // Cleanup
                delete[] (float*)m_work.data;
                delete[] (float*)src_work.data;
                delete[] (float*)src_coords_work.data;
                delete[] (float*)u_work.data;
            }

            // Cleanup master arrays
            delete[] (float*)m_master.data;
            delete[] (float*)src_master.data;
            delete[] (float*)src_coords_master.data;
            delete[] (float*)u_master.data;
            delete[] (float*)u_baseline.data;

            std::cout << std::endl;
        }

        // Save and summarize results
        save_results("comprehensive_benchmark_results.csv", all_results);
        print_summary(all_results);
        generate_analysis(all_results);
    }

private:
    void print_result(const BenchmarkResult& result) {
        std::cout << result.implementation << " Results:" << std::endl;
        std::cout << "  Kernel Time: " << std::fixed << std::setprecision(3) << result.kernel_time << " s" << std::endl;
        std::cout << "  Total Time: " << std::fixed << std::setprecision(3) << result.total_time << " s" << std::endl;
        std::cout << "  Memory Time: " << std::fixed << std::setprecision(3) << result.memory_time << " s" << std::endl;
        if (result.conversion_time > 0) {
            std::cout << "  Conversion Time: " << std::fixed << std::setprecision(3) << result.conversion_time << " s" << std::endl;
        }
        std::cout << "  Memory Usage: " << std::fixed << std::setprecision(3) << result.memory_gb << " GB" << std::endl;
        std::cout << "  Performance: " << std::fixed << std::setprecision(1) << result.gflops << " GFLOPS" << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << result.throughput_gb_s << " GB/s" << std::endl;
        if (result.speedup_vs_baseline != 1.0) {
            std::cout << "  Speedup vs Baseline: " << std::fixed << std::setprecision(2) << result.speedup_vs_baseline << "×" << std::endl;
        }
    }

    void save_results(const std::string& filename, const std::vector<BenchmarkResult>& results) {
        std::ofstream file(filename);
        file << "implementation,grid_size,total_time,kernel_time,memory_time,conversion_time,memory_gb,gflops,throughput_gb_s,speedup\n";

        for (const auto& result : results) {
            file << result.implementation << ","
                 << result.grid_size << ","
                 << result.total_time << ","
                 << result.kernel_time << ","
                 << result.memory_time << ","
                 << result.conversion_time << ","
                 << result.memory_gb << ","
                 << result.gflops << ","
                 << result.throughput_gb_s << ","
                 << result.speedup_vs_baseline << "\n";
        }
        file.close();
        std::cout << "Detailed results saved to " << filename << std::endl;
    }

    void print_summary(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n=== Performance Summary ===" << std::endl;
        std::cout << std::left << std::setw(25) << "Implementation"
                  << std::setw(12) << "Grid Size"
                  << std::setw(12) << "Time (ms)"
                  << std::setw(12) << "GFLOPS"
                  << std::setw(15) << "Throughput"
                  << std::setw(10) << "Speedup" << std::endl;
        std::cout << std::string(90, '-') << std::endl;

        for (const auto& result : results) {
            std::cout << std::left << std::setw(25) << result.implementation
                      << std::setw(12) << (std::to_string(result.grid_size) + "³")
                      << std::setw(12) << std::fixed << std::setprecision(1) << (result.kernel_time * 1000)
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.gflops
                      << std::setw(15) << std::fixed << std::setprecision(1) << (std::to_string((int)result.throughput_gb_s) + " GB/s")
                      << std::setw(10) << std::fixed << std::setprecision(2) << result.speedup_vs_baseline << "×" << std::endl;
        }
    }

    void generate_analysis(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n=== Performance Analysis ===" << std::endl;

        // Find best speedups
        double max_speedup = 0;
        std::string best_impl;
        int best_size = 0;

        for (const auto& result : results) {
            if (result.speedup_vs_baseline > max_speedup) {
                max_speedup = result.speedup_vs_baseline;
                best_impl = result.implementation;
                best_size = result.grid_size;
            }
        }

        std::cout << "Best speedup: " << std::fixed << std::setprecision(2) << max_speedup
                  << "× achieved by " << best_impl << " on " << best_size << "³ grid" << std::endl;

        // Memory bandwidth analysis
        std::cout << "\nMemory bandwidth comparison (largest grid):" << std::endl;
        for (const auto& impl : {"CUDA_Baseline", "Mixed_Precision_Shared", "Mixed_Precision_Half2"}) {
            auto it = std::find_if(results.rbegin(), results.rend(),
                                  [&](const BenchmarkResult& r) {
                                      return r.implementation == impl && r.grid_size == grid_sizes.back();
                                  });
            if (it != results.rend()) {
                std::cout << "  " << std::left << std::setw(25) << impl
                          << ": " << std::fixed << std::setprecision(1) << it->throughput_gb_s << " GB/s" << std::endl;
            }
        }

        // Scaling analysis
        std::cout << "\nScaling efficiency (kernel time scaling with grid size):" << std::endl;
        for (const auto& impl : {"CUDA_Baseline", "Mixed_Precision_Shared", "Mixed_Precision_Half2"}) {
            std::vector<double> times;
            std::vector<int> sizes;

            for (const auto& result : results) {
                if (result.implementation == impl) {
                    times.push_back(result.kernel_time);
                    sizes.push_back(result.grid_points);
                }
            }

            if (times.size() >= 2) {
                double scaling_factor = times.back() / times.front();
                double size_factor = (double)sizes.back() / sizes.front();
                double efficiency = log(scaling_factor) / log(size_factor);

                std::cout << "  " << std::left << std::setw(25) << impl
                          << ": " << std::fixed << std::setprecision(2) << efficiency
                          << " (ideal = 1.0)" << std::endl;
            }
        }
    }
};

int main() {
    ComprehensiveBenchmark benchmark;
    benchmark.run_comprehensive_benchmark();
    return 0;
}