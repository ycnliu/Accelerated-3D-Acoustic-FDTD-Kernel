#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <sys/time.h>

// Include the original OpenACC function prototype
extern "C" int Kernel(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec,
                     struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                     const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                     const float dt, const float h_x, const float h_y, const float h_z,
                     const float o_x, const float o_y, const float o_z,
                     const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                     const int deviceid, const int devicerm, struct profiler *timers);

// Include the CUDA function prototype
extern "C" int Kernel_CUDA(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec,
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
    double section0{};
    double section1{};
    double cuda_malloc{};
    double cuda_memcpy{};
    double kernel_time{};
    double conversion_time{};  // required by CUDA/OpenACC impls
};

class BenchmarkSuite {
private:
    std::vector<int> grid_sizes = {64, 96, 128, 160, 192};
    int time_steps = 100;
    int num_sources = 10;
    std::mt19937 rng{42};

    void initialize_arrays(dataobj* m_vec, dataobj* src_vec, dataobj* src_coords_vec, dataobj* u_vec, int nx, int ny, int nz) {
        std::uniform_real_distribution<float> dist(0.1f, 2.0f);
        std::uniform_real_distribution<float> coord_dist(-1.0f, 1.0f);

        // Initialize velocity model (m = 1/c^2)
        float* m_data = (float*)m_vec->data;
        for (int i = 0; i < nx * ny * nz; i++) {
            m_data[i] = 1.0f / (dist(rng) * dist(rng)); // Random velocity model
        }

        // Initialize source data
        float* src_data = (float*)src_vec->data;
        for (int t = 0; t < time_steps; t++) {
            for (int s = 0; s < num_sources; s++) {
                // Ricker wavelet source
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
            src_coords_data[s * 3 + 0] = coord_dist(rng) * nx * 0.01f; // x coordinate
            src_coords_data[s * 3 + 1] = coord_dist(rng) * ny * 0.01f; // y coordinate
            src_coords_data[s * 3 + 2] = coord_dist(rng) * nz * 0.01f; // z coordinate
        }

        // Initialize wavefield (u) to zero
        float* u_data = (float*)u_vec->data;
        memset(u_data, 0, 3 * nx * ny * nz * sizeof(float));
    }

public:
    struct BenchmarkResult {
        int grid_size;
        double total_time;
        double kernel_time;
        double memory_time;
        double memory_gb;
        double gflops;
        double throughput_gb_s;
    };

    void run_benchmark() {
        std::cout << "=== FDTD Benchmark Suite ===" << std::endl;
        std::cout << "Grid sizes: ";
        for (int size : grid_sizes) std::cout << size << "³ ";
        std::cout << std::endl;
        std::cout << "Time steps: " << time_steps << std::endl;
        std::cout << "Number of sources: " << num_sources << std::endl << std::endl;

        std::vector<BenchmarkResult> cuda_results;

        for (int grid_size : grid_sizes) {
            std::cout << "Testing grid size: " << grid_size << "³" << std::endl;

            // Test CUDA version
            auto cuda_result = benchmark_cuda(grid_size);
            cuda_results.push_back(cuda_result);

            print_result("CUDA", cuda_result);
            std::cout << std::endl;
        }

        // Save results to file
        save_results("cuda_benchmark_results.csv", cuda_results);

        // Print summary
        print_summary(cuda_results);
    }

private:
    BenchmarkResult benchmark_cuda(int grid_size) {
        // Setup dimensions
        int nx = grid_size + 8; // Add halo
        int ny = grid_size + 8;
        int nz = grid_size + 8;

        // Allocate and initialize data structures
        dataobj m_vec, src_vec, src_coords_vec, u_vec;

        // Setup m_vec (velocity model)
        m_vec.data = new float[nx * ny * nz];
        int m_size[] = {nx, ny, nz};
        m_vec.size = m_size;

        // Setup src_vec (source data)
        src_vec.data = new float[time_steps * num_sources];
        int src_size[] = {time_steps, num_sources};
        src_vec.size = src_size;

        // Setup src_coords_vec (source coordinates)
        src_coords_vec.data = new float[num_sources * 3];
        int src_coords_size[] = {num_sources, 3};
        src_coords_vec.size = src_coords_size;

        // Setup u_vec (wavefield)
        u_vec.data = new float[3 * nx * ny * nz];
        int u_size[] = {3, nx, ny, nz};
        u_vec.size = u_size;

        initialize_arrays(&m_vec, &src_vec, &src_coords_vec, &u_vec, nx, ny, nz);

        // Setup profiler
        profiler timers = {0};

        // Parameters
        float dt = 0.001f;
        float h_x = 0.01f, h_y = 0.01f, h_z = 0.01f;
        float o_x = 0.0f, o_y = 0.0f, o_z = 0.0f;

        // Run benchmark
        auto start = std::chrono::high_resolution_clock::now();

        int ret = Kernel_CUDA(&m_vec, &src_vec, &src_coords_vec, &u_vec,
                             grid_size-1, 0, grid_size-1, 0, grid_size-1, 0,
                             dt, h_x, h_y, h_z, o_x, o_y, o_z,
                             num_sources-1, 0, time_steps-1, 0, 0, 1, &timers);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Calculate metrics
        BenchmarkResult result;
        result.grid_size = grid_size;
        result.total_time = duration.count() / 1e6; // Convert to seconds
        result.kernel_time = timers.kernel_time;
        result.memory_time = timers.cuda_malloc + timers.cuda_memcpy;

        // Memory usage (GB)
        size_t total_memory = (3 * nx * ny * nz + nx * ny * nz + time_steps * num_sources + num_sources * 3) * sizeof(float);
        result.memory_gb = total_memory / (1024.0 * 1024.0 * 1024.0);

        // GFLOPS calculation (approximate)
        long long ops_per_point = 37; // Approximate FLOPs for 4th order stencil
        long long total_ops = (long long)grid_size * grid_size * grid_size * time_steps * ops_per_point;
        result.gflops = (total_ops / 1e9) / result.kernel_time;

        // Throughput (GB/s)
        size_t bytes_per_timestep = grid_size * grid_size * grid_size * sizeof(float) * 8; // Read/write multiple arrays
        result.throughput_gb_s = (bytes_per_timestep * time_steps / (1024.0 * 1024.0 * 1024.0)) / result.kernel_time;

        // Cleanup
        delete[] (float*)m_vec.data;
        delete[] (float*)src_vec.data;
        delete[] (float*)src_coords_vec.data;
        delete[] (float*)u_vec.data;

        return result;
    }

    void print_result(const std::string& name, const BenchmarkResult& result) {
        std::cout << name << " Results:" << std::endl;
        std::cout << "  Total Time: " << std::fixed << std::setprecision(3) << result.total_time << " s" << std::endl;
        std::cout << "  Kernel Time: " << std::fixed << std::setprecision(3) << result.kernel_time << " s" << std::endl;
        std::cout << "  Memory Time: " << std::fixed << std::setprecision(3) << result.memory_time << " s" << std::endl;
        std::cout << "  Memory Usage: " << std::fixed << std::setprecision(2) << result.memory_gb << " GB" << std::endl;
        std::cout << "  Performance: " << std::fixed << std::setprecision(1) << result.gflops << " GFLOPS" << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << result.throughput_gb_s << " GB/s" << std::endl;
    }

    void save_results(const std::string& filename, const std::vector<BenchmarkResult>& results) {
        std::ofstream file(filename);
        file << "grid_size,total_time,kernel_time,memory_time,memory_gb,gflops,throughput_gb_s\n";

        for (const auto& result : results) {
            file << result.grid_size << ","
                 << result.total_time << ","
                 << result.kernel_time << ","
                 << result.memory_time << ","
                 << result.memory_gb << ","
                 << result.gflops << ","
                 << result.throughput_gb_s << "\n";
        }
        file.close();
        std::cout << "Results saved to " << filename << std::endl;
    }

    void print_summary(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n=== Performance Summary ===" << std::endl;
        std::cout << std::left << std::setw(12) << "Grid Size"
                  << std::setw(12) << "Time (s)"
                  << std::setw(12) << "GFLOPS"
                  << std::setw(15) << "Throughput" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        for (const auto& result : results) {
            std::cout << std::left << std::setw(12) << (std::to_string(result.grid_size) + "³")
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.kernel_time
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.gflops
                      << std::setw(15) << std::fixed << std::setprecision(1) << (std::to_string((int)result.throughput_gb_s) + " GB/s") << std::endl;
        }
    }
};

int main() {
    BenchmarkSuite benchmark;
    benchmark.run_benchmark();
    return 0;
}