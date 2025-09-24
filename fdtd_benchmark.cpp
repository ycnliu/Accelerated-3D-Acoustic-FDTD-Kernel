#define _POSIX_C_SOURCE 200809L
#include <sys/time.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>

// Shared ABI structs
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
  double cuda_memcpy_h2d{};
  double cuda_memcpy_d2h{};
  double kernel_time{};
  double conversion_time{};
};

// External kernel entry points
extern "C" int Kernel_CUDA(
  dataobj* __restrict m_vec,
  dataobj* __restrict src_vec,
  dataobj* __restrict src_coords_vec,
  dataobj* __restrict u_vec,
  const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
  const float dt, const float h_x, const float h_y, const float h_z,
  const float o_x, const float o_y, const float o_z,
  const int p_src_M, const int p_src_m, const int time_M, const int time_m,
  const int deviceid, const int devicerm, profiler *timers);

extern "C" int Kernel_Mixed_Precision(
  dataobj* __restrict m_vec,
  dataobj* __restrict src_vec,
  dataobj* __restrict src_coords_vec,
  dataobj* __restrict u_vec,
  const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
  const float dt, const float h_x, const float h_y, const float h_z,
  const float o_x, const float o_y, const float o_z,
  const int p_src_M, const int p_src_m, const int time_M, const int time_m,
  const int deviceid, const int devicerm, profiler *timers);

extern "C" int Kernel_Temporal_Blocking(
  dataobj* __restrict m_vec,
  dataobj* __restrict src_vec,
  dataobj* __restrict src_coords_vec,
  dataobj* __restrict u_vec,
  const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
  const float dt, const float h_x, const float h_y, const float h_z,
  const float o_x, const float o_y, const float o_z,
  const int p_src_M, const int p_src_m, const int time_M, const int time_m,
  const int deviceid, const int devicerm, profiler *timers);

extern "C" int Kernel_Optimized_Advanced(
  dataobj* __restrict m_vec,
  dataobj* __restrict src_vec,
  dataobj* __restrict src_coords_vec,
  dataobj* __restrict u_vec,
  const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
  const float dt, const float h_x, const float h_y, const float h_z,
  const float o_x, const float o_y, const float o_z,
  const int p_src_M, const int p_src_m, const int time_M, const int time_m,
  const int deviceid, const int devicerm, profiler *timers);

#ifdef USE_OPENACC
extern "C" int Kernel_OpenACC(
  dataobj* __restrict m_vec,
  dataobj* __restrict src_vec,
  dataobj* __restrict src_coords_vec,
  dataobj* __restrict u_vec,
  const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
  const float dt, const float h_x, const float h_y, const float h_z,
  const float o_x, const float o_y, const float o_z,
  const int p_src_M, const int p_src_m, const int time_M, const int time_m,
  const int deviceid, const int devicerm, profiler *timers);
#endif

#define START(t) do { struct timeval tv1; gettimeofday(&tv1, NULL); timers->t = -(double)(tv1.tv_sec) - (double)(tv1.tv_usec)/1e6; } while (0)
#define STOP(t, timers) do { struct timeval tv2; gettimeofday(&tv2, NULL); timers->t += (double)(tv2.tv_sec) + (double)(tv2.tv_usec)/1e6; } while (0)

struct TestResult {
  std::string method;
  int grid_size;
  int time_steps;
  int sources;
  int gpu_id;
  bool success;
  double total_time_s;
  double kernel_time_s;
  double section0_time_s;
  double section1_time_s;
  double malloc_time_s;
  double h2d_time_s;
  double d2h_time_s;
  double gflops;
  double memory_bandwidth_gb_s;
  std::string error_msg;
};

dataobj* allocate_dataobj(int ndim, int* shape, size_t element_size) {
    dataobj* d = new dataobj();
    d->size = new int[ndim];
    memcpy(d->size, shape, ndim * sizeof(int));

    size_t total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    d->nbytes = total_elements * element_size;
    d->data = malloc(d->nbytes);
    memset(d->data, 0, d->nbytes);

    return d;
}

void deallocate_dataobj(dataobj* d) {
    if (d) {
        free(d->data);
        delete[] d->size;
        delete d;
    }
}

TestResult benchmark_method(const std::string& method_name,
                          int (*kernel_func)(dataobj*, dataobj*, dataobj*, dataobj*,
                                           int, int, int, int, int, int,
                                           float, float, float, float, float, float, float,
                                           int, int, int, int, int, int, profiler*),
                          int grid_size, int time_steps, int sources, int gpu_id) {

    TestResult result;
    result.method = method_name;
    result.grid_size = grid_size;
    result.time_steps = time_steps;
    result.sources = sources;
    result.gpu_id = gpu_id;
    result.success = false;

    try {
        // Domain parameters
        const int x_m = 0, x_M = grid_size - 1;
        const int y_m = 0, y_M = grid_size - 1;
        const int z_m = 0, z_M = grid_size - 1;
        const int time_m = 1, time_M = time_steps;
        const int p_src_m = 0, p_src_M = sources - 1;

        const float dt = 1e-3f;
        const float h_x = 1.0f, h_y = 1.0f, h_z = 1.0f;
        const float o_x = 0.0f, o_y = 0.0f, o_z = 0.0f;

        // Allocate arrays with halo (+8 for boundary)
        int u_shape[4] = {3, grid_size + 8, grid_size + 8, grid_size + 8};
        int m_shape[3] = {grid_size + 8, grid_size + 8, grid_size + 8};
        int src_shape[2] = {time_steps + 1, sources};
        int coords_shape[2] = {sources, 3};

        dataobj* u_vec = allocate_dataobj(4, u_shape, sizeof(float));
        dataobj* m_vec = allocate_dataobj(3, m_shape, sizeof(float));
        dataobj* src_vec = allocate_dataobj(2, src_shape, sizeof(float));
        dataobj* coords_vec = allocate_dataobj(2, coords_shape, sizeof(float));

        // Initialize data
        float* u = (float*)u_vec->data;
        float* m = (float*)m_vec->data;
        float* src = (float*)src_vec->data;
        float* coords = (float*)coords_vec->data;

        const int sz = grid_size + 8;
        const int sy = sz * sz;

        // Initialize with deterministic patterns
        #pragma omp parallel for
        for (int i = 0; i < 3 * sy * sz; i++) {
            u[i] = 0.1f * sin(i * 0.01f);
        }

        #pragma omp parallel for
        for (int i = 0; i < sy * sz; i++) {
            m[i] = 1.0f + 0.1f * cos(i * 0.02f);
        }

        for (int i = 0; i < (time_steps + 1) * sources; i++) {
            src[i] = sin(i * 0.1f);
        }

        for (int s = 0; s < sources; s++) {
            coords[s*3 + 0] = grid_size/4.0f;
            coords[s*3 + 1] = grid_size/4.0f;
            coords[s*3 + 2] = grid_size/4.0f;
        }

        // Run benchmark
        profiler timers = {};
        struct timeval start, end;
        gettimeofday(&start, NULL);

        int ret = kernel_func(m_vec, src_vec, coords_vec, u_vec,
                             x_M, x_m, y_M, y_m, z_M, z_m,
                             dt, h_x, h_y, h_z, o_x, o_y, o_z,
                             p_src_M, p_src_m, time_M, time_m,
                             gpu_id, 0, &timers);

        gettimeofday(&end, NULL);
        double total_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

        // Calculate performance metrics
        const long long grid_points = (long long)(x_M - x_m + 1) * (y_M - y_m + 1) * (z_M - z_m + 1);
        const long long flops_per_point = 37; // 4th-order stencil operations
        const long long total_flops = grid_points * flops_per_point * time_steps;

        double kernel_time = timers.section0 + timers.section1;
        if (kernel_time <= 0.0 && timers.kernel_time > 0.0) {
            kernel_time = timers.kernel_time;
        }

        const double gflops = (kernel_time > 0.0) ? (total_flops / 1e9) / kernel_time : 0.0;

        const long long bytes_per_point = 8 * sizeof(float); // reads + writes
        const long long total_bytes = grid_points * bytes_per_point * time_steps;
        const double bandwidth = (kernel_time > 0.0) ? (total_bytes / 1e9) / kernel_time : 0.0;

        // Fill result
        result.success = (ret == 0);
        result.total_time_s = total_time;
        result.kernel_time_s = kernel_time;
        result.section0_time_s = timers.section0;
        result.section1_time_s = timers.section1;
        result.malloc_time_s = timers.cuda_malloc;
        result.h2d_time_s = timers.cuda_memcpy_h2d;
        result.d2h_time_s = timers.cuda_memcpy_d2h;
        result.gflops = gflops;
        result.memory_bandwidth_gb_s = bandwidth;

        // Cleanup
        deallocate_dataobj(u_vec);
        deallocate_dataobj(m_vec);
        deallocate_dataobj(src_vec);
        deallocate_dataobj(coords_vec);

    } catch (const std::exception& e) {
        result.error_msg = e.what();
    }

    return result;
}

void save_results_csv(const std::vector<TestResult>& results, const std::string& filename) {
    std::ofstream file(filename);

    // Header
    file << "method,grid_size,time_steps,sources,gpu_id,success,total_time_s,kernel_time_s,"
         << "section0_time_s,section1_time_s,malloc_time_s,h2d_time_s,d2h_time_s,"
         << "gflops,memory_bandwidth_gb_s,error_msg\n";

    // Data
    for (const auto& r : results) {
        file << r.method << "," << r.grid_size << "," << r.time_steps << "," << r.sources << ","
             << r.gpu_id << "," << (r.success ? "TRUE" : "FALSE") << ","
             << std::fixed << std::setprecision(6) << r.total_time_s << ","
             << r.kernel_time_s << "," << r.section0_time_s << "," << r.section1_time_s << ","
             << r.malloc_time_s << "," << r.h2d_time_s << "," << r.d2h_time_s << ","
             << std::setprecision(3) << r.gflops << "," << r.memory_bandwidth_gb_s << ","
             << "\"" << r.error_msg << "\"\n";
    }
}

void print_summary(const std::vector<TestResult>& results) {
    std::cout << "\n=== BENCHMARK SUMMARY ===\n";
    std::cout << std::left << std::setw(20) << "Method"
              << std::setw(10) << "Grid"
              << std::setw(8) << "GPU"
              << std::setw(10) << "GFLOPS"
              << std::setw(12) << "Bandwidth"
              << std::setw(8) << "Status\n";
    std::cout << std::string(68, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(20) << r.method
                  << std::setw(10) << r.grid_size
                  << std::setw(8) << r.gpu_id
                  << std::setw(10) << std::fixed << std::setprecision(1) << r.gflops
                  << std::setw(12) << std::setprecision(1) << r.memory_bandwidth_gb_s
                  << (r.success ? "PASS" : "FAIL") << "\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "=== FDTD MULTI-GPU BENCHMARK SUITE ===\n\n";

    // Check available GPUs
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    std::cout << "Available GPUs: " << gpu_count << "\n";

    if (gpu_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    // Use up to 4 GPUs
    int max_gpus = std::min(4, gpu_count);
    std::cout << "Using " << max_gpus << " GPUs\n\n";

    // Test configurations: larger grid sizes
    std::vector<int> grid_sizes = {64, 96, 128, 160, 192};
    std::vector<int> time_steps_list = {10, 25, 50};
    int sources = 1;

    std::vector<TestResult> all_results;

    // Benchmark each method on each GPU
    for (int gpu = 0; gpu < max_gpus; gpu++) {
        std::cout << "=== GPU " << gpu << " ===\n";

        for (int grid_size : grid_sizes) {
            for (int time_steps : time_steps_list) {
                std::cout << "Testing grid " << grid_size << "³, " << time_steps << " steps on GPU " << gpu << "\n";

                // OpenACC
                #ifdef USE_OPENACC
                std::cout << "  OpenACC... ";
                auto result = benchmark_method("OpenACC", Kernel_OpenACC, grid_size, time_steps, sources, gpu);
                all_results.push_back(result);
                std::cout << (result.success ? "PASS" : "FAIL") << " (" << std::fixed << std::setprecision(1) << result.gflops << " GFLOPS)\n";
                #endif

                // CUDA Baseline
                std::cout << "  CUDA_Baseline... ";
                auto result_cuda = benchmark_method("CUDA_Baseline", Kernel_CUDA, grid_size, time_steps, sources, gpu);
                all_results.push_back(result_cuda);
                std::cout << (result_cuda.success ? "PASS" : "FAIL") << " (" << std::fixed << std::setprecision(1) << result_cuda.gflops << " GFLOPS)\n";

                // Mixed Precision
                std::cout << "  Mixed_Precision... ";
                auto result_mixed = benchmark_method("Mixed_Precision", Kernel_Mixed_Precision, grid_size, time_steps, sources, gpu);
                all_results.push_back(result_mixed);
                std::cout << (result_mixed.success ? "PASS" : "FAIL") << " (" << std::fixed << std::setprecision(1) << result_mixed.gflops << " GFLOPS)\n";

                // Temporal Blocking
                std::cout << "  Temporal_Blocking... ";
                auto result_temporal = benchmark_method("Temporal_Blocking", Kernel_Temporal_Blocking, grid_size, time_steps, sources, gpu);
                all_results.push_back(result_temporal);
                std::cout << (result_temporal.success ? "PASS" : "FAIL") << " (" << std::fixed << std::setprecision(1) << result_temporal.gflops << " GFLOPS)\n";

                // Optimized Advanced
                std::cout << "  Optimized_Advanced... ";
                auto result_opt = benchmark_method("Optimized_Advanced", Kernel_Optimized_Advanced, grid_size, time_steps, sources, gpu);
                all_results.push_back(result_opt);
                std::cout << (result_opt.success ? "PASS" : "FAIL") << " (" << std::fixed << std::setprecision(1) << result_opt.gflops << " GFLOPS)\n";
            }
        }
        std::cout << "\n";
    }

    // Save results
    std::string timestamp = std::to_string(time(nullptr));
    std::string csv_filename = "fdtd_benchmark_results_" + timestamp + ".csv";
    save_results_csv(all_results, csv_filename);
    std::cout << "Results saved to: " << csv_filename << "\n";

    // Print summary
    print_summary(all_results);

    return 0;
}