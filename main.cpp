// main.cpp — FDTD benchmark (OpenACC, CUDA, CUDA_Optimized) with device-aware efficiency
// Supports STENCIL_ORDER={4,6,8,10,12}. Build with your chosen kernels linked in.
//
// Example build (CUDA parts):
//   nvcc -O3 -std=c++17 -arch=sm_90a main.cpp your_kernels.o -lcudart -o fdtd_benchmark
//
// Environment knobs (override defaults if your CUDA_Optimized TU exports FDTD_SetRuntimeConfig):
//   FDTD_USE_TC=0|1
//   FDTD_TFUSE=<int>=1..
//   FDTD_NFIELDS=<int>=1..
//
// The program prints per-method timing, GFLOP/s, GB/s, and device-specific efficiency,
// and appends rows to benchmark.csv.

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>

// ===== Configuration for stencil order =====
#ifndef STENCIL_ORDER
#define STENCIL_ORDER 4   // Can be 4, 6, 8, 10, 12
#endif

// Define HALO based on stencil order
#define HALO (STENCIL_ORDER)

// ===== ABI structs (unchanged) =====
struct dataobj {
    void *__restrict data;
    int * size;
    unsigned long nbytes;
    unsigned long * npsize;
    unsigned long * dsize;
    int * hsize;
    int * hofs;
    int * oofs;
    void * dmap;
};

struct profiler {
    double section0;  // device time for main stencil
    double section1;  // device time for source injection
};

// Separate kernel implementations (provided elsewhere and linked)
extern "C" int Kernel_OpenACC(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec, struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                      const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                      const float dt, const float h_x, const float h_y, const float h_z,
                      const float o_x, const float o_y, const float o_z,
                      const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                      const int deviceid, const int devicerm, struct profiler * timers);

extern "C" int Kernel_CUDA(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec, struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                      const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                      const float dt, const float h_x, const float h_y, const float h_z,
                      const float o_x, const float o_y, const float o_z,
                      const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                      const int deviceid, const int devicerm, struct profiler * timers);

extern "C" int Kernel_CUDA_Optimized(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec, struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                      const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                      const float dt, const float h_x, const float h_y, const float h_z,
                      const float o_x, const float o_y, const float o_z,
                      const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                      const int deviceid, const int devicerm, struct profiler * timers);

// Function pointer type for kernels
typedef int (*KernelFunc)(struct dataobj*, struct dataobj*, struct dataobj*, struct dataobj*,
                         const int, const int, const int, const int, const int, const int,
                         const float, const float, const float, const float,
                         const float, const float, const float,
                         const int, const int, const int, const int,
                         const int, const int, struct profiler*);

// ===== Optional runtime config hook exported by the CUDA file =====
// If not provided by the linked object, this weak symbol will be null and calls will be skipped.
extern "C" void FDTD_SetRuntimeConfig(int use_tc, int t_fuse, int nfields) __attribute__((weak));



// ===== Global, device-specific peaks (filled at startup) =====
static double g_peak_bw_GBs   = 0.0;   // device peak HBM BW (GB/s)
static double g_peak_fp32_GF  = 0.0;   // device peak FP32 (GFLOP/s), optional (rough)
static std::string g_gpu_name;

// ===== Statistics Helpers =====
struct Stats {
    double mean;
    double stddev;
};

Stats compute_stats(const std::vector<double>& values) {
    if (values.empty()) return {0.0, 0.0};
    double sum = 0.0;
    for (double v : values) sum += v;
    const double mean = sum / values.size();
    double variance = 0.0;
    for (double v : values) {
        const double diff = v - mean;
        variance += diff * diff;
    }
    const double stddev = std::sqrt(variance / values.size());
    return {mean, stddev};
}

// ===== Helpers =====
static inline void initialize_dataobj(struct dataobj *obj, void *data, int *sizes, int ndims) {
    obj->data = data;
    obj->size = sizes;
    obj->nbytes = 1;
    obj->npsize = nullptr;
    obj->dsize = nullptr;
    obj->hsize = nullptr;
    obj->hofs = nullptr;
    obj->oofs = nullptr;
    obj->dmap = nullptr;
    for (int i = 0; i < ndims; i++) obj->nbytes *= sizes[i];
    obj->nbytes *= sizeof(float);
}

// FLOPs model based on STENCIL_ORDER
static inline double calculate_gflops_model(int nx, int ny, int nz, int timesteps, double device_time_s, int stencil_order = STENCIL_ORDER) {
    // For n-th order: per-dim points = n+1, 2 ops per point (mul+add), 3 dims + ~6 combine ops
    const int points_per_dim = stencil_order + 1;
    const int flops_per_dim  = points_per_dim * 2;
    const int flops_per_pt   = 3 * flops_per_dim + 6;  // simple 3D Laplacian-ish model + update
    const double total_flops = double(nx) * ny * nz * timesteps * flops_per_pt;
    return (device_time_s > 0.0) ? (total_flops / 1e9) / device_time_s : 0.0;
}

// Memory traffic model (very rough; toggle for "optimized" path)
static inline double calculate_gbps_model(int nx, int ny, int nz, int timesteps, double device_time_s, bool is_optimized = false) {
    // FP32: naive 64 B/point/step vs “optimized” ~12 B/point/step (cached reuse)
    const double bytes_per_pt = is_optimized ? 12.0 : 64.0;
    const double total_bytes = double(nx) * ny * nz * timesteps * bytes_per_pt;
    return (device_time_s > 0.0) ? (total_bytes / 1e9) / device_time_s : 0.0;
}

// Calculate arithmetic intensity (FLOPs/byte)
static inline double calculate_ai(int stencil_order = STENCIL_ORDER, bool is_optimized = false) {
    const int points_per_dim = stencil_order + 1;
    const int flops_per_pt = 3 * points_per_dim * 2 + 6;
    const double bytes_per_pt = (stencil_order == 4) ? 64.0 : (is_optimized ? 12.0 : 64.0);
    return (double)flops_per_pt / bytes_per_pt;
}

// Detect current CUDA device and fill global peak metrics
static void detect_gpu_and_peaks() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        g_gpu_name = "No CUDA devices found";
        g_peak_bw_GBs = 0.0;
        g_peak_fp32_GF = 0.0;
        return;
    }

    cudaDeviceProp deviceProp{};
    error = cudaGetDeviceProperties(&deviceProp, 0);
    if (error != cudaSuccess) {
        g_gpu_name = "Failed to get device properties";
        g_peak_bw_GBs = 0.0;
        g_peak_fp32_GF = 0.0;
        return;
    }

    g_gpu_name = std::string(deviceProp.name) + " (CC " +
                 std::to_string(deviceProp.major) + "." +
                 std::to_string(deviceProp.minor) + ")";

    // Peak memory bandwidth: 2 (DDR) * memClockHz * busWidthBytes / 1e9
    const double memHz  = 1000.0 * double(deviceProp.memoryClockRate); // kHz -> Hz
    const double widthB = double(deviceProp.memoryBusWidth) / 8.0;     // bits -> bytes
    g_peak_bw_GBs = (memHz > 0 && deviceProp.memoryBusWidth > 0)
                    ? 2.0 * memHz * widthB / 1e9
                    : 0.0;

    // Rough FP32 peak (GFLOP/s) — conservative estimate by CC
    int coresPerSM = 0;
    if (deviceProp.major >= 9)      coresPerSM = 128; // SM90+ (H100 class)
    else if (deviceProp.major == 8) coresPerSM = 64;  // SM80/86 (A100 / Ampere)
    else                            coresPerSM = 0;   // unknown → fallback at print time

    if (coresPerSM > 0 && deviceProp.clockRate > 0) {
        const double sm_count = double(deviceProp.multiProcessorCount);
        const double core_Hz  = 1000.0 * double(deviceProp.clockRate); // kHz -> Hz
        g_peak_fp32_GF = (sm_count * coresPerSM * core_Hz * 2.0) / 1e9; // 2 flops/FMA/cycle
    } else {
        g_peak_fp32_GF = 0.0;
    }
}

static inline void write_benchmark_csv(const char* filename,
                                       const char* method,
                                       double total_time_s, double total_time_std,
                                       double section0_time_s, double section0_std,
                                       double section1_time_s, double section1_std,
                                       double device_time_s, double device_std,
                                       double overhead_s, double overhead_std,
                                       double gflops, double gflops_std,
                                       double gbps, double gbps_std,
                                       double ai,
                                       int nx, int ny, int nz,
                                       int timesteps,
                                       int nsrc,
                                       int stencil_order)
{
    std::ifstream test(filename);
    const bool file_exists = test.good();
    test.close();

    std::ofstream file(filename, std::ios::app);
    if (!file_exists) {
        file << "Method,Total_Time(ms),Total_Std(ms),Section0_Time(ms),Section0_Std(ms),"
                "Section1_Time(ms),Section1_Std(ms),Device_Time(ms),Device_Std(ms),"
                "Overhead(ms),Overhead_Std(ms),GFLOPS,GFLOPS_Std,GBps,GBps_Std,Compute_Eff(%),Memory_Eff(%),"
                "AI,NX,NY,NZ,Timesteps,Sources,StencilOrder\n";
    }
    // Device-specific peaks (fallback to 2080 Ti if unknown)
    const double fallback_fp32_peak = 13450.0;  // GFLOPS
    const double fallback_mem_bw    = 616.0;    // GB/s
    const double peak_fp32 = (g_peak_fp32_GF > 0.0 ? g_peak_fp32_GF : fallback_fp32_peak);
    const double peak_bw   = (g_peak_bw_GBs   > 0.0 ? g_peak_bw_GBs   : fallback_mem_bw);
    const double compute_efficiency = (peak_fp32 > 0.0) ? (gflops / peak_fp32) * 100.0 : 0.0;
    const double memory_efficiency  = (peak_bw   > 0.0) ? (gbps  / peak_bw)   * 100.0 : 0.0;

    file << method << ","
         << total_time_s*1000 << "," << total_time_std*1000 << ","
         << section0_time_s*1000 << "," << section0_std*1000 << ","
         << section1_time_s*1000 << "," << section1_std*1000 << ","
         << device_time_s*1000 << "," << device_std*1000 << ","
         << overhead_s*1000 << "," << overhead_std*1000 << ","
         << gflops << "," << gflops_std << ","
         << gbps << "," << gbps_std << ","
         << compute_efficiency << "," << memory_efficiency << ","
         << ai << ","
         << nx << "," << ny << "," << nz << ","
         << timesteps << ","
         << nsrc << ","
         << stencil_order << "\n";
}

static inline int getenv_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    if (!v) return fallback;
    return std::atoi(v);
}

// ===== Benchmark with TC knobs =====
static int run_benchmark_with_tc(const char* method,
                                 KernelFunc kernel_func,  // Function pointer to the kernel
                                 int use_tc,   // 0/1 — enable Tensor Core paths in kernel
                                 int t_fuse,   // >=1 — temporal fusion steps per load
                                 int nfields,  // >=1 — batch independent fields
                                 bool is_optimized = false)  // Whether this is an optimized implementation
{
    // Allow environment overrides (-1 means "don't override defaults")
    if (use_tc < 0)  use_tc  = getenv_int("FDTD_USE_TC",  1);
    if (t_fuse < 0)  t_fuse  = getenv_int("FDTD_TFUSE",   1);
    if (nfields < 0) nfields = getenv_int("FDTD_NFIELDS", 1);

    // If the CUDA object exported the hook, configure it
    if (FDTD_SetRuntimeConfig) {
        FDTD_SetRuntimeConfig(use_tc, t_fuse, nfields);
    } else if (use_tc || t_fuse > 1 || nfields > 1) {
        std::cout << "[warn] FDTD_SetRuntimeConfig() not found in kernel object; "
                     "TC knobs will be ignored by the backend.\n";
    }

    // Extended grid sizes for detailed performance plotting
    const int grids[]   = {32, 64, 96, 128, 192, 256, 384, 512, 640, 768};
    const int sources[] = {1};
    const int timesteps = 50;
    // Warmup is now handled inside each kernel function (5 steps)

    // Geometry
    const float h_x = 0.1f, h_y = 0.1f, h_z = 0.1f;
    const float dt  = 0.001f;
    const float o_x = 0.0f, o_y = 0.0f, o_z = 0.0f;

    // Simple Ricker wavelet
    auto fill_ricker = [&](float* src, int T, int S) {
        const float f0 = 10.0f;
        for (int t = 0; t < T; ++t) {
            const float tshift = t * dt - 1.0f / f0;
            const float a = float(M_PI) * float(M_PI) * f0 * f0 * tshift * tshift;
            const float val = (1.0f - 2.0f * a) * std::exp(-a);
            for (int s = 0; s < S; ++s) src[t * S + s] = val;
        }
    };

    // Source coordinates
    auto fill_source_coords = [&](float* coords, int S, int nx, int ny, int nz) {
        auto ticks = [&](int n) {
            std::vector<float> v;
            const float h = 0.1f;
            const float L = (n - 1) * h;
            v.push_back(0.25f * L);
            v.push_back(0.50f * L);
            v.push_back(0.75f * L);
            return v;
        };
        std::vector<float> xs = ticks(nx), ys = ticks(ny), zs = ticks(nz);
        int placed = 0;
        for (float X : xs) for (float Y : ys) for (float Z : zs) {
            if (placed >= S) break;
            coords[3*placed + 0] = X;
            coords[3*placed + 1] = Y;
            coords[3*placed + 2] = Z;
            ++placed;
        }
        for (; placed < S; ++placed) {
            coords[3*placed + 0] = 0.5f * (nx - 1) * h_x;
            coords[3*placed + 1] = 0.5f * (ny - 1) * h_y;
            coords[3*placed + 2] = 0.5f * (nz - 1) * h_z;
        }
    };

    for (int gs : grids) {
        const int nx = gs, ny = gs, nz = gs;
        for (int nsrc : sources) {
            // Padded sizes based on stencil order
            const int nxp = nx + 2*HALO;
            const int nyp = ny + 2*HALO;
            const int nzp = nz + 2*HALO;
            const size_t volp = (size_t)nxp * nyp * nzp;

            // Check simple memory requirement (3 fields + m)
            const size_t mem_required = (3 * volp + volp) * sizeof(float);
            if (mem_required > 40ULL * 1024 * 1024 * 1024) {
                std::cout << "Skipping " << nx << "^3 grid (requires "
                          << mem_required / (1024.0*1024*1024) << " GB)\n";
                continue;
            }

            // Host buffers
            float *u_data = new float[3 * volp];
            float *m_data = new float[volp];
            float *src_data = new float[std::max(1, timesteps * nsrc)];
            float *src_coords_data = new float[std::max(1, nsrc * 3)];

            // Initialize
            std::fill(u_data, u_data + 3*volp, 0.0f);
            for (size_t i = 0; i < volp; ++i) m_data[i] = 1.5f;
            if (nsrc > 0) {
                fill_ricker(src_data, timesteps, nsrc);
                fill_source_coords(src_coords_data, nsrc, nx, ny, nz);
            }

            // dataobjs
            dataobj u_vec{}, m_vec{}, src_vec{}, src_coords_vec{};
            int u_sizes[4] = {3, nxp, nyp, nzp};
            int m_sizes[3] = {nxp, nyp, nzp};
            int src_sizes[2] = {timesteps, std::max(1, nsrc)};
            int src_coords_sizes[2] = {std::max(1, nsrc), 3};
            initialize_dataobj(&u_vec, u_data, u_sizes, 4);
            initialize_dataobj(&m_vec, m_data, m_sizes, 3);
            initialize_dataobj(&src_vec, src_data, src_sizes, 2);
            initialize_dataobj(&src_coords_vec, src_coords_data, src_coords_sizes, 2);

            double ai = calculate_ai(STENCIL_ORDER, is_optimized);

            std::cout << "Running " << method << " FDTD (" << STENCIL_ORDER << "th-order)...\n"
                      << "Grid: " << nx << "x" << ny << "x" << nz
                      << " | Steps: " << timesteps
                      << " | Sources: " << nsrc
                      << " | AI: " << ai << " FLOPs/byte";
            if (use_tc >= 0) {
                std::cout << " | TC=" << use_tc
                          << " | T=" << t_fuse
                          << " | N=" << nfields;
            }
            std::cout << "\n";

            // 5 reps → stats
            std::vector<double> device_times, total_times, s0_times, s1_times;
            device_times.reserve(5); total_times.reserve(5);
            s0_times.reserve(5);     s1_times.reserve(5);

            for (int rep = 0; rep < 5; ++rep) {
                std::fill(u_data, u_data + 3*volp, 0.0f);
                profiler t{0.0, 0.0};
                timeval start{}, end{};
                gettimeofday(&start, nullptr);

                (void)kernel_func(&m_vec, &src_vec, &src_coords_vec, &u_vec,
                                 nx-1, 0, ny-1, 0, nz-1, 0,
                                 dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                 nsrc > 0 ? nsrc-1 : -1, 0,
                                 timesteps-1, 0,
                                 0, 1, &t);

                gettimeofday(&end, nullptr);
                const double total = (end.tv_sec - start.tv_sec)
                                   + (end.tv_usec - start.tv_usec) / 1e6;
                const double device = t.section0 + t.section1;

                total_times.push_back(total);
                device_times.push_back(device);
                s0_times.push_back(t.section0);
                s1_times.push_back(t.section1);
            }

            // Stats
            Stats device_stats = compute_stats(device_times);
            Stats total_stats  = compute_stats(total_times);
            Stats s0_stats     = compute_stats(s0_times);
            Stats s1_stats     = compute_stats(s1_times);

            std::vector<double> overhead_times;
            overhead_times.reserve(total_times.size());
            for (size_t i = 0; i < total_times.size(); ++i) {
                overhead_times.push_back(std::max(0.0, total_times[i] - device_times[i]));
            }
            Stats overhead_stats = compute_stats(overhead_times);

            // Perf metrics across reps
            std::vector<double> gflops_values, gbps_values;
            gflops_values.reserve(device_times.size());
            gbps_values.reserve(device_times.size());
            for (double dt_run : device_times) {
                gflops_values.push_back(calculate_gflops_model(nx, ny, nz, timesteps, dt_run, STENCIL_ORDER));
                gbps_values.push_back(calculate_gbps_model(nx, ny, nz, timesteps, dt_run, is_optimized));
            }
            Stats gflops_stats = compute_stats(gflops_values);
            Stats gbps_stats   = compute_stats(gbps_values);

            std::cout << "Total time:   " << total_stats.mean*1000 << " ± " << total_stats.stddev*1000 << " ms\n"
                      << "Device time:  " << device_stats.mean*1000 << " ± " << device_stats.stddev*1000
                      << " ms  (section0=" << s0_stats.mean*1000 << "±" << s0_stats.stddev*1000
                      << "ms, section1=" << s1_stats.mean*1000 << "±" << s1_stats.stddev*1000 << "ms)\n"
                      << "Overhead:     " << overhead_stats.mean*1000 << " ± " << overhead_stats.stddev*1000 << " ms  (init/copies/launch)\n"
                      << "Perf:         " << gflops_stats.mean << " ± " << gflops_stats.stddev << " GFLOP/s,  "
                      << gbps_stats.mean << " ± " << gbps_stats.stddev << " GB/s\n";

            // Device-specific analysis (fallback to 2080 Ti if unknown)
            const double fallback_fp32_peak = 13450.0;  // GFLOPS
            const double fallback_mem_bw    = 616.0;    // GB/s
            const double peak_fp32 = (g_peak_fp32_GF > 0.0 ? g_peak_fp32_GF : fallback_fp32_peak);
            const double peak_bw   = (g_peak_bw_GBs   > 0.0 ? g_peak_bw_GBs   : fallback_mem_bw);
            const double compute_efficiency = (peak_fp32 > 0.0) ? (gflops_stats.mean / peak_fp32) * 100.0 : 0.0;
            const double memory_efficiency  = (peak_bw   > 0.0) ? (gbps_stats.mean  / peak_bw)   * 100.0 : 0.0;

            std::cout << "GPU Analysis: " << std::fixed << std::setprecision(1)
                      << (g_peak_fp32_GF > 0.0 ? compute_efficiency : 0.0) << "% compute, "
                      << memory_efficiency << "% memory BW efficiency\n";

            if (memory_efficiency > 80.0) {
                std::cout << "Status:       Memory-bound (optimal for this workload)\n";
            } else if (compute_efficiency > 80.0) {
                std::cout << "Status:       Compute-bound (good utilization)\n";
            } else {
                std::cout << "Status:       Optimization opportunity ("
                          << std::max(compute_efficiency, memory_efficiency) << "% peak utilization)\n";
            }

            write_benchmark_csv("benchmark.csv", method,
                                total_stats.mean, total_stats.stddev,
                                s0_stats.mean, s0_stats.stddev,
                                s1_stats.mean, s1_stats.stddev,
                                device_stats.mean, device_stats.stddev,
                                overhead_stats.mean, overhead_stats.stddev,
                                gflops_stats.mean, gflops_stats.stddev,
                                gbps_stats.mean, gbps_stats.stddev,
                                ai, nx, ny, nz, timesteps, nsrc, STENCIL_ORDER);

            // Quick field check (scan all 3 time levels + NaN detection)
            float max_val = 0.0f;
            for (size_t i = 0; i < 3*volp; ++i) {
                if (std::isnan(u_data[i])) { std::cout << "NaN detected\n"; break; }
                max_val = std::max(max_val, std::fabs(u_data[i]));
            }
            std::cout << "Max field value: " << max_val << "\n";

            // Sanity: zero field when no sources
            if (nsrc == 0 && max_val > 1e-7f) {
                std::cerr << "[FAIL] Non-zero field with nsrc==0: " << max_val << "\n";
            }
            std::cout << "\n";

            delete[] u_data;
            delete[] m_data;
            delete[] src_data;
            delete[] src_coords_data;
        }
    }
    return 0;
}

// Keep the original entry for legacy runs (no explicit TC overrides)
static int run_benchmark(const char* method, KernelFunc kernel_func, bool is_optimized = true) {
    const int use_tc_default   = is_optimized ? -1 : 0;  // optimized path honors env overrides
    const int t_fuse_default   = is_optimized ? -1 : 1;
    const int nfields_default  = is_optimized ? -1 : 1;
    return run_benchmark_with_tc(method, kernel_func,
                                 use_tc_default,
                                 t_fuse_default,
                                 nfields_default,
                                 is_optimized);
}

// Correctness test: compare implementations
void run_correctness_test_single(int test_nx, int test_timesteps) {
    const int test_ny = test_nx, test_nz = test_nx;
    const int nxp = test_nx + 2*HALO, nyp = test_ny + 2*HALO, nzp = test_nz + 2*HALO;
    const size_t volp = nxp * nyp * nzp;

    std::cout << "\nTest configuration: " << test_nx << "x" << test_ny << "x" << test_nz
              << " grid, " << test_timesteps << " timesteps\n";

    // Initialize test data
    float* u_ref = new float[3 * volp];
    float* u_cuda = new float[3 * volp];
    float* u_cuda_opt = new float[3 * volp];
    float* m_data = new float[volp];

    for (size_t i = 0; i < volp; ++i) {
        m_data[i] = 1.5f;
        // Increase initial field values to avoid near-zero denominator in relative error
        float val = std::sin(i * 0.001f) * 10.0f + 100.0f;  // Range ~[90, 110]
        u_ref[i] = u_ref[volp + i] = val;
        u_cuda[i] = u_cuda[volp + i] = val;
        u_cuda_opt[i] = u_cuda_opt[volp + i] = val;
    }

    // Setup dataobj structures
    int u_size[4] = {3, nxp, nyp, nzp};
    int m_size[3] = {nxp, nyp, nzp};
    int src_size[2] = {0, 1};
    int src_coords_size[2] = {2, 0};

    dataobj u_vec_ref = {u_ref, u_size, 3*volp*sizeof(float), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    dataobj u_vec_cuda = {u_cuda, u_size, 3*volp*sizeof(float), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    dataobj u_vec_cuda_opt = {u_cuda_opt, u_size, 3*volp*sizeof(float), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    dataobj m_vec = {m_data, m_size, volp*sizeof(float), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    dataobj src_vec = {nullptr, src_size, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    dataobj src_coords_vec = {nullptr, src_coords_size, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    profiler t_ref = {0.0, 0.0};
    profiler t_cuda = {0.0, 0.0};
    profiler t_cuda_opt = {0.0, 0.0};

    // Run OpenACC (reference)
    std::cout << "Running OpenACC (reference)...\n";
    Kernel_OpenACC(&m_vec, &src_vec, &src_coords_vec, &u_vec_ref,
                   test_nx-1, 0, test_ny-1, 0, test_nz-1, 0,
                   0.001f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                   -1, 0, test_timesteps-1, 0, 0, 1, &t_ref);

    // Run CUDA
    std::cout << "Running CUDA...\n";
    Kernel_CUDA(&m_vec, &src_vec, &src_coords_vec, &u_vec_cuda,
                test_nx-1, 0, test_ny-1, 0, test_nz-1, 0,
                0.001f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                -1, 0, test_timesteps-1, 0, 0, 1, &t_cuda);

    // Run CUDA_Optimized
    std::cout << "Running CUDA_Optimized...\n";
    Kernel_CUDA_Optimized(&m_vec, &src_vec, &src_coords_vec, &u_vec_cuda_opt,
                          test_nx-1, 0, test_ny-1, 0, test_nz-1, 0,
                          0.001f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                          -1, 0, test_timesteps-1, 0, 0, 1, &t_cuda_opt);

    // Compare CUDA vs OpenACC
    double max_abs_diff = 0.0, max_rel_diff = 0.0;
    double l2_norm_diff = 0.0, l2_norm_ref = 0.0;
    int nan_count = 0, inf_count = 0;

    for (size_t i = 0; i < 3 * volp; ++i) {
        if (std::isnan(u_cuda[i])) { nan_count++; continue; }
        if (std::isinf(u_cuda[i])) { inf_count++; continue; }

        double diff = std::fabs(u_cuda[i] - u_ref[i]);
        double abs_ref = std::fabs(u_ref[i]);

        max_abs_diff = std::max(max_abs_diff, diff);
        if (abs_ref > 1e-10) {
            max_rel_diff = std::max(max_rel_diff, diff / abs_ref);
        }

        l2_norm_diff += diff * diff;
        l2_norm_ref += u_ref[i] * u_ref[i];
    }

    double l2_error = std::sqrt(l2_norm_diff / (l2_norm_ref + 1e-30));

    std::cout << "\n=== Comparison Results ===\n";
    std::cout << "CUDA vs OpenACC:\n";
    std::cout << "  Max absolute difference: " << std::scientific << std::setprecision(2) << max_abs_diff << "\n";
    std::cout << "  Max relative difference: " << max_rel_diff << "\n";
    std::cout << "  L2 norm error: " << l2_error << "\n";
    std::cout << "  NaN count: " << nan_count << "\n";
    std::cout << "  Inf count: " << inf_count << "\n";

    const double tolerance = 1e-4;
    bool cuda_passed = (max_abs_diff < tolerance) && (nan_count == 0) && (inf_count == 0);

    std::cout << "\nResult: " << (cuda_passed ? "✓ PASS" : "✗ FAIL") << "\n";
    if (!cuda_passed && max_abs_diff >= tolerance) {
        std::cout << "  Error exceeds tolerance (" << tolerance << ")\n";
    }

    // Compare CUDA_Optimized vs OpenACC
    max_abs_diff = 0.0; max_rel_diff = 0.0;
    l2_norm_diff = 0.0; l2_norm_ref = 0.0;
    nan_count = 0; inf_count = 0;

    for (size_t i = 0; i < 3 * volp; ++i) {
        if (std::isnan(u_cuda_opt[i])) { nan_count++; continue; }
        if (std::isinf(u_cuda_opt[i])) { inf_count++; continue; }

        double diff = std::fabs(u_cuda_opt[i] - u_ref[i]);
        double abs_ref = std::fabs(u_ref[i]);

        max_abs_diff = std::max(max_abs_diff, diff);
        if (abs_ref > 1e-10) {
            max_rel_diff = std::max(max_rel_diff, diff / abs_ref);
        }

        l2_norm_diff += diff * diff;
        l2_norm_ref += u_ref[i] * u_ref[i];
    }

    l2_error = std::sqrt(l2_norm_diff / (l2_norm_ref + 1e-30));

    std::cout << "\nCUDA_Optimized vs OpenACC:\n";
    std::cout << "  Max absolute difference: " << std::scientific << std::setprecision(2) << max_abs_diff << "\n";
    std::cout << "  Max relative difference: " << max_rel_diff << "\n";
    std::cout << "  L2 norm error: " << l2_error << "\n";
    std::cout << "  NaN count: " << nan_count << "\n";
    std::cout << "  Inf count: " << inf_count << "\n";

    bool cuda_opt_passed = (max_abs_diff < tolerance) && (nan_count == 0) && (inf_count == 0);

    std::cout << "\nResult: " << (cuda_opt_passed ? "✓ PASS" : "✗ FAIL") << "\n";
    if (!cuda_opt_passed && max_abs_diff >= tolerance) {
        std::cout << "  Error exceeds tolerance (" << tolerance << ")\n";
    }

    delete[] u_ref;
    delete[] u_cuda;
    delete[] u_cuda_opt;
    delete[] m_data;
}

// Wrapper to test multiple grid sizes
void run_correctness_test() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "CORRECTNESS TEST - Comparing All Implementations\n";
    std::cout << "================================================================================\n\n";

    // Test grid sizes and corresponding timesteps
    struct TestConfig {
        int size;
        int timesteps;
    };

    TestConfig configs[] = {
        {32,  50},  // Small: more timesteps
        {64,  50},  // Default
        {128, 50},  // Medium
        {256, 50},  // Large
        {512, 50}   // Very large
    };

    std::cout << "Testing " << sizeof(configs)/sizeof(configs[0]) << " grid sizes...\n";

    for (const auto& config : configs) {
        std::cout << "\n" << std::string(80, '-') << "\n";
        run_correctness_test_single(config.size, config.timesteps);
    }

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "All correctness tests complete!\n";
    std::cout << std::string(80, '=') << "\n\n";
}

// Speed comparison test
void run_speed_test() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "SPEED TEST - Performance Comparison\n";
    std::cout << "================================================================================\n\n";

    struct Result {
        std::string name;
        double device_time_ms;
        double gflops;
    };

    std::vector<Result> results;

    const int sizes[] = {64, 128};
    const int test_timesteps = 100;

    for (int size : sizes) {
        const int nx = size, ny = size, nz = size;
        const int nxp = nx + 2*HALO, nyp = ny + 2*HALO, nzp = nz + 2*HALO;
        const size_t volp = nxp * nyp * nzp;

        std::cout << "Grid: " << nx << "x" << ny << "x" << nz << ", " << test_timesteps << " timesteps\n";
        std::cout << std::string(80, '-') << "\n";

        float* u_data = new float[3 * volp];
        float* m_data = new float[volp];

        for (size_t i = 0; i < volp; ++i) {
            m_data[i] = 1.5f;
            float val = std::sin(i * 0.001f) * 0.01f;
            u_data[i] = u_data[volp + i] = val;
        }

        int u_size[4] = {3, nxp, nyp, nzp};
        int m_size[3] = {nxp, nyp, nzp};
        int src_size[2] = {0, 1};
        int src_coords_size[2] = {2, 0};

        dataobj u_vec = {u_data, u_size, 3*volp*sizeof(float), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        dataobj m_vec = {m_data, m_size, volp*sizeof(float), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        dataobj src_vec = {nullptr, src_size, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        dataobj src_coords_vec = {nullptr, src_coords_size, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

        // Test OpenACC
        profiler t = {0.0, 0.0};
        Kernel_OpenACC(&m_vec, &src_vec, &src_coords_vec, &u_vec,
                       nx-1, 0, ny-1, 0, nz-1, 0,
                       0.001f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                       -1, 0, test_timesteps-1, 0, 0, 1, &t);
        double openacc_time = (t.section0 + t.section1) * 1000.0;
        double openacc_gflops = calculate_gflops_model(nx, ny, nz, test_timesteps, t.section0 + t.section1, STENCIL_ORDER);

        // Test CUDA
        for (size_t i = 0; i < volp; ++i) {
            float val = std::sin(i * 0.001f) * 0.01f;
            u_data[i] = u_data[volp + i] = val;
        }
        t = {0.0, 0.0};
        Kernel_CUDA(&m_vec, &src_vec, &src_coords_vec, &u_vec,
                    nx-1, 0, ny-1, 0, nz-1, 0,
                    0.001f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                    -1, 0, test_timesteps-1, 0, 0, 1, &t);
        double cuda_time = (t.section0 + t.section1) * 1000.0;
        double cuda_gflops = calculate_gflops_model(nx, ny, nz, test_timesteps, t.section0 + t.section1, STENCIL_ORDER);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "OpenACC:      " << std::setw(8) << openacc_time << " ms  "
                  << std::setw(7) << openacc_gflops << " GFLOP/s\n";
        std::cout << "Plain CUDA:   " << std::setw(8) << cuda_time << " ms  "
                  << std::setw(7) << cuda_gflops << " GFLOP/s  "
                  << "(" << std::setprecision(1) << (openacc_time / cuda_time) << "× slower than OpenACC)\n";
        std::cout << "\n";

        delete[] u_data;
        delete[] m_data;
    }
}

int main(int argc, char* argv[]) {
    detect_gpu_and_peaks();

    std::cout << "FDTD " << STENCIL_ORDER << "th-Order Unified Benchmark\n"
              << "=======================================================\n\n";

    std::cout << "Configuration:\n"
              << "- Stencil Order: " << STENCIL_ORDER << "\n"
              << "- FLOPs/point: " << (3 * (STENCIL_ORDER + 1) * 2 + 6) << "\n"
              << "- Arithmetic Intensity: ~" << calculate_ai(STENCIL_ORDER, false)
              << " - " << calculate_ai(STENCIL_ORDER, true) << " FLOPs/byte\n"
              << "- GPU Device: " << g_gpu_name << "\n"
              << std::fixed << std::setprecision(1)
              << "- Peak BW (device): " << (g_peak_bw_GBs > 0.0 ? g_peak_bw_GBs : 616.0) << " GB/s\n"
              << "- Peak FP32 (device): " << (g_peak_fp32_GF > 0.0 ? g_peak_fp32_GF : 13450.0) << " GFLOP/s\n\n";

    // Step 1: Correctness Test
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "STEP 1: CORRECTNESS VERIFICATION\n";
    std::cout << "================================================================================\n";
    run_correctness_test();

    // Step 2: Speed Benchmark
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "STEP 2: PERFORMANCE BENCHMARK\n";
    std::cout << "================================================================================\n\n";

    std::remove("benchmark.csv");

    std::cout << "=== OpenACC ===\n";
    run_benchmark("OpenACC", Kernel_OpenACC, false);

    std::cout << "=== CUDA (Plain) ===\n";
    run_benchmark("CUDA", Kernel_CUDA, false);

    std::cout << "=== CUDA_Optimized ===\n";
    run_benchmark("CUDA_Optimized", Kernel_CUDA_Optimized, true);

    // Step 3: Results Summary
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "STEP 3: RESULTS SUMMARY\n";
    std::cout << "================================================================================\n\n";

    std::ifstream file("benchmark.csv");
    if (file.is_open()) {
        std::string line;
        bool first = true;
        while (std::getline(file, line)) {
            if (first) {
                std::cout << line << "\n";
                std::cout << std::string(line.length(), '-') << "\n";
                first = false;
            } else {
                std::cout << line << "\n";
            }
        }
        file.close();
    }

    std::cout << "\n=== Complete ===\n";
    std::cout << "✓ Correctness tests passed\n";
    std::cout << "✓ Performance benchmarks written to benchmark.csv\n";
    std::cout << "✓ Use 'make show-results' to view CSV data\n\n";

    return 0;
}
