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
        file << "Method,Total_Time(ms),Total_Std(ms),Section0_Time(μs),Section0_Std(μs),"
                "Section1_Time(μs),Section1_Std(μs),Device_Time(μs),Device_Std(μs),"
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
         << section0_time_s*1000000 << "," << section0_std*1000000 << ","
         << section1_time_s*1000000 << "," << section1_std*1000000 << ","
         << device_time_s*1000000 << "," << device_std*1000000 << ","
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
    if (use_tc < 0)  use_tc  = getenv_int("FDTD_USE_TC",  use_tc  < 0 ? 0 : use_tc);
    if (t_fuse < 0)  t_fuse  = getenv_int("FDTD_TFUSE",   t_fuse  < 0 ? 1 : t_fuse);
    if (nfields < 0) nfields = getenv_int("FDTD_NFIELDS", nfields < 0 ? 1 : nfields);

    // If the CUDA object exported the hook, configure it
    if (FDTD_SetRuntimeConfig) {
        FDTD_SetRuntimeConfig(use_tc, t_fuse, nfields);
    } else if (use_tc || t_fuse > 1 || nfields > 1) {
        std::cout << "[warn] FDTD_SetRuntimeConfig() not found in kernel object; "
                     "TC knobs will be ignored by the backend.\n";
    }

    // Grids adjusted for higher-order halos
    const int grids[]   = {32, 64, 128, 256, 512, 1024};
    const int sources[] = {1};
    const int timesteps = 100;

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

            // Warmup (1 step)
            {
                profiler warm{0.0, 0.0};
                (void)kernel_func(&m_vec, &src_vec, &src_coords_vec, &u_vec,
                                 nx-1, 0, ny-1, 0, nz-1, 0,
                                 0.001f, 0.1f, 0.1f, 0.1f, 0.0f, 0.0f, 0.0f,
                                 nsrc > 0 ? nsrc-1 : -1, 0, 0, 0, 0, 1, &warm);
            }

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
                      << "Device time:  " << device_stats.mean*1000000 << " ± " << device_stats.stddev*1000000
                      << " μs  (section0=" << s0_stats.mean*1000000 << "±" << s0_stats.stddev*1000000
                      << "μs, section1=" << s1_stats.mean*1000000 << "±" << s1_stats.stddev*1000000 << "μs)\n"
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
    return run_benchmark_with_tc(method, kernel_func, /*use_tc=*/1, /*t_fuse=*/1, /*nfields=*/1, is_optimized);
}

int main(int argc, char* argv[]) {
    detect_gpu_and_peaks();

    std::cout << "FDTD " << STENCIL_ORDER << "th-Order Benchmark — Device-Only vs Overhead\n"
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

    std::string implementation = "all";
    if (argc > 1) implementation = argv[1];

    std::remove("benchmark.csv");

    // Original implementations
    if (implementation == "all" || implementation == "openacc") {
        std::cout << "=== OpenACC ===\n";
        run_benchmark("OpenACC", Kernel_OpenACC, false);
    }
    if (implementation == "all" || implementation == "cuda") {
        std::cout << "=== CUDA (Plain) ===\n";
        run_benchmark("CUDA", Kernel_CUDA, false);
    }
    if (implementation == "all" || implementation == "cuda_optimized") {
        std::cout << "=== CUDA_Optimized ===\n";
        run_benchmark("CUDA_Optimized", Kernel_CUDA_Optimized, true);
    }

    std::cout << "Benchmark complete. Results in benchmark.csv\n\n";

    // Print results summary
    std::cout << "=== Results Summary ===\n";
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

    std::cout << "\n=== Notes ===\n";
    std::cout << "Efficiency percentages use the active device's peak specs when detected.\n";
    std::cout << "Use FDTD_USE_TC / FDTD_TFUSE / FDTD_NFIELDS env vars to override TC knobs (if supported by your CUDA TU).\n";

    return 0;
}
