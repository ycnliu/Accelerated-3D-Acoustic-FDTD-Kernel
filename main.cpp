// main.cpp — apples-to-apples benchmark (device-only vs overhead, fixed FLOPs/pt)

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <cstdlib>
#include <vector>
#include <algorithm>

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

// Single symbol that resolves to whichever backend you link
extern "C" int Kernel(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec, struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                      const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                      const float dt, const float h_x, const float h_y, const float h_z,
                      const float o_x, const float o_y, const float o_z,
                      const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                      const int deviceid, const int devicerm, struct profiler * timers);

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

static inline double calculate_gflops_model(int nx, int ny, int nz, int timesteps, double device_time_s) {
    const double flops_per_pt = 36.0;
    const double total_flops = double(nx) * ny * nz * timesteps * flops_per_pt;
    return (device_time_s > 0.0) ? (total_flops / 1e9) / device_time_s : 0.0;
}

// Model memory traffic: ~64 bytes / point / step (FP32 model)
static inline double calculate_gbps_model(int nx, int ny, int nz, int timesteps, double device_time_s) {
    const double bytes_per_pt = 64.0;
    const double total_bytes = double(nx) * ny * nz * timesteps * bytes_per_pt;
    return (device_time_s > 0.0) ? (total_bytes / 1e9) / device_time_s : 0.0;
}


static inline void write_benchmark_csv(const char* filename,
                                       const char* method,
                                       double total_time_s,
                                       double section0_time_s,
                                       double section1_time_s,
                                       double device_time_s,
                                       double overhead_s,
                                       double gflops,
                                       double gbps,
                                       int nx, int ny, int nz,
                                       int timesteps,
                                       int nsrc)
{
    std::ifstream test(filename);
    const bool file_exists = test.good();
    test.close();

    std::ofstream file(filename, std::ios::app);
    if (!file_exists) {
        file << "Method,Total_Time(s),Section0_Time(s),Section1_Time(s),Device_Time(s),Overhead(s),"
                "GFLOPS,GBps,NX,NY,NZ,Timesteps,Sources\n";
    }
    file << method << ","
         << total_time_s << ","
         << section0_time_s << ","
         << section1_time_s << ","
         << device_time_s << ","
         << overhead_s << ","
         << gflops << ","
         << gbps << ","
         << nx << "," << ny << "," << nz << ","
         << timesteps << ","
         << nsrc << "\n";
}

// Correct FLOPs model for this stencil
static inline double calculate_gflops(int nx, int ny, int nz, int timesteps, double device_time_s) {
    const int flops_per_point = 36; // 4th-order 3D Laplacian + leapfrog (see breakdown)
    const long long pts = 1LL * nx * ny * nz;
    const long long updates = pts * timesteps;
    const long long flops = updates * flops_per_point;
    return (device_time_s > 0.0) ? (double)flops / device_time_s / 1e9 : 0.0;
}

// FP32 I/O model for bandwidth (baseline apples-to-apples)
static inline double calculate_gbps(int nx, int ny, int nz, int timesteps, double device_time_s) {
    const int bytes_per_point = 64; // 13 reads of u^n + u^{n-1} + m + store u^{n+1}
    const long long pts = 1LL * nx * ny * nz;
    const long long updates = pts * timesteps;
    const long long bytes = updates * bytes_per_point;
    return (device_time_s > 0.0) ? (double)bytes / device_time_s / 1e9 : 0.0;
}

static inline void reset_data(float *u_data, float *m_data, float *src_data, float *src_coords_data,
                              int nx, int ny, int nz, int timesteps) {
    const int nxp = nx + 8, nyp = ny + 8, nzp = nz + 8;
    // u (3 time levels)
    std::memset(u_data, 0, 3ULL * nxp * nyp * nzp * sizeof(float));
    // m
    for (long long i = 0; i < 1LL * nxp * nyp * nzp; ++i) m_data[i] = 1.5f;

    // Ricker source at one position
    const float f0 = 10.0f, dt = 1e-3f;
    for (int t = 0; t < timesteps; ++t) {
        float ts = t * dt - 1.0f / f0;
        float arg = (float)M_PI * (float)M_PI * f0 * f0 * ts * ts;
        src_data[t] = (1.0f - 2.0f * arg) * std::exp(-arg);
    }
    const float h_x = 0.1f, h_y = 0.1f, h_z = 0.1f;
    src_coords_data[0] = nx * h_x * 0.5f;
    src_coords_data[1] = ny * h_y * 0.5f;
    src_coords_data[2] = nz * h_z * 0.5f;
}

static int run_benchmark(const char* method) {
    // Grids and source counts to sweep
    const int grids[]   = {96, 128, 160, 192, 224, 256, 384, 512};
    const int sources[] = {8};
    const int timesteps = 200;

    // Geometry
    const float h_x = 0.1f, h_y = 0.1f, h_z = 0.1f;
    const float dt  = 0.001f;
    const float o_x = 0.0f, o_y = 0.0f, o_z = 0.0f;

    // Simple Ricker for all sources
    auto fill_ricker = [&](float* src, int T, int S) {
        const float f0 = 10.0f;
        for (int t = 0; t < T; ++t) {
            const float tshift = t * dt - 1.0f / f0;
            const float a = float(M_PI) * float(M_PI) * f0 * f0 * tshift * tshift;
            const float val = (1.0f - 2.0f * a) * std::exp(-a);
            for (int s = 0; s < S; ++s) src[t * S + s] = val;
        }
    };

    // Evenly distribute sources in the interior (1/4, 1/2, 3/4 positions)
    auto fill_source_coords = [&](float* coords, int S, int nx, int ny, int nz) {
        std::vector<float> xs, ys, zs;
        auto ticks = [&](int n) {
            std::vector<float> v;
            const float h = 0.1f;
            const float L = (n - 1) * h;
            v.push_back(0.25f * L);
            v.push_back(0.50f * L);
            v.push_back(0.75f * L);
            return v;
        };
        xs = ticks(nx); ys = ticks(ny); zs = ticks(nz);
        int placed = 0;
        for (float X : xs) for (float Y : ys) for (float Z : zs) {
            if (placed >= S) break;
            coords[3*placed + 0] = X;
            coords[3*placed + 1] = Y;
            coords[3*placed + 2] = Z;
            ++placed;
        }
        // If S < 27, center-fill remainder
        for (; placed < S; ++placed) {
            coords[3*placed + 0] = 0.5f * (nx - 1) * h_x;
            coords[3*placed + 1] = 0.5f * (ny - 1) * h_y;
            coords[3*placed + 2] = 0.5f * (nz - 1) * h_z;
        }
    };

    for (int gs : grids) {
        const int nx = gs, ny = gs, nz = gs;

        for (int nsrc : sources) {
            // Padded sizes (+4 halo per side)
            const int nxp = nx + 8, nyp = ny + 8, nzp = nz + 8;
            const long long volp = 1LL * nxp * nyp * nzp;

            // Host buffers
            float *u_data = new float[3LL * volp];
            float *m_data = new float[volp];
            float *src_data = new float[1LL * timesteps * nsrc];
            float *src_coords_data = new float[1LL * nsrc * 3];

            // Initialize u, m, sources/coords deterministically (no reset_data dependency)
            std::fill(u_data, u_data + 3LL*volp, 0.0f);
            for (long long i = 0; i < volp; ++i) m_data[i] = 1.5f;
            fill_ricker(src_data, timesteps, nsrc);
            fill_source_coords(src_coords_data, nsrc, nx, ny, nz);

            // dataobjs
            dataobj u_vec, m_vec, src_vec, src_coords_vec;
            int u_sizes[4] = {3, nxp, nyp, nzp};
            int m_sizes[3] = {nxp, nyp, nzp};
            int src_sizes[2] = {timesteps, nsrc};
            int src_coords_sizes[2] = {nsrc, 3};
            initialize_dataobj(&u_vec, u_data, u_sizes, 4);
            initialize_dataobj(&m_vec, m_data, m_sizes, 3);
            initialize_dataobj(&src_vec, src_data, src_sizes, 2);
            initialize_dataobj(&src_coords_vec, src_coords_data, src_coords_sizes, 2);

            std::cout << "Running " << method << " FDTD...\n"
                      << "Grid: " << nx << "x" << ny << "x" << nz
                      << " | Steps: " << timesteps
                      << " | Sources: " << nsrc << "\n";

            // Warmup (1 step)
            {
                profiler warm{0.0, 0.0};
                (void)Kernel(&m_vec, &src_vec, &src_coords_vec, &u_vec,
                             nx-1, 0, ny-1, 0, nz-1, 0,
                             dt, h_x, h_y, h_z, o_x, o_y, o_z,
                             /*p_src_M=*/nsrc-1, /*p_src_m=*/0,
                             /*time_M=*/0, /*time_m=*/0,
                             /*deviceid=*/0, /*devicerm=*/1, &warm);
            }

            // 5 reps → median
            std::vector<double> device_times, total_times, s0_times, s1_times;
            device_times.reserve(5); total_times.reserve(5);
            s0_times.reserve(5);     s1_times.reserve(5);

            for (int rep = 0; rep < 5; ++rep) {
                // re-init only u (sources/model are constant)
                std::fill(u_data, u_data + 3LL*volp, 0.0f);
                profiler t{0.0, 0.0};
                timeval start{}, end{};
                gettimeofday(&start, nullptr);

                int ok = Kernel(&m_vec, &src_vec, &src_coords_vec, &u_vec,
                                nx-1, 0, ny-1, 0, nz-1, 0,
                                dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                /*p_src_M=*/nsrc-1, /*p_src_m=*/0,
                                /*time_M=*/timesteps-1, /*time_m=*/0,
                                /*deviceid=*/0, /*devicerm=*/1, &t);
                (void)ok;

                gettimeofday(&end, nullptr);
                const double total = (end.tv_sec - start.tv_sec)
                                   + (end.tv_usec - start.tv_usec) / 1e6;
                const double device = t.section0 + t.section1;

                total_times.push_back(total);
                device_times.push_back(device);
                s0_times.push_back(t.section0);
                s1_times.push_back(t.section1);
            }

            auto median_of_5 = [](std::vector<double>& v) {
                std::sort(v.begin(), v.end());
                return v[2];
            };

            const double device_time_s = median_of_5(device_times);
            const double total_time_s  = median_of_5(total_times);
            const double s0_median     = median_of_5(s0_times);
            const double s1_median     = median_of_5(s1_times);
            const double overhead_s    = std::max(0.0, total_time_s - device_time_s);

            const double gflops = calculate_gflops_model(nx, ny, nz, timesteps, device_time_s);
            const double gbps   = calculate_gbps_model(nx, ny, nz, timesteps, device_time_s);

            std::cout << "Total time:   " << total_time_s  << " s\n"
                      << "Device time:  " << device_time_s << " s  (section0=" << s0_median
                      << ", section1=" << s1_median << ")\n"
                      << "Overhead:     " << overhead_s    << " s  (init/copies/launch)\n"
                      << "Perf:         " << gflops        << " GFLOP/s,  "
                      << gbps           << " GB/s\n";

            write_benchmark_csv("benchmark.csv", method,
                                total_time_s, s0_median, s1_median,
                                device_time_s, overhead_s,
                                gflops, gbps,
                                nx, ny, nz, timesteps, nsrc);

            // quick field check (center region)
            float max_val = 0.0f;
            for (long long i = 0; i < volp; ++i) max_val = std::max(max_val, std::fabs(u_data[i]));
            std::cout << "Max field value: " << max_val << "\n\n";

            delete[] u_data;
            delete[] m_data;
            delete[] src_data;
            delete[] src_coords_data;
        } // nsrc
    } // grids
    return 0;
}


int main(int argc, char* argv[]) {
    std::cout << "FDTD Benchmark — apples-to-apples (device-only vs overhead)\n"
              << "===========================================================\n\n";

    std::string implementation = "all";
    if (argc > 1) {
        implementation = argv[1];
    }

    std::remove("benchmark.csv");

    if (implementation == "all" || implementation == "openacc") {
        std::cout << "=== OpenACC ===\n";
        run_benchmark("OpenACC");
    }
    if (implementation == "all" || implementation == "cuda") {
        std::cout << "=== CUDA ===\n";
        run_benchmark("CUDA");
    }
    if (implementation == "all" || implementation == "cuda_optimized") {
        std::cout << "=== CUDA_Optimized ===\n";
        run_benchmark("CUDA_Optimized");
    }

    std::cout << "Benchmark complete. Results in benchmark.csv\n\n";
    std::cout << "=== Results Summary ===\n";
    std::ifstream file("benchmark.csv");
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) std::cout << line << "\n";
        file.close();
    }
    return 0;
}
