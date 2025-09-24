// main.cpp — apples-to-apples benchmark (device-only vs overhead, fixed FLOPs/pt)

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <cstdlib>

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

static inline void write_benchmark_csv(const char* filename, const char* method,
                                       double total_time_s, double section0_time_s, double section1_time_s, double device_time_s, double overhead_s,
                                       double gflops, double gbs,
                                       int nx, int ny, int nz, int timesteps) {
    std::ofstream file;
    // write header if not present
    std::ifstream test(filename);
    bool exists = test.good();
    test.close();

    file.open(filename, std::ios::app);
    if (!exists) {
        file << "Method,Total_Time(s),Section0_Time(s),Section1_Time(s),Device_Time(s),Overhead(s),GFLOPS,GBps,NX,NY,NZ,Timesteps\n";
    }
    file << method << ","
         << total_time_s << ","
         << section0_time_s << ","
         << section1_time_s << ","
         << device_time_s << ","
         << overhead_s << ","
         << gflops << ","
         << gbs << ","
         << nx << "," << ny << "," << nz << "," << timesteps << "\n";
    file.close();
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
    // Problem
    const int grid_size[5] = {64, 128, 256, 512, 768};
    for (auto gs : grid_size) {
        const int nx = gs, ny = gs, nz = gs;
        const int timesteps = 100;

        // Geometry
        const float h_x = 0.1f, h_y = 0.1f, h_z = 0.1f;
        const float dt  = 0.001f;
        const float o_x = 0.0f, o_y = 0.0f, o_z = 0.0f;

        // Padded sizes (+4 halo per side)
        const int nxp = nx + 8, nyp = ny + 8, nzp = nz + 8;

        // Host buffers
        float *u_data = new float[3ULL * nxp * nyp * nzp];
        float *m_data = new float[1ULL * nxp * nyp * nzp];
        const int nsrc = 1;
        float *src_data = new float[1ULL * timesteps * nsrc];
        float *src_coords_data = new float[1ULL * nsrc * 3];

        reset_data(u_data, m_data, src_data, src_coords_data, nx, ny, nz, timesteps);

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

        // Timers accumulated inside kernels
        profiler timers{0.0, 0.0};

        std::cout << "Running " << method << " FDTD...\n"
                << "Grid: " << nx << "x" << ny << "x" << nz
                << " | Steps: " << timesteps << "\n";

        // --- Warmup: 1 step, not timed in wall clock ---
        {
            profiler warm{0.0, 0.0};
            int warm_ok = Kernel(&m_vec, &src_vec, &src_coords_vec, &u_vec,
                                nx-1, 0, ny-1, 0, nz-1, 0,
                                dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                /*p_src_M=*/0, /*p_src_m=*/0,
                                /*time_M=*/0,  /*time_m=*/0,
                                /*deviceid=*/0, /*devicerm=*/1, &warm);
            (void)warm_ok;
            // Reinit for clean timing
            reset_data(u_data, m_data, src_data, src_coords_data, nx, ny, nz, timesteps);
        }

        // --- Wall timer around the entire call ---
        timeval start{}, end{};
        gettimeofday(&start, nullptr);

        int result = Kernel(&m_vec, &src_vec, &src_coords_vec, &u_vec,
                            nx-1, 0, ny-1, 0, nz-1, 0,
                            dt, h_x, h_y, h_z, o_x, o_y, o_z,
                            /*p_src_M=*/0, /*p_src_m=*/0,
                            /*time_M=*/timesteps-1, /*time_m=*/0,
                            /*deviceid=*/0, /*devicerm=*/1, &timers);

        gettimeofday(&end, nullptr);
        double total_time_s = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

        // Device-only time from kernel timers
        const double device_time_s = timers.section0 + timers.section1;
        const double overhead_s    = std::max(0.0, total_time_s - device_time_s);

        // Performance (apples-to-apples FP32 model)
        const double gflops = calculate_gflops(nx, ny, nz, timesteps, device_time_s);
        const double gbs    = calculate_gbps(nx, ny, nz, timesteps, device_time_s);

        std::cout << "Total time:   " << total_time_s  << " s\n"
                << "Device time:  " << device_time_s << " s  (section0=" << timers.section0
                << ", section1=" << timers.section1 << ")\n"
                << "Overhead:     " << overhead_s    << " s  (init/copies/launch)\n"
                << "Perf:         " << gflops        << " GFLOP/s,  "
                << gbs           << " GB/s\n";

        // CSV
        write_benchmark_csv("benchmark.csv", method, total_time_s, timers.section0, timers.section1, device_time_s, overhead_s, gflops, gbs, nx, ny, nz, timesteps);

        // Quick field check (same as before)
        float max_val = 0.0f;
        for (long long i = 0; i < 1LL * nxp * nyp * nzp; ++i) {
            float v = std::fabs(u_data[i]);
            if (v > max_val) max_val = v;
        }
        std::cout << "Max field value: " << max_val << "\n\n";

        delete[] u_data;
        delete[] m_data;
        delete[] src_data;
        delete[] src_coords_data;
    }
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
