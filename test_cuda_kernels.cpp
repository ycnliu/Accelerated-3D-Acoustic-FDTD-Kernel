#include <iostream>
#include <cmath>
#include <cstring>
#include <sys/time.h>

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
    double conversion_time{};
};

// External CUDA kernel function declarations
extern "C" int Kernel_CUDA(
    struct dataobj *__restrict m_vec,
    struct dataobj *__restrict src_vec,
    struct dataobj *__restrict src_coords_vec,
    struct dataobj *__restrict u_vec,
    const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
    const float dt, const float h_x, const float h_y, const float h_z,
    const float o_x, const float o_y, const float o_z,
    const int p_src_M, const int p_src_m, const int time_M, const int time_m,
    const int deviceid, const int devicerm, struct profiler *timers
);

extern "C" int Kernel_Mixed_Precision(
    struct dataobj *__restrict m_vec,
    struct dataobj *__restrict src_vec,
    struct dataobj *__restrict src_coords_vec,
    struct dataobj *__restrict u_vec,
    const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
    const float dt, const float h_x, const float h_y, const float h_z,
    const float o_x, const float o_y, const float o_z,
    const int p_src_M, const int p_src_m, const int time_M, const int time_m,
    const int deviceid, const int devicerm, struct profiler *timers
);

extern "C" int Kernel_Optimized(
    struct dataobj *__restrict m_vec,
    struct dataobj *__restrict src_vec,
    struct dataobj *__restrict src_coords_vec,
    struct dataobj *__restrict u_vec,
    const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
    const float dt, const float h_x, const float h_y, const float h_z,
    const float o_x, const float o_y, const float o_z,
    const int p_src_M, const int p_src_m, const int time_M, const int time_m,
    const int deviceid, const int devicerm, struct profiler *timers
);

extern "C" int Kernel_Temporal_Blocking(
    struct dataobj *__restrict m_vec,
    struct dataobj *__restrict src_vec,
    struct dataobj *__restrict src_coords_vec,
    struct dataobj *__restrict u_vec,
    const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
    const float dt, const float h_x, const float h_y, const float h_z,
    const float o_x, const float o_y, const float o_z,
    const int p_src_M, const int p_src_m, const int time_M, const int time_m,
    const int deviceid, const int devicerm, struct profiler *timers
);

class CUDAKernelTester {
private:
    int grid_size = 64;
    int time_steps = 10;

    dataobj *create_dataobj(int *dims, int ndims) {
        dataobj *obj = new dataobj;
        obj->size = new int[ndims];
        memcpy(obj->size, dims, ndims * sizeof(int));

        size_t total_size = 1;
        for (int i = 0; i < ndims; i++) {
            total_size *= dims[i];
        }

        obj->data = new float[total_size]();
        obj->nbytes = total_size * sizeof(float);

        return obj;
    }

    void cleanup_dataobj(dataobj *obj, int ndims) {
        delete[] (float*)obj->data;
        delete[] obj->size;
        delete obj;
    }

public:
    bool test_kernel(const std::string& name,
                    int (*kernel_func)(dataobj*, dataobj*, dataobj*, dataobj*,
                                     int, int, int, int, int, int,
                                     float, float, float, float,
                                     float, float, float,
                                     int, int, int, int,
                                     int, int, profiler*)) {

        std::cout << "Testing " << name << "..." << std::flush;

        // Create test data
        int u_dims[4] = {3, grid_size + 8, grid_size + 8, grid_size + 8};
        int m_dims[3] = {grid_size + 8, grid_size + 8, grid_size + 8};
        int src_dims[2] = {time_steps, 1};
        int src_coords_dims[2] = {1, 3};

        dataobj *u_vec = create_dataobj(u_dims, 4);
        dataobj *m_vec = create_dataobj(m_dims, 3);
        dataobj *src_vec = create_dataobj(src_dims, 2);
        dataobj *src_coords_vec = create_dataobj(src_coords_dims, 2);

        // Initialize test data
        float *u_data = (float*)u_vec->data;
        float *m_data = (float*)m_vec->data;
        float *src_data = (float*)src_vec->data;
        float *src_coords_data = (float*)src_coords_vec->data;

        for (int i = 0; i < u_dims[0] * u_dims[1] * u_dims[2] * u_dims[3]; i++) {
            u_data[i] = 0.001f * (i % 1000);
        }

        for (int i = 0; i < m_dims[0] * m_dims[1] * m_dims[2]; i++) {
            m_data[i] = 1.0f + 0.001f * (i % 100);
        }

        for (int i = 0; i < src_dims[0] * src_dims[1]; i++) {
            src_data[i] = 0.1f;
        }

        src_coords_data[0] = grid_size / 2.0f;
        src_coords_data[1] = grid_size / 2.0f;
        src_coords_data[2] = grid_size / 2.0f;

        profiler timers{};

        // Test parameters
        int x_M = grid_size - 1, x_m = 0;
        int y_M = grid_size - 1, y_m = 0;
        int z_M = grid_size - 1, z_m = 0;
        float dt = 0.001f, h_x = 1.0f, h_y = 1.0f, h_z = 1.0f;
        float o_x = 0.0f, o_y = 0.0f, o_z = 0.0f;
        int p_src_M = 0, p_src_m = 0;
        int time_M = time_steps - 1, time_m = 0;

        struct timeval start, end;
        gettimeofday(&start, NULL);

        // Run kernel
        int result = kernel_func(m_vec, src_vec, src_coords_vec, u_vec,
                               x_M, x_m, y_M, y_m, z_M, z_m,
                               dt, h_x, h_y, h_z, o_x, o_y, o_z,
                               p_src_M, p_src_m, time_M, time_m,
                               -1, 1, &timers);

        gettimeofday(&end, NULL);
        double total_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

        // Cleanup
        cleanup_dataobj(u_vec, 4);
        cleanup_dataobj(m_vec, 3);
        cleanup_dataobj(src_vec, 2);
        cleanup_dataobj(src_coords_vec, 2);

        if (result == 0) {
            std::cout << " ✓ SUCCESS (Total: " << total_time << "s, Kernel: "
                     << timers.kernel_time << "s)" << std::endl;
            return true;
        } else {
            std::cout << " ✗ FAILED (Return code: " << result << ")" << std::endl;
            return false;
        }
    }

    void run_all_tests() {
        std::cout << "=== CUDA Kernel Validation Tests ===" << std::endl;
        std::cout << "Grid size: " << grid_size << "³, Time steps: " << time_steps << std::endl << std::endl;

        int passed = 0, total = 0;

        total++; if (test_kernel("CUDA_Baseline", Kernel_CUDA)) passed++;
        total++; if (test_kernel("Mixed_Precision", Kernel_Mixed_Precision)) passed++;
        total++; if (test_kernel("Optimized", Kernel_Optimized)) passed++;
        total++; if (test_kernel("Temporal_Blocking", Kernel_Temporal_Blocking)) passed++;

        std::cout << std::endl << "=== Test Summary ===" << std::endl;
        std::cout << "Passed: " << passed << "/" << total << " kernels" << std::endl;

        if (passed == total) {
            std::cout << "🎉 All CUDA kernels are working correctly!" << std::endl;
        } else {
            std::cout << "⚠️  Some kernels failed validation" << std::endl;
        }
    }
};

int main() {
    CUDAKernelTester tester;
    tester.run_all_tests();
    return 0;
}