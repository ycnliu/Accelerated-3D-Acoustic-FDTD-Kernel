#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <sys/time.h>
#include <vector>

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

struct TestResult {
    std::string name;
    bool success;
    double total_time;
    double kernel_time;
    double gflops;
    std::string error_msg;
};

class QuickValidator {
private:
    const int grid_size = 64;
    const int time_steps = 10;

    dataobj* create_dataobj(const std::vector<int>& dims) {
        dataobj *obj = new dataobj;
        obj->size = new int[dims.size()];
        for (size_t i = 0; i < dims.size(); i++) {
            obj->size[i] = dims[i];
        }

        size_t total_size = 1;
        for (int dim : dims) {
            total_size *= dim;
        }

        obj->data = new float[total_size]();
        obj->nbytes = total_size * sizeof(float);
        return obj;
    }

    void cleanup_dataobj(dataobj *obj) {
        delete[] (float*)obj->data;
        delete[] obj->size;
        delete obj;
    }

    void initialize_test_data(dataobj *u_vec, dataobj *m_vec, dataobj *src_vec, dataobj *src_coords_vec) {
        // Initialize u field
        float *u_data = (float*)u_vec->data;
        int u_total = u_vec->size[0] * u_vec->size[1] * u_vec->size[2] * u_vec->size[3];
        for (int i = 0; i < u_total; i++) {
            u_data[i] = 0.001f * sin(0.1f * i);
        }

        // Initialize medium properties
        float *m_data = (float*)m_vec->data;
        int m_total = m_vec->size[0] * m_vec->size[1] * m_vec->size[2];
        for (int i = 0; i < m_total; i++) {
            m_data[i] = 1.0f + 0.1f * cos(0.05f * i);
        }

        // Initialize source
        float *src_data = (float*)src_vec->data;
        int src_total = src_vec->size[0] * src_vec->size[1];
        for (int i = 0; i < src_total; i++) {
            src_data[i] = 0.1f * sin(0.2f * i);
        }

        // Source coordinates
        float *coords_data = (float*)src_coords_vec->data;
        coords_data[0] = grid_size / 2.0f;
        coords_data[1] = grid_size / 2.0f;
        coords_data[2] = grid_size / 2.0f;
    }

    bool validate_output(float *data, int size) {
        for (int i = 0; i < size; i++) {
            if (!std::isfinite(data[i])) return false;
        }
        return true;
    }

public:
    TestResult test_kernel(const std::string& name,
                          int (*kernel_func)(dataobj*, dataobj*, dataobj*, dataobj*, int, int, int, int, int, int,
                                           float, float, float, float, float, float, float, int, int, int, int,
                                           int, int, profiler*)) {
        TestResult result;
        result.name = name;
        result.success = false;

        try {
            // Create test data
            dataobj *u_vec = create_dataobj({3, grid_size + 8, grid_size + 8, grid_size + 8});
            dataobj *m_vec = create_dataobj({grid_size + 8, grid_size + 8, grid_size + 8});
            dataobj *src_vec = create_dataobj({time_steps, 1});
            dataobj *src_coords_vec = create_dataobj({1, 3});

            initialize_test_data(u_vec, m_vec, src_vec, src_coords_vec);

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
            int kernel_result = kernel_func(m_vec, src_vec, src_coords_vec, u_vec,
                                          x_M, x_m, y_M, y_m, z_M, z_m,
                                          dt, h_x, h_y, h_z, o_x, o_y, o_z,
                                          p_src_M, p_src_m, time_M, time_m,
                                          -1, 1, &timers);

            gettimeofday(&end, NULL);
            result.total_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
            result.kernel_time = timers.kernel_time;

            // Validate output
            float *u_data = (float*)u_vec->data;
            int u_total = u_vec->size[0] * u_vec->size[1] * u_vec->size[2] * u_vec->size[3];
            bool output_valid = validate_output(u_data, u_total);

            // Calculate GFLOPS
            double ops = (double)grid_size * grid_size * grid_size * time_steps * 37.0;
            result.gflops = ops / (timers.kernel_time * 1e9);

            result.success = (kernel_result == 0) && output_valid && (timers.kernel_time > 0);

            if (!result.success) {
                if (kernel_result != 0) result.error_msg = "Kernel returned error code " + std::to_string(kernel_result);
                else if (!output_valid) result.error_msg = "Output contains invalid values";
                else result.error_msg = "Zero kernel time";
            }

            // Cleanup
            cleanup_dataobj(u_vec);
            cleanup_dataobj(m_vec);
            cleanup_dataobj(src_vec);
            cleanup_dataobj(src_coords_vec);

        } catch (const std::exception& e) {
            result.error_msg = std::string("Exception: ") + e.what();
        } catch (...) {
            result.error_msg = "Unknown exception";
        }

        return result;
    }

    void run_all_tests() {
        std::cout << "=== Quick CUDA Validation Test ===" << std::endl;
        std::cout << "Grid: " << grid_size << "³, Time steps: " << time_steps << std::endl << std::endl;

        std::vector<TestResult> results;

        // Test each kernel
        results.push_back(test_kernel("CUDA_Baseline", Kernel_CUDA));
        results.push_back(test_kernel("Mixed_Precision", Kernel_Mixed_Precision));
        results.push_back(test_kernel("Temporal_Blocking", Kernel_Temporal_Blocking));

        // Display results
        std::cout << "Results:" << std::endl;
        std::cout << "--------" << std::endl;

        int passed = 0;
        for (const auto& result : results) {
            std::cout << result.name << ": ";
            if (result.success) {
                std::cout << "✓ SUCCESS (" << std::fixed << std::setprecision(1)
                         << result.gflops << " GFLOPS, "
                         << std::setprecision(3) << result.kernel_time << "s)" << std::endl;
                passed++;
            } else {
                std::cout << "✗ FAILED";
                if (!result.error_msg.empty()) {
                    std::cout << " - " << result.error_msg;
                }
                std::cout << std::endl;
            }
        }

        std::cout << std::endl << "Summary: " << passed << "/" << results.size()
                  << " kernels passed" << std::endl;

        if (passed == results.size()) {
            std::cout << "🎉 All CUDA implementations working!" << std::endl;
        } else {
            std::cout << "⚠️  " << (results.size() - passed) << " implementation(s) failed" << std::endl;
        }
    }
};

int main() {
    QuickValidator validator;
    validator.run_all_tests();
    return 0;
}