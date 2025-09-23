#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <fstream>
#include <chrono>
#include <iomanip>

// Include common data structures
#include "fdtd_common.h"

// Forward declarations for existing solver interfaces
extern "C" {

    int Kernel_CUDA(struct dataobj *m_vec, struct dataobj *src_vec,
                   struct dataobj *src_coords_vec, struct dataobj *u_vec,
                   const int x_M, const int x_m, const int y_M, const int y_m,
                   const int z_M, const int z_m, const float dt,
                   const float h_x, const float h_y, const float h_z,
                   const float o_x, const float o_y, const float o_z,
                   const int p_src_M, const int p_src_m,
                   const int time_M, const int time_m,
                   const int deviceid, const int devicerm, struct profiler *timers);

    int Kernel_OpenACC(struct dataobj *m_vec, struct dataobj *src_vec,
                      struct dataobj *src_coords_vec, struct dataobj *u_vec,
                      const int x_M, const int x_m, const int y_M, const int y_m,
                      const int z_M, const int z_m, const float dt,
                      const float h_x, const float h_y, const float h_z,
                      const float o_x, const float o_y, const float o_z,
                      const int p_src_M, const int p_src_m,
                      const int time_M, const int time_m,
                      const int deviceid, const int devicerm, struct profiler *timers);
}

#include "solver_validation.cpp"

class ValidationDriver {
private:
    struct TestConfig {
        int nx, ny, nz;
        float hx, hy, hz;
        float dt;
        float c_max;
        int num_timesteps;
        bool use_source;
        bool is_fp16_storage;
        std::string test_name;

        TestConfig(int n = 64, float h = 0.01f, float dt_val = 0.001f,
                  float c = 1.0f, int steps = 100, bool src = false,
                  bool fp16 = false, const std::string& name = "default")
            : nx(n), ny(n), nz(n), hx(h), hy(h), hz(h), dt(dt_val),
              c_max(c), num_timesteps(steps), use_source(src),
              is_fp16_storage(fp16), test_name(name) {}
    };

    SolverValidator validator;
    std::vector<TestConfig> test_suite;

    // Helper to allocate and initialize data objects
    struct dataobj* create_dataobj(const std::vector<int>& dims) {
        auto* obj = new dataobj();
        obj->size = new int[dims.size()];
        for (size_t i = 0; i < dims.size(); i++) {
            obj->size[i] = dims[i];
        }

        size_t total_size = 1;
        for (int dim : dims) total_size *= dim;
        obj->data = new float[total_size];
        std::memset(obj->data, 0, total_size * sizeof(float));

        return obj;
    }

    void cleanup_dataobj(struct dataobj* obj) {
        delete[] static_cast<float*>(obj->data);
        delete[] obj->size;
        delete obj;
    }

    void initialize_velocity_model(float* m, int nx, int ny, int nz, float c = 1.0f) {
        size_t total = nx * ny * nz;
        float m_val = 1.0f / (c * c); // m = 1/c^2
        for (size_t i = 0; i < total; i++) {
            static_cast<float*>(m)[i] = m_val;
        }
    }

    void setup_mms_initial_conditions(float* u, int nx, int ny, int nz,
                                     float hx, float hy, float hz,
                                     float kx = 2.0f, float ky = 2.0f, float kz = 2.0f) {
        // Set up time level arrays - assuming u has shape [3, nx, ny, nz]
        int vol = nx * ny * nz;

        // t0 = current time (t=0)
        float* u_t0 = u + 0 * vol;
        float* u_t1 = u + 1 * vol; // previous time (t=-dt)
        float* u_t2 = u + 2 * vol; // next time (t=+dt)

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    float x = i * hx;
                    float y = j * hy;
                    float z = k * hz;

                    int idx = i * ny * nz + j * nz + k;
                    float spatial = std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z);

                    u_t0[idx] = spatial;  // u(x,y,z,0) = sin(kx*x)*sin(ky*y)*sin(kz*z)
                    u_t1[idx] = spatial;  // u(x,y,z,-dt) ≈ u(x,y,z,0) for small dt
                    u_t2[idx] = 0.0f;     // Will be computed
                }
            }
        }
    }

    void setup_single_source_test(float* u, int nx, int ny, int nz) {
        size_t total = 3 * nx * ny * nz; // 3 time levels
        std::memset(u, 0, total * sizeof(float));

        // Place single nonzero value at center
        int center_x = nx / 2;
        int center_y = ny / 2;
        int center_z = nz / 2;
        int idx = center_x * ny * nz + center_y * nz + center_z;

        u[idx] = 1.0f; // t0
        u[nx*ny*nz + idx] = 1.0f; // t1 (previous)
    }

public:
    ValidationDriver() {
        // Setup test suite with various configurations
        test_suite = {
            TestConfig(64, 0.01f, 0.001f, 1.0f, 1, false, false, "parity_1step_fp32"),
            TestConfig(64, 0.01f, 0.001f, 1.0f, 10, false, false, "parity_10step_fp32"),
            TestConfig(64, 0.01f, 0.001f, 1.0f, 100, false, false, "parity_100step_fp32"),
            TestConfig(64, 0.01f, 0.001f, 1.0f, 100, false, true, "parity_100step_fp16"),
            TestConfig(64, 0.01f, 0.001f, 1.0f, 50, false, false, "zero_source_test"),
            TestConfig(64, 0.01f, 0.001f, 1.0f, 50, false, false, "single_voxel_test"),
            TestConfig(64, 0.01f, 0.001f, 1.0f, 200, false, false, "energy_conservation"),
            TestConfig(128, 0.005f, 0.0005f, 1.0f, 100, false, false, "mms_convergence_128"),
            TestConfig(96, 0.0067f, 0.00067f, 1.0f, 100, false, false, "mms_convergence_96"),
        };
    }

    // Test 1: Bit-for-bit parity checks
    bool run_parity_tests() {
        std::cout << "\n=== Bit-for-bit Parity Tests ===" << std::endl;
        bool all_passed = true;

        for (const auto& config : test_suite) {
            if (config.test_name.find("parity") == std::string::npos) continue;

            std::cout << "\nRunning: " << config.test_name << std::endl;

            // Create data objects
            auto u_vec = create_dataobj({3, config.nx + 8, config.ny + 8, config.nz + 8});
            auto m_vec = create_dataobj({config.nx + 8, config.ny + 8, config.nz + 8});
            auto src_vec = create_dataobj({config.num_timesteps + 1, 0}); // No sources
            auto src_coords_vec = create_dataobj({0, 3});

            // Initialize
            initialize_velocity_model(static_cast<float*>(m_vec->data),
                                    config.nx + 8, config.ny + 8, config.nz + 8, config.c_max);

            // Setup identical initial conditions
            float* u_data = static_cast<float*>(u_vec->data);
            setup_mms_initial_conditions(u_data, config.nx + 8, config.ny + 8, config.nz + 8,
                                       config.hx, config.hy, config.hz);

            // Copy for CUDA run
            size_t u_size = 3 * (config.nx + 8) * (config.ny + 8) * (config.nz + 8);
            std::vector<float> u_cuda_data(u_size);
            std::memcpy(u_cuda_data.data(), u_data, u_size * sizeof(float));

            // Create separate data object for CUDA
            auto u_cuda_vec = create_dataobj({3, config.nx + 8, config.ny + 8, config.nz + 8});
            std::memcpy(u_cuda_vec->data, u_cuda_data.data(), u_size * sizeof(float));

            // Setup profilers
            profiler cuda_timer = {0}, acc_timer = {0};

            // Run OpenACC
            int ret_acc = Kernel_OpenACC(m_vec, src_vec, src_coords_vec, u_vec,
                                       config.nx - 1, 2, config.ny - 1, 2, config.nz - 1, 2,
                                       config.dt, config.hx, config.hy, config.hz,
                                       0.0f, 0.0f, 0.0f, -1, 0,
                                       config.num_timesteps, 0, 0, 0, &acc_timer);

            // Run CUDA
            int ret_cuda = Kernel_CUDA(m_vec, src_vec, src_coords_vec, u_cuda_vec,
                                     config.nx - 1, 2, config.ny - 1, 2, config.nz - 1, 2,
                                     config.dt, config.hx, config.hy, config.hz,
                                     0.0f, 0.0f, 0.0f, -1, 0,
                                     config.num_timesteps, 0, 0, 0, &cuda_timer);

            if (ret_acc != 0 || ret_cuda != 0) {
                std::cout << "Error: Solver execution failed" << std::endl;
                all_passed = false;
                continue;
            }

            // Compare results
            auto result = validator.compute_parity_check(
                static_cast<float*>(u_cuda_vec->data),
                static_cast<float*>(u_vec->data),
                u_size, config.is_fp16_storage
            );

            validator.add_result(result);

            std::cout << "  L2 rel error: " << std::scientific << result.l2_rel_error << std::endl;
            std::cout << "  Linf rel error: " << std::scientific << result.linf_rel_error << std::endl;
            std::cout << "  Max abs error: " << std::scientific << result.max_abs_error << std::endl;
            std::cout << "  Result: " << (result.passes_threshold ? "PASS" : "FAIL") << std::endl;

            if (!result.passes_threshold) all_passed = false;

            // Cleanup
            cleanup_dataobj(u_vec);
            cleanup_dataobj(u_cuda_vec);
            cleanup_dataobj(m_vec);
            cleanup_dataobj(src_vec);
            cleanup_dataobj(src_coords_vec);
        }

        return all_passed;
    }

    // Test 2: No-source regression tests
    bool run_regression_tests() {
        std::cout << "\n=== No-source Regression Tests ===" << std::endl;
        bool all_passed = true;

        // Zero initial field test
        {
            auto config = TestConfig(64, 0.01f, 0.001f, 1.0f, 50, false, false, "zero_field");

            auto u_vec = create_dataobj({3, config.nx + 8, config.ny + 8, config.nz + 8});
            auto m_vec = create_dataobj({config.nx + 8, config.ny + 8, config.nz + 8});
            auto src_vec = create_dataobj({config.num_timesteps + 1, 0});
            auto src_coords_vec = create_dataobj({0, 3});

            initialize_velocity_model(static_cast<float*>(m_vec->data),
                                    config.nx + 8, config.ny + 8, config.nz + 8, config.c_max);

            // u_vec already initialized to zero by create_dataobj

            profiler timer = {0};
            int ret = Kernel_OpenACC(m_vec, src_vec, src_coords_vec, u_vec,
                                   config.nx - 1, 2, config.ny - 1, 2, config.nz - 1, 2,
                                   config.dt, config.hx, config.hy, config.hz,
                                   0.0f, 0.0f, 0.0f, -1, 0,
                                   config.num_timesteps, 0, 0, 0, &timer);

            if (ret == 0) {
                size_t total = 3 * (config.nx + 8) * (config.ny + 8) * (config.nz + 8);
                bool zero_test = validator.test_zero_initial_field(
                    static_cast<float*>(u_vec->data), total, config.num_timesteps
                );
                if (!zero_test) all_passed = false;
            } else {
                all_passed = false;
            }

            cleanup_dataobj(u_vec);
            cleanup_dataobj(m_vec);
            cleanup_dataobj(src_vec);
            cleanup_dataobj(src_coords_vec);
        }

        // Single voxel symmetry test
        {
            auto config = TestConfig(64, 0.01f, 0.001f, 1.0f, 20, false, false, "single_voxel");

            auto u_vec = create_dataobj({3, config.nx + 8, config.ny + 8, config.nz + 8});
            auto m_vec = create_dataobj({config.nx + 8, config.ny + 8, config.nz + 8});
            auto src_vec = create_dataobj({config.num_timesteps + 1, 0});
            auto src_coords_vec = create_dataobj({0, 3});

            initialize_velocity_model(static_cast<float*>(m_vec->data),
                                    config.nx + 8, config.ny + 8, config.nz + 8, config.c_max);

            setup_single_source_test(static_cast<float*>(u_vec->data),
                                   config.nx + 8, config.ny + 8, config.nz + 8);

            profiler timer = {0};
            int ret = Kernel_OpenACC(m_vec, src_vec, src_coords_vec, u_vec,
                                   config.nx - 1, 2, config.ny - 1, 2, config.nz - 1, 2,
                                   config.dt, config.hx, config.hy, config.hz,
                                   0.0f, 0.0f, 0.0f, -1, 0,
                                   config.num_timesteps, 0, 0, 0, &timer);

            if (ret == 0) {
                // Check final time level for symmetry
                int vol = (config.nx + 8) * (config.ny + 8) * (config.nz + 8);
                int final_t = config.num_timesteps % 3;
                float* u_final = static_cast<float*>(u_vec->data) + final_t * vol;

                bool sym_test = validator.test_single_voxel_symmetry(
                    u_final, config.nx + 8, config.ny + 8, config.nz + 8,
                    (config.nx + 8) / 2, (config.ny + 8) / 2, (config.nz + 8) / 2
                );
                if (!sym_test) all_passed = false;
            } else {
                all_passed = false;
            }

            cleanup_dataobj(u_vec);
            cleanup_dataobj(m_vec);
            cleanup_dataobj(src_vec);
            cleanup_dataobj(src_coords_vec);
        }

        return all_passed;
    }

    // Test 3: MMS convergence study
    bool run_mms_tests() {
        std::cout << "\n=== Method of Manufactured Solutions Tests ===" << std::endl;

        std::vector<TestConfig> mms_configs = {
            TestConfig(64, 0.01f, 0.001f, 1.0f, 50, false, false, "mms_64"),
            TestConfig(96, 0.0067f, 0.00067f, 1.0f, 75, false, false, "mms_96"),
            TestConfig(128, 0.005f, 0.0005f, 1.0f, 100, false, false, "mms_128"),
        };

        std::vector<double> errors, grid_sizes;

        for (const auto& config : mms_configs) {
            std::cout << "\nRunning MMS test: " << config.test_name << std::endl;

            auto u_vec = create_dataobj({3, config.nx + 8, config.ny + 8, config.nz + 8});
            auto m_vec = create_dataobj({config.nx + 8, config.ny + 8, config.nz + 8});
            auto src_vec = create_dataobj({config.num_timesteps + 1, 0});
            auto src_coords_vec = create_dataobj({0, 3});

            initialize_velocity_model(static_cast<float*>(m_vec->data),
                                    config.nx + 8, config.ny + 8, config.nz + 8, config.c_max);

            // Setup manufactured solution
            float* u_data = static_cast<float*>(u_vec->data);
            setup_mms_initial_conditions(u_data, config.nx + 8, config.ny + 8, config.nz + 8,
                                       config.hx, config.hy, config.hz);

            // Compute exact solution at final time
            size_t vol = (config.nx + 8) * (config.ny + 8) * (config.nz + 8);
            std::vector<float> u_exact(vol);
            float final_time = config.num_timesteps * config.dt;
            validator.setup_mms_solution(nullptr, u_exact.data(),
                                       config.nx + 8, config.ny + 8, config.nz + 8,
                                       config.hx, final_time);

            profiler timer = {0};
            int ret = Kernel_OpenACC(m_vec, src_vec, src_coords_vec, u_vec,
                                   config.nx - 1, 2, config.ny - 1, 2, config.nz - 1, 2,
                                   config.dt, config.hx, config.hy, config.hz,
                                   0.0f, 0.0f, 0.0f, -1, 0,
                                   config.num_timesteps, 0, 0, 0, &timer);

            if (ret == 0) {
                // Get final time level
                int final_t = config.num_timesteps % 3;
                float* u_final = u_data + final_t * vol;

                double error = validator.compute_mms_error(u_final, u_exact.data(), vol);
                errors.push_back(error);
                grid_sizes.push_back(config.hx);

                std::cout << "  Grid size: " << config.hx << ", Error: " << std::scientific << error << std::endl;
            }

            cleanup_dataobj(u_vec);
            cleanup_dataobj(m_vec);
            cleanup_dataobj(src_vec);
            cleanup_dataobj(src_coords_vec);
        }

        if (errors.size() >= 2) {
            double order = validator.estimate_convergence_order(errors, grid_sizes);
            std::cout << "Estimated convergence order: " << std::fixed << std::setprecision(2) << order << std::endl;

            // Expect at least 1.5 order (time-limited) for reasonable schemes
            bool order_ok = order >= 1.5;
            std::cout << "Convergence order: " << (order_ok ? "PASS" : "FAIL") << std::endl;
            return order_ok;
        }

        return false;
    }

    // Test 4: Energy conservation
    bool run_energy_conservation_test() {
        std::cout << "\n=== Energy Conservation Test ===" << std::endl;

        auto config = TestConfig(64, 0.01f, 0.001f, 1.0f, 200, false, false, "energy_test");

        auto u_vec = create_dataobj({3, config.nx + 8, config.ny + 8, config.nz + 8});
        auto m_vec = create_dataobj({config.nx + 8, config.ny + 8, config.nz + 8});
        auto src_vec = create_dataobj({config.num_timesteps + 1, 0});
        auto src_coords_vec = create_dataobj({0, 3});

        initialize_velocity_model(static_cast<float*>(m_vec->data),
                                config.nx + 8, config.ny + 8, config.nz + 8, config.c_max);

        // Setup initial wave packet
        float* u_data = static_cast<float*>(u_vec->data);
        setup_mms_initial_conditions(u_data, config.nx + 8, config.ny + 8, config.nz + 8,
                                   config.hx, config.hy, config.hz);

        std::vector<double> energy_history;
        float* m_data = static_cast<float*>(m_vec->data);

        // Compute initial energy
        int vol = (config.nx + 8) * (config.ny + 8) * (config.nz + 8);
        double initial_energy = validator.compute_discrete_energy(
            u_data + vol, u_data, u_data + 2*vol, m_data,
            config.nx + 8, config.ny + 8, config.nz + 8,
            config.dt, config.hx, config.hy, config.hz, config.c_max
        );
        energy_history.push_back(initial_energy);

        // Run simulation and monitor energy every 20 steps
        for (int step = 0; step < config.num_timesteps; step += 20) {
            int steps_to_run = std::min(20, config.num_timesteps - step);

            profiler timer = {0};
            int ret = Kernel_OpenACC(m_vec, src_vec, src_coords_vec, u_vec,
                                   config.nx - 1, 2, config.ny - 1, 2, config.nz - 1, 2,
                                   config.dt, config.hx, config.hy, config.hz,
                                   0.0f, 0.0f, 0.0f, -1, 0,
                                   steps_to_run, step, 0, 0, &timer);

            if (ret != 0) {
                std::cout << "Energy test failed: solver error" << std::endl;
                return false;
            }

            // Compute energy at current time
            int t0 = (step + steps_to_run) % 3;
            int t1 = (step + steps_to_run + 2) % 3;
            int t2 = (step + steps_to_run + 1) % 3;

            double energy = validator.compute_discrete_energy(
                u_data + t1*vol, u_data + t0*vol, u_data + t2*vol, m_data,
                config.nx + 8, config.ny + 8, config.nz + 8,
                config.dt, config.hx, config.hy, config.hz, config.c_max
            );
            energy_history.push_back(energy);
        }

        // Analyze energy drift
        double max_drift = 0.0;
        for (size_t i = 1; i < energy_history.size(); i++) {
            double drift = std::abs(energy_history[i] - energy_history[0]) / energy_history[0];
            max_drift = std::max(max_drift, drift);
        }

        std::cout << "Initial energy: " << std::scientific << initial_energy << std::endl;
        std::cout << "Final energy: " << std::scientific << energy_history.back() << std::endl;
        std::cout << "Max energy drift: " << std::fixed << std::setprecision(4) << max_drift * 100 << "%" << std::endl;

        bool conserved = max_drift < 0.001; // < 0.1% drift
        std::cout << "Energy conservation: " << (conserved ? "PASS" : "FAIL") << std::endl;

        cleanup_dataobj(u_vec);
        cleanup_dataobj(m_vec);
        cleanup_dataobj(src_vec);
        cleanup_dataobj(src_coords_vec);

        return conserved;
    }

    // Test 5: CFL and stability audit
    bool run_stability_audit() {
        std::cout << "\n=== CFL and Stability Audit ===" << std::endl;

        std::vector<TestConfig> stability_configs = {
            TestConfig(64, 0.01f, 0.001f, 1.0f, 10, false, false, "stable_cfl"),
            TestConfig(64, 0.01f, 0.002f, 1.0f, 10, false, false, "marginal_cfl"),
            TestConfig(64, 0.01f, 0.004f, 2.0f, 10, false, false, "unstable_cfl"),
        };

        bool all_stable = true;

        for (const auto& config : stability_configs) {
            double cfl = validator.compute_cfl_number(config.c_max, config.dt,
                                                    config.hx, config.hy, config.hz);

            std::cout << "\nTest: " << config.test_name << std::endl;
            bool stable = validator.check_stability_condition(cfl, 0.6f);

            if (config.test_name.find("unstable") == std::string::npos && !stable) {
                all_stable = false;
            }
        }

        return all_stable;
    }

    // Master validation runner
    void run_all_validations() {
        std::cout << "========================================" << std::endl;
        std::cout << "     FDTD Solver Validation Suite      " << std::endl;
        std::cout << "========================================" << std::endl;

        bool parity_pass = run_parity_tests();
        bool regression_pass = run_regression_tests();
        bool mms_pass = run_mms_tests();
        bool energy_pass = run_energy_conservation_test();
        bool stability_pass = run_stability_audit();

        std::cout << "\n========================================" << std::endl;
        std::cout << "           VALIDATION SUMMARY           " << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Parity Tests:        " << (parity_pass ? "PASS" : "FAIL") << std::endl;
        std::cout << "Regression Tests:    " << (regression_pass ? "PASS" : "FAIL") << std::endl;
        std::cout << "MMS Tests:           " << (mms_pass ? "PASS" : "FAIL") << std::endl;
        std::cout << "Energy Conservation: " << (energy_pass ? "PASS" : "FAIL") << std::endl;
        std::cout << "Stability Audit:     " << (stability_pass ? "PASS" : "FAIL") << std::endl;

        bool all_pass = parity_pass && regression_pass && mms_pass && energy_pass && stability_pass;
        std::cout << "\nOVERALL RESULT:      " << (all_pass ? "PASS" : "FAIL") << std::endl;
        std::cout << "========================================" << std::endl;

        // Generate comprehensive report
        validator.generate_validation_report("validation_report.md");
    }
};

int main() {
    ValidationDriver driver;
    driver.run_all_validations();
    return 0;
}