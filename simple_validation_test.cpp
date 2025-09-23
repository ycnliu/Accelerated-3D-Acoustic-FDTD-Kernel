#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

// Include the validation framework
#include "solver_validation.cpp"

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   Simple FDTD Validation Test         " << std::endl;
    std::cout << "========================================" << std::endl;

    SolverValidator validator;
    bool all_tests_passed = true;

    // Test 1: Basic parity check with synthetic data
    std::cout << "\n=== Test 1: Bit-for-bit Parity Check ===" << std::endl;
    {
        size_t N = 1000;
        std::vector<float> u_cuda(N), u_acc(N);

        // Generate synthetic test data
        for (size_t i = 0; i < N; i++) {
            float val = std::sin(i * 0.1f) * std::exp(-i * 0.001f);
            u_cuda[i] = val;
            u_acc[i] = val + 1e-7f * std::sin(i * 0.3f); // Small perturbation
        }

        // Add some non-zero values to avoid division by zero
        u_cuda[0] = 1.0f;
        u_acc[0] = 1.0f + 1e-6f;

        auto result = validator.compute_parity_check(u_cuda.data(), u_acc.data(), N, false);
        validator.add_result(result);

        std::cout << "L2 relative error: " << std::scientific << result.l2_rel_error << std::endl;
        std::cout << "L∞ relative error: " << std::scientific << result.linf_rel_error << std::endl;
        std::cout << "Max absolute error: " << std::scientific << result.max_abs_error << std::endl;
        std::cout << "Result: " << (result.passes_threshold ? "PASS" : "FAIL") << std::endl;

        if (!result.passes_threshold) all_tests_passed = false;
    }

    // Test 2: Zero field test
    std::cout << "\n=== Test 2: Zero Field Test ===" << std::endl;
    {
        size_t N = 1000;
        std::vector<float> u_zeros(N, 0.0f);

        bool zero_test = validator.test_zero_initial_field(u_zeros.data(), N, 100);
        if (!zero_test) all_tests_passed = false;
    }

    // Test 3: Single voxel symmetry test
    std::cout << "\n=== Test 3: Single Voxel Symmetry ===" << std::endl;
    {
        int nx = 32, ny = 32, nz = 32;
        std::vector<float> u(nx * ny * nz, 0.0f);

        // Create symmetric pattern
        auto get_idx = [&](int x, int y, int z) { return x * ny * nz + y * nz + z; };

        int cx = nx/2, cy = ny/2, cz = nz/2;
        for (int r = 1; r < 10; r++) {
            float val = 1.0f / (r * r); // Decay with distance
            if (cx + r < nx) u[get_idx(cx + r, cy, cz)] = val;
            if (cx - r >= 0) u[get_idx(cx - r, cy, cz)] = val;
            if (cy + r < ny) u[get_idx(cx, cy + r, cz)] = val;
            if (cy - r >= 0) u[get_idx(cx, cy - r, cz)] = val;
            if (cz + r < nz) u[get_idx(cx, cy, cz + r)] = val;
            if (cz - r >= 0) u[get_idx(cx, cy, cz - r)] = val;
        }

        bool sym_test = validator.test_single_voxel_symmetry(u.data(), nx, ny, nz, cx, cy, cz);
        if (!sym_test) all_tests_passed = false;
    }

    // Test 4: Receiver comparison
    std::cout << "\n=== Test 4: Receiver Comparison ===" << std::endl;
    {
        std::vector<float> ref_series, test_series;
        int n_samples = 200;

        // Generate synthetic seismograms
        for (int i = 0; i < n_samples; i++) {
            float t = i * 0.01f;
            float signal = std::sin(2.0f * M_PI * 5.0f * t) * std::exp(-t * 0.5f);
            ref_series.push_back(signal);
            test_series.push_back(signal + 0.01f * std::sin(2.0f * M_PI * 15.0f * t)); // Add noise
        }

        auto result = validator.compare_receivers(ref_series, test_series);
        std::cout << "Correlation: " << std::fixed << std::setprecision(6) << result.correlation << std::endl;
        std::cout << "Amplitude ratio: " << std::fixed << std::setprecision(3) << result.amplitude_ratio << std::endl;
        std::cout << "Phase delay: " << result.phase_delay << " samples" << std::endl;
        std::cout << "Result: " << (result.passes_threshold ? "PASS" : "FAIL") << std::endl;

        if (!result.passes_threshold) all_tests_passed = false;
    }

    // Test 5: MMS error computation
    std::cout << "\n=== Test 5: Method of Manufactured Solutions ===" << std::endl;
    {
        int nx = 32, ny = 32, nz = 32; // Smaller grid to avoid memory issues
        float h = 0.01f, t = 0.1f;
        std::vector<float> u_exact(nx * ny * nz), u_numerical(nx * ny * nz);

        try {
            // Create dummy initial array since we only need exact solution
            std::vector<float> u_initial(nx * ny * nz);
            validator.setup_mms_solution(u_initial.data(), u_exact.data(), nx, ny, nz, h, t);

            // Simulate numerical solution with some error
            for (size_t i = 0; i < u_exact.size(); i++) {
                u_numerical[i] = u_exact[i] * (1.0f + 1e-4f * std::sin(i * 0.1f));
            }

            double error = validator.compute_mms_error(u_numerical.data(), u_exact.data(), u_exact.size());
            std::cout << "MMS relative error: " << std::scientific << error << std::endl;

            bool mms_ok = error < 1e-3 && std::isfinite(error); // Reasonable threshold for this test
            std::cout << "Result: " << (mms_ok ? "PASS" : "FAIL") << std::endl;
            if (!mms_ok) all_tests_passed = false;
        } catch (const std::exception& e) {
            std::cout << "MMS test failed with exception: " << e.what() << std::endl;
            all_tests_passed = false;
        }
    }

    // Test 6: Energy conservation
    std::cout << "\n=== Test 6: Energy Conservation ===" << std::endl;
    {
        int nx = 32, ny = 32, nz = 32;
        float dt = 0.001f, hx = 0.01f, hy = 0.01f, hz = 0.01f, c = 1.0f;

        std::vector<float> u_prev(nx * ny * nz), u_curr(nx * ny * nz), u_next(nx * ny * nz);
        std::vector<float> m(nx * ny * nz, 1.0f); // Constant velocity model

        // Initialize with a wave packet
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    float x = i * hx, y = j * hy, z = k * hz;
                    float r2 = (x-0.15f)*(x-0.15f) + (y-0.15f)*(y-0.15f) + (z-0.15f)*(z-0.15f);
                    int idx = i * ny * nz + j * nz + k;
                    u_curr[idx] = std::exp(-r2 / 0.01f);
                    u_prev[idx] = u_curr[idx];
                    u_next[idx] = 0.0f;
                }
            }
        }

        double energy = validator.compute_discrete_energy(
            u_prev.data(), u_curr.data(), u_next.data(), m.data(),
            nx, ny, nz, dt, hx, hy, hz, c
        );

        std::cout << "Discrete energy: " << std::scientific << energy << std::endl;

        bool energy_ok = energy > 0.0 && std::isfinite(energy);
        std::cout << "Result: " << (energy_ok ? "PASS" : "FAIL") << std::endl;
        if (!energy_ok) all_tests_passed = false;
    }

    // Test 7: CFL computation
    std::cout << "\n=== Test 7: CFL and Stability ===" << std::endl;
    {
        float c_max = 1.5f, dt = 0.001f, hx = 0.01f, hy = 0.01f, hz = 0.01f;

        double cfl = validator.compute_cfl_number(c_max, dt, hx, hy, hz);
        bool stable = validator.check_stability_condition(cfl, 0.6f);

        if (!stable) all_tests_passed = false;
    }

    // Test 8: Convergence order estimation
    std::cout << "\n=== Test 8: Convergence Order Estimation ===" << std::endl;
    {
        std::vector<double> errors = {1e-2, 2.5e-3, 6.25e-4, 1.56e-4};
        std::vector<double> grid_sizes = {0.02, 0.01, 0.005, 0.0025};

        double order = validator.estimate_convergence_order(errors, grid_sizes);
        std::cout << "Estimated convergence order: " << std::fixed << std::setprecision(2) << order << std::endl;

        bool order_ok = order >= 1.5 && order <= 5.0; // Reasonable range
        std::cout << "Result: " << (order_ok ? "PASS" : "FAIL") << std::endl;
        if (!order_ok) all_tests_passed = false;
    }

    // Generate validation report
    validator.generate_validation_report("simple_validation_report.md");

    // Final summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "           VALIDATION SUMMARY           " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Overall result: " << (all_tests_passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "Report saved to: simple_validation_report.md" << std::endl;
    std::cout << "========================================" << std::endl;

    return all_tests_passed ? 0 : 1;
}