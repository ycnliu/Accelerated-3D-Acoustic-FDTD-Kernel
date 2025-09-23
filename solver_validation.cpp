#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

struct ValidationResults {
    double l2_rel_error;
    double linf_rel_error;
    double max_abs_error;
    double correlation;
    double phase_delay;
    double amplitude_ratio;
    bool passes_threshold;
};

struct ValidationConfig {
    double fp32_l2_threshold = 1e-6;
    double fp32_linf_threshold = 1e-6;
    double fp16_l2_threshold = 1e-3;
    double fp16_linf_threshold = 1e-3;
    double correlation_threshold = 0.99;
    int validation_steps[3] = {1, 10, 100};
    int num_validation_points = 3;
};

class SolverValidator {
private:
    ValidationConfig config;
    std::vector<ValidationResults> results_history;

public:
    SolverValidator(ValidationConfig cfg = ValidationConfig()) : config(cfg) {}

    // 1. Bit-for-bit sanity & parity checks
    ValidationResults compute_parity_check(
        const float* u_cuda, const float* u_acc, size_t N,
        bool is_fp16_storage = false
    ) {
        ValidationResults result = {};

        double l2 = 0.0, l2ref = 0.0, linf = 0.0, linfden = 0.0;
        double max_abs = 0.0;

        for (size_t i = 0; i < N; i++) {
            double d = double(u_cuda[i]) - double(u_acc[i]);
            l2 += d * d;
            l2ref += double(u_acc[i]) * double(u_acc[i]);

            double abs_d = std::abs(d);
            linf = std::max(linf, abs_d);
            max_abs = std::max(max_abs, abs_d);
            linfden = std::max(linfden, std::abs(double(u_acc[i])));
        }

        result.l2_rel_error = std::sqrt(l2) / std::sqrt(l2ref + 1e-30);
        result.linf_rel_error = linf / (linfden + 1e-30);
        result.max_abs_error = max_abs;

        // Determine pass/fail based on precision
        if (is_fp16_storage) {
            result.passes_threshold = (result.l2_rel_error < config.fp16_l2_threshold) &&
                                     (result.linf_rel_error < config.fp16_linf_threshold);
        } else {
            result.passes_threshold = (result.l2_rel_error < config.fp32_l2_threshold) &&
                                     (result.linf_rel_error < config.fp32_linf_threshold);
        }

        return result;
    }

    // 2. No-source regression tests
    bool test_zero_initial_field(const float* u_final, size_t N, int steps) {
        double max_val = 0.0;
        for (size_t i = 0; i < N; i++) {
            max_val = std::max(max_val, std::abs(double(u_final[i])));
        }

        // Should remain essentially zero (within numerical precision)
        bool passes = max_val < 1e-10;

        std::cout << "Zero initial field test (" << steps << " steps): "
                  << "max_val = " << std::scientific << max_val
                  << " [" << (passes ? "PASS" : "FAIL") << "]" << std::endl;

        return passes;
    }

    bool test_single_voxel_symmetry(const float* u, int nx, int ny, int nz,
                                   int center_x, int center_y, int center_z) {
        auto get_idx = [&](int x, int y, int z) {
            return x * ny * nz + y * nz + z;
        };

        double center_val = u[get_idx(center_x, center_y, center_z)];

        // Check symmetry in x-direction
        double asymmetry = 0.0;
        int checks = 0;

        for (int r = 1; r < std::min({center_x, nx-center_x-1, center_y, ny-center_y-1, center_z, nz-center_z-1}); r++) {
            double val_px = u[get_idx(center_x + r, center_y, center_z)];
            double val_mx = u[get_idx(center_x - r, center_y, center_z)];
            double val_py = u[get_idx(center_x, center_y + r, center_z)];
            double val_my = u[get_idx(center_x, center_y - r, center_z)];
            double val_pz = u[get_idx(center_x, center_y, center_z + r)];
            double val_mz = u[get_idx(center_x, center_y, center_z - r)];

            asymmetry += std::abs(val_px - val_mx) + std::abs(val_py - val_my) + std::abs(val_pz - val_mz);
            checks += 3;
        }

        double avg_asymmetry = asymmetry / checks;
        bool symmetric = avg_asymmetry < 1e-5 * std::abs(center_val);

        std::cout << "Single voxel symmetry test: avg_asymmetry = " << std::scientific
                  << avg_asymmetry << " [" << (symmetric ? "PASS" : "FAIL") << "]" << std::endl;

        return symmetric;
    }

    // 3. Receiver (seismogram) comparisons
    ValidationResults compare_receivers(
        const std::vector<float>& ref_timeseries,
        const std::vector<float>& test_timeseries
    ) {
        ValidationResults result = {};

        if (ref_timeseries.size() != test_timeseries.size()) {
            std::cerr << "Error: timeseries length mismatch" << std::endl;
            return result;
        }

        size_t n = ref_timeseries.size();

        // Compute normalized cross-correlation
        double sum_ref = 0.0, sum_test = 0.0;
        for (size_t i = 0; i < n; i++) {
            sum_ref += ref_timeseries[i];
            sum_test += test_timeseries[i];
        }
        double mean_ref = sum_ref / n;
        double mean_test = sum_test / n;

        double numerator = 0.0, denom_ref = 0.0, denom_test = 0.0;
        for (size_t i = 0; i < n; i++) {
            double ref_centered = ref_timeseries[i] - mean_ref;
            double test_centered = test_timeseries[i] - mean_test;

            numerator += ref_centered * test_centered;
            denom_ref += ref_centered * ref_centered;
            denom_test += test_centered * test_centered;
        }

        result.correlation = numerator / std::sqrt(denom_ref * denom_test + 1e-30);

        // Amplitude ratio (RMS-based)
        double rms_ref = std::sqrt(denom_ref / n);
        double rms_test = std::sqrt(denom_test / n);
        result.amplitude_ratio = rms_test / (rms_ref + 1e-30);

        // Simple phase delay estimation (peak-to-peak)
        auto find_peak = [](const std::vector<float>& data) {
            return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
        };

        int peak_ref = find_peak(ref_timeseries);
        int peak_test = find_peak(test_timeseries);
        result.phase_delay = peak_test - peak_ref;

        result.passes_threshold = result.correlation >= config.correlation_threshold;

        return result;
    }

    // 4. Method of Manufactured Solutions (MMS)
    void setup_mms_solution(float* u_initial, float* u_exact,
                           int nx, int ny, int nz, float h, float t,
                           float kx = 1.0, float ky = 1.0, float kz = 1.0, float c = 1.0) {
        float omega = c * std::sqrt(kx*kx + ky*ky + kz*kz);

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    float x = i * h;
                    float y = j * h;
                    float z = k * h;

                    int idx = i * ny * nz + j * nz + k;
                    float spatial = std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z);

                    u_initial[idx] = spatial; // cos(omega * 0) = 1
                    u_exact[idx] = spatial * std::cos(omega * t);
                }
            }
        }
    }

    double compute_mms_error(const float* u_numerical, const float* u_exact, size_t N) {
        double error = 0.0, norm = 0.0;

        for (size_t i = 0; i < N; i++) {
            double diff = u_numerical[i] - u_exact[i];
            error += diff * diff;
            norm += u_exact[i] * u_exact[i];
        }

        return std::sqrt(error) / std::sqrt(norm + 1e-30);
    }

    double estimate_convergence_order(const std::vector<double>& errors,
                                    const std::vector<double>& grid_sizes) {
        if (errors.size() < 2) return 0.0;

        // Fit slope of log(error) vs log(h)
        double sum_log_h = 0.0, sum_log_err = 0.0, sum_log_h2 = 0.0, sum_log_h_log_err = 0.0;
        int n = errors.size();

        for (int i = 0; i < n; i++) {
            double log_h = std::log(grid_sizes[i]);
            double log_err = std::log(errors[i]);

            sum_log_h += log_h;
            sum_log_err += log_err;
            sum_log_h2 += log_h * log_h;
            sum_log_h_log_err += log_h * log_err;
        }

        double slope = (n * sum_log_h_log_err - sum_log_h * sum_log_err) /
                       (n * sum_log_h2 - sum_log_h * sum_log_h);

        return slope; // This should be ~4 for 4th-order spatial, ~2 for 2nd-order temporal
    }

    // 5. Discrete energy conservation check
    double compute_discrete_energy(const float* u_prev, const float* u_curr, const float* u_next,
                                  const float* m, int nx, int ny, int nz,
                                  float dt, float hx, float hy, float hz, float c) {
        double energy = 0.0;
        float dt2 = 2.0f * dt;
        float inv_hx2 = 1.0f / (hx * hx);
        float inv_hy2 = 1.0f / (hy * hy);
        float inv_hz2 = 1.0f / (hz * hz);

        auto get_idx = [&](int x, int y, int z) {
            return x * ny * nz + y * nz + z;
        };

        // Interior points only (avoid boundary effects)
        for (int i = 2; i < nx - 2; i++) {
            for (int j = 2; j < ny - 2; j++) {
                for (int k = 2; k < nz - 2; k++) {
                    int idx = get_idx(i, j, k);

                    // Kinetic energy term: (u^{n+1} - u^{n-1})^2 / (2*dt)^2
                    float dudtdt = (u_next[idx] - u_prev[idx]) / dt2;
                    double kinetic = dudtdt * dudtdt;

                    // Potential energy term: c^2 * |grad u|^2
                    float dudx = inv_hx2 * (
                        -8.33333333e-2f * (u_curr[get_idx(i-2,j,k)] + u_curr[get_idx(i+2,j,k)]) +
                        1.333333330f * (u_curr[get_idx(i-1,j,k)] + u_curr[get_idx(i+1,j,k)]) +
                        -2.5f * u_curr[idx]
                    );

                    float dudy = inv_hy2 * (
                        -8.33333333e-2f * (u_curr[get_idx(i,j-2,k)] + u_curr[get_idx(i,j+2,k)]) +
                        1.333333330f * (u_curr[get_idx(i,j-1,k)] + u_curr[get_idx(i,j+1,k)]) +
                        -2.5f * u_curr[idx]
                    );

                    float dudz = inv_hz2 * (
                        -8.33333333e-2f * (u_curr[get_idx(i,j,k-2)] + u_curr[get_idx(i,j,k+2)]) +
                        1.333333330f * (u_curr[get_idx(i,j,k-1)] + u_curr[get_idx(i,j,k+1)]) +
                        -2.5f * u_curr[idx]
                    );

                    double potential = c * c * (dudx * dudx + dudy * dudy + dudz * dudz);

                    energy += kinetic + potential;
                }
            }
        }

        return energy;
    }

    // 6. CFL and stability audit
    double compute_cfl_number(float c_max, float dt, float hx, float hy, float hz) {
        float h_min_inv = std::sqrt(1.0f/(hx*hx) + 1.0f/(hy*hy) + 1.0f/(hz*hz));
        return c_max * dt * h_min_inv;
    }

    bool check_stability_condition(float cfl, float stability_limit = 0.5f) {
        bool stable = cfl <= stability_limit;
        std::cout << "CFL number: " << cfl << " (limit: " << stability_limit
                  << ") [" << (stable ? "STABLE" : "UNSTABLE") << "]" << std::endl;
        return stable;
    }

    // Comprehensive validation report
    void generate_validation_report(const std::string& filename) {
        std::ofstream report(filename);
        report << "# Solver Validation Report" << std::endl;
        report << "Generated on: " << __DATE__ << " " << __TIME__ << std::endl << std::endl;

        report << "## Configuration" << std::endl;
        report << "FP32 L2 threshold: " << config.fp32_l2_threshold << std::endl;
        report << "FP32 Linf threshold: " << config.fp32_linf_threshold << std::endl;
        report << "FP16 L2 threshold: " << config.fp16_l2_threshold << std::endl;
        report << "FP16 Linf threshold: " << config.fp16_linf_threshold << std::endl;
        report << "Correlation threshold: " << config.correlation_threshold << std::endl << std::endl;

        report << "## Validation Results" << std::endl;
        for (size_t i = 0; i < results_history.size(); i++) {
            const auto& result = results_history[i];
            report << "Test " << i+1 << ":" << std::endl;
            report << "  L2 relative error: " << std::scientific << result.l2_rel_error << std::endl;
            report << "  Linf relative error: " << std::scientific << result.linf_rel_error << std::endl;
            report << "  Max absolute error: " << std::scientific << result.max_abs_error << std::endl;
            report << "  Correlation: " << std::fixed << std::setprecision(6) << result.correlation << std::endl;
            report << "  Passed: " << (result.passes_threshold ? "YES" : "NO") << std::endl << std::endl;
        }

        report.close();
        std::cout << "Validation report written to: " << filename << std::endl;
    }

    void add_result(const ValidationResults& result) {
        results_history.push_back(result);
    }
};

// External interface functions for integration with existing code
extern "C" {
    SolverValidator* create_validator() {
        return new SolverValidator();
    }

    void destroy_validator(SolverValidator* validator) {
        delete validator;
    }

    ValidationResults validate_parity(SolverValidator* validator,
                                    const float* u_cuda, const float* u_acc,
                                    size_t N, int is_fp16) {
        return validator->compute_parity_check(u_cuda, u_acc, N, is_fp16 != 0);
    }

    int validate_zero_field(SolverValidator* validator,
                           const float* u_final, size_t N, int steps) {
        return validator->test_zero_initial_field(u_final, N, steps) ? 1 : 0;
    }

    double compute_cfl(float c_max, float dt, float hx, float hy, float hz) {
        SolverValidator validator;
        return validator.compute_cfl_number(c_max, dt, hx, hy, hz);
    }

    void write_validation_report(SolverValidator* validator, const char* filename) {
        validator->generate_validation_report(std::string(filename));
    }
}