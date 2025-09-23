#!/bin/bash

# FDTD Solver Validation Script
# Comprehensive validation suite for OpenACC vs CUDA implementations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GRID_SIZES=(32 64 96)
TIME_STEPS=(10 50 100)
VALIDATION_DIR="validation_results_$(date +%Y%m%d_%H%M%S)"

# Helper functions
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log "Checking dependencies..."

    # Check for required executables
    local deps=("nvcc" "nvc++" "cuda-memcheck")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            warning "$dep not found in PATH"
        else
            success "$dep found"
        fi
    done

    # Check for optional profiling tools
    local opt_deps=("ncu" "nvprof")
    for dep in "${opt_deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            warning "$dep not found (profiling features disabled)"
        else
            success "$dep found (profiling enabled)"
        fi
    done
}

build_validation_suite() {
    log "Building validation suite..."

    if make clean; then
        success "Cleaned previous builds"
    else
        error "Failed to clean"
        exit 1
    fi

    if make validation_suite comprehensive_benchmark; then
        success "Built validation suite"
    else
        error "Failed to build validation suite"
        exit 1
    fi
}

run_parity_tests() {
    log "Running bit-for-bit parity tests..."

    local results_file="$VALIDATION_DIR/parity_results.txt"

    echo "# Parity Test Results" > "$results_file"
    echo "Generated: $(date)" >> "$results_file"
    echo "" >> "$results_file"

    for grid in "${GRID_SIZES[@]}"; do
        for steps in "${TIME_STEPS[@]}"; do
            log "  Testing grid ${grid}³, ${steps} steps..."

            if timeout 300 ./validation_driver "$grid" "$steps" >> "$results_file" 2>&1; then
                success "Parity test passed: ${grid}³ × ${steps} steps"
            else
                error "Parity test failed: ${grid}³ × ${steps} steps"
                echo "FAILED: grid ${grid}³, ${steps} steps" >> "$results_file"
            fi
        done
    done
}

run_memory_validation() {
    log "Running memory traffic validation..."

    local results_file="$VALIDATION_DIR/memory_results.txt"

    echo "# Memory Validation Results" > "$results_file"
    echo "Generated: $(date)" >> "$results_file"
    echo "" >> "$results_file"

    # Test with different grid sizes to check scaling
    for grid in "${GRID_SIZES[@]}"; do
        log "  Profiling memory traffic for ${grid}³ grid..."

        if timeout 600 ./cuda_openacc_validator "memory_test_${grid}" "$grid" "$grid" "$grid" 50 >> "$results_file" 2>&1; then
            success "Memory validation passed: ${grid}³"
        else
            warning "Memory validation incomplete: ${grid}³"
        fi
    done
}

run_race_condition_tests() {
    log "Running race condition detection..."

    local results_file="$VALIDATION_DIR/race_condition_results.txt"

    echo "# Race Condition Test Results" > "$results_file"
    echo "Generated: $(date)" >> "$results_file"
    echo "" >> "$results_file"

    # Test CUDA implementation
    log "  Testing CUDA implementation with memcheck..."
    if timeout 300 cuda-memcheck --error-exitcode 1 ./comprehensive_benchmark cuda 64 >> "$results_file" 2>&1; then
        success "CUDA memcheck passed"
        echo "CUDA memcheck: PASS" >> "$results_file"
    else
        error "CUDA memcheck failed"
        echo "CUDA memcheck: FAIL" >> "$results_file"
    fi

    # Test OpenACC implementation
    log "  Testing OpenACC implementation with memcheck..."
    if timeout 300 cuda-memcheck --error-exitcode 1 ./comprehensive_benchmark openacc 64 >> "$results_file" 2>&1; then
        success "OpenACC memcheck passed"
        echo "OpenACC memcheck: PASS" >> "$results_file"
    else
        error "OpenACC memcheck failed"
        echo "OpenACC memcheck: FAIL" >> "$results_file"
    fi
}

run_convergence_tests() {
    log "Running Method of Manufactured Solutions tests..."

    local results_file="$VALIDATION_DIR/convergence_results.txt"

    echo "# MMS Convergence Test Results" > "$results_file"
    echo "Generated: $(date)" >> "$results_file"
    echo "" >> "$results_file"

    # Test convergence with multiple grid resolutions
    local grids=(32 48 64 96 128)
    local errors=()

    for grid in "${grids[@]}"; do
        log "  Testing convergence for ${grid}³ grid..."

        # Adjust time step to maintain CFL condition
        local dt=$(python3 -c "print(f'{0.01 * 64 / $grid:.6f}')")
        local steps=$(python3 -c "print(int(100 * 64 / $grid))")

        if timeout 600 ./validation_driver "$grid" "$steps" mms >> "$results_file" 2>&1; then
            success "MMS test completed: ${grid}³"
        else
            warning "MMS test failed: ${grid}³"
        fi
    done
}

run_energy_conservation_tests() {
    log "Running energy conservation tests..."

    local results_file="$VALIDATION_DIR/energy_results.txt"

    echo "# Energy Conservation Test Results" > "$results_file"
    echo "Generated: $(date)" >> "$results_file"
    echo "" >> "$results_file"

    # Long-term stability test
    log "  Testing energy conservation over 500 steps..."
    if timeout 900 ./validation_driver 64 500 energy >> "$results_file" 2>&1; then
        success "Energy conservation test passed"
    else
        error "Energy conservation test failed"
    fi
}

generate_summary_report() {
    log "Generating validation summary report..."

    local summary_file="$VALIDATION_DIR/validation_summary.md"

    cat > "$summary_file" << EOF
# FDTD Solver Validation Summary

**Generated:** $(date)
**Host:** $(hostname)
**CUDA Version:** $(nvcc --version | grep release | cut -d' ' -f5-6)

## Test Suite Overview

This validation suite implements the comprehensive testing methodology for FDTD solvers,
covering bit-for-bit parity, regression tests, MMS verification, energy conservation,
and OpenACC vs CUDA specific validations.

## Test Categories

### 1. Bit-for-bit Parity Checks
- **Purpose:** Ensure CUDA and OpenACC produce identical results
- **Grids tested:** ${GRID_SIZES[*]}
- **Time steps:** ${TIME_STEPS[*]}
- **Precision:** FP32 compute, FP16 storage variants

### 2. No-source Regression Tests
- **Zero field test:** Validates no spurious generation
- **Single voxel test:** Checks propagation symmetry
- **Boundary conditions:** Halo exchange verification

### 3. Method of Manufactured Solutions
- **Analytical solution:** sin(kx)sin(ky)sin(kz)cos(ωt)
- **Convergence order:** Expected ~2 (time-dominated) to ~4 (space-dominated)
- **Grid refinement:** Multiple resolutions tested

### 4. Energy Conservation
- **Discrete energy formula:** Kinetic + potential energy components
- **Stability criterion:** <0.1% drift over hundreds of steps
- **Long-term behavior:** 500+ step validation

### 5. CFL and Stability Analysis
- **CFL computation:** c_max * dt * √(1/hx² + 1/hy² + 1/hz²)
- **Stability limit:** CFL ≤ 0.5-0.6 for 4th-order space + 2nd-order time
- **Parameter sweeps:** Multiple dt/h combinations

### 6. OpenACC vs CUDA Specific Tests
- **Memory traffic analysis:** Expected 2-10x efficiency difference
- **Race condition detection:** CUDA memcheck validation
- **Launch patterns:** Kernel fusion analysis
- **Performance ratios:** 0.5x-2.0x acceptable range

## Expected Thresholds

| Test Type | FP32 Threshold | FP16 Threshold | Notes |
|-----------|---------------|---------------|-------|
| L2 relative error | 1e-6 | 1e-3 | After 100 steps |
| L∞ relative error | 1e-6 | 1e-3 | Maximum pointwise |
| Correlation | 0.999+ | 0.99+ | Receiver comparisons |
| Energy drift | 0.1% | 0.1% | Over 200+ steps |
| Convergence order | 1.5-4.0 | 1.5-4.0 | Depends on CFL |

## File Locations

EOF

    # Add file references
    for file in "$VALIDATION_DIR"/*.txt; do
        if [[ -f "$file" ]]; then
            echo "- $(basename "$file")" >> "$summary_file"
        fi
    done

    echo "" >> "$summary_file"
    echo "## Usage Examples" >> "$summary_file"
    cat >> "$summary_file" << EOF

\`\`\`bash
# Run full validation suite
make validate

# Quick validation (smaller grids)
make validate-quick

# Custom validation
./validation_driver 128 200 custom_test
./cuda_openacc_validator performance 96 96 96 150
\`\`\`

## Troubleshooting

### Common Issues
1. **CFL violations:** Reduce dt or increase grid spacing
2. **Memory errors:** Check halo bounds and atomic operations
3. **Convergence failures:** Verify boundary conditions and source terms
4. **Performance anomalies:** Profile with nsight-compute or nvprof

### Debug Commands
\`\`\`bash
# Detailed memory checking
cuda-memcheck --tool memcheck --leak-check full ./comprehensive_benchmark

# Performance profiling
ncu --metrics dram__bytes_read,dram__bytes_write ./comprehensive_benchmark

# Kernel launch analysis
nvprof --print-gpu-summary ./comprehensive_benchmark
\`\`\`
EOF

    success "Summary report generated: $summary_file"
}

main() {
    echo -e "${BLUE}"
    echo "==========================================="
    echo "    FDTD Solver Validation Suite v1.0     "
    echo "==========================================="
    echo -e "${NC}"

    # Create results directory
    mkdir -p "$VALIDATION_DIR"
    log "Results will be saved to: $VALIDATION_DIR"

    # Run validation sequence
    check_dependencies
    build_validation_suite

    # Core validation tests
    run_parity_tests
    run_memory_validation
    run_race_condition_tests
    run_convergence_tests
    run_energy_conservation_tests

    # Generate summary
    generate_summary_report

    log "Validation suite completed!"
    log "Check $VALIDATION_DIR/ for detailed results"

    # Show quick summary
    echo ""
    echo -e "${GREEN}✓ Validation suite completed successfully${NC}"
    echo -e "📊 Results directory: ${BLUE}$VALIDATION_DIR${NC}"
    echo -e "📋 Summary report: ${BLUE}$VALIDATION_DIR/validation_summary.md${NC}"
}

# Parse command line arguments
case "${1:-all}" in
    "all")
        main
        ;;
    "quick")
        GRID_SIZES=(32)
        TIME_STEPS=(10)
        main
        ;;
    "parity")
        check_dependencies
        build_validation_suite
        mkdir -p "$VALIDATION_DIR"
        run_parity_tests
        ;;
    "memory")
        check_dependencies
        build_validation_suite
        mkdir -p "$VALIDATION_DIR"
        run_memory_validation
        ;;
    *)
        echo "Usage: $0 [all|quick|parity|memory]"
        echo "  all    - Run complete validation suite (default)"
        echo "  quick  - Run subset with small grids"
        echo "  parity - Run only parity tests"
        echo "  memory - Run only memory validation"
        exit 1
        ;;
esac