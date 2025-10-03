# FDTD Unified Benchmark - OpenACC vs CUDA Performance Comparison

A comprehensive benchmarking suite for comparing OpenACC and CUDA implementations of 3D Finite-Difference Time-Domain (FDTD) acoustic wave simulation with automated correctness verification and performance measurement.

## Overview

This repository contains three optimized implementations of a 3D acoustic FDTD solver:
- **OpenACC**: High-level directive-based GPU programming
- **CUDA**: Hand-optimized CUDA kernels
- **CUDA_Optimized**: CUDA with shared memory tiling and loop unrolling

## Quick Start

```bash
# Build for RTX 2080 Ti
make clean && GPU_ARCH=sm_75 make -j

# Run unified benchmark (correctness + performance)
make run

# View results
make show-results
```

## Unified Benchmark Workflow

The benchmark automatically runs in three phases:

### Phase 1: Correctness Verification
- Tests all implementations against OpenACC reference
- Grid sizes: 32³, 64³, 128³, 256³, 512³
- 50 timesteps per test
- Pass criteria: L2 error < 1e-4

### Phase 2: Performance Benchmark
- Runs all three implementations
- Grid sizes: 32³, 64³, 128³, 256³, 512³
- 50 timesteps, 1 source
- 5 repetitions with statistics
- Writes detailed CSV output

### Phase 3: Results Summary
- Displays benchmark.csv data
- Shows timing breakdown and efficiency metrics

## Repository Structure

```
.
├── main.cpp              # Unified benchmark driver with correctness tests
├── openacc.cpp          # OpenACC implementation
├── cuda.cu              # Regular CUDA implementation
├── cuda_optimized.cu    # Optimized CUDA with shared memory + unrolling
├── Makefile             # Simplified build system
├── README.md            # This file
├── DEBUG.md             # Development history and debugging notes
└── H100_README.md       # H100 optimization notes
```

## Build Requirements

- **NVIDIA HPC SDK** (for OpenACC): `nvc++` with `-acc` flag
- **CUDA Toolkit**: `nvcc` with compute capability 7.5+ (sm_75 for RTX 2080 Ti)
- **GNU Make**
- **CUDA-capable GPU**

## Building

```bash
# RTX 2080 Ti (Turing) - default
make clean && make -j

# H100 (Hopper)
make clean && GPU_ARCH=sm_90 make -j

# A100 (Ampere)
make clean && GPU_ARCH=sm_80 make -j
```

## Running

```bash
# Complete benchmark (correctness + performance)
make run

# View CSV results
make show-results

# Clean everything
make clean

# Show help
make help
```

## Benchmark Output

### Console Output
```
STEP 1: CORRECTNESS VERIFICATION
- Tests 5 grid sizes (32³ to 512³)
- Compares CUDA vs OpenACC
- Compares CUDA_Optimized vs OpenACC
- Reports max absolute/relative error, L2 norm

STEP 2: PERFORMANCE BENCHMARK
- Runs all implementations
- Shows timing breakdown per grid size
- Computes GFLOP/s and GB/s metrics
- Displays GPU efficiency percentages

STEP 3: RESULTS SUMMARY
- CSV table with all metrics
- Ready for analysis/plotting
```

### CSV Schema (benchmark.csv)
```
Method,Total_Time(ms),Total_Std(ms),
Section0_Time(ms),Section0_Std(ms),    # Main stencil compute
Section1_Time(ms),Section1_Std(ms),    # Source injection
Device_Time(ms),Device_Std(ms),        # Total GPU time
Overhead(ms),Overhead_Std(ms),         # Host overhead
GFLOPS,GFLOPS_Std,GBps,GBps_Std,
Compute_Eff(%),Memory_Eff(%),          # vs GPU peak specs
AI,NX,NY,NZ,Timesteps,Sources,StencilOrder
```

## Implementation Details

### FDTD Algorithm
- **4th-order finite difference** spatial discretization (radius-2 stencil)
- **2nd-order leapfrog** time integration
- **3D acoustic wave equation**: ∂²u/∂t² = v²∇²u
- **Ricker wavelet source** injection with trilinear interpolation
- **Halo padding**: 4 cells per side for boundary conditions

### Optimizations

#### OpenACC
- `#pragma acc parallel loop collapse(3)`
- Compiler-managed data movement
- Automatic kernel fusion

#### CUDA
- Coalesced global memory access
- 3D thread block tiling

#### CUDA_Optimized
- **Shared memory tiling**: 8×8×8 blocks with 12×12×12 tiles (radius-2 halos)
- **Constant memory**: Stencil coefficients (-1/12, 4/3, -5/2)
- **Loop unrolling**: `#pragma unroll` on halo loads and stencil ops
- **Mixed precision ready**: Infrastructure for FP16 storage (disabled)

### Performance Metrics
- **FLOPs per point**: 3×(4+1)×2 + 6 = 36 (4th-order 3D Laplacian + leapfrog)
- **Bytes per point**: ~64 (naive) or ~12 (optimized with cache reuse)
- **Arithmetic Intensity**: 0.56 - 3.0 FLOPs/byte (memory-bound)

## Recent Updates

### Unified Benchmark (Latest)
- Single executable runs correctness + performance tests
- Simplified Makefile (just `make run`)
- All timing in milliseconds (CSV and console)
- Automatic GPU detection and peak specs

### CUDA_Optimized Enhancements
- Added `#pragma unroll` to stencil computation and halo loading
- Fixed halo loading to include both ±1 and ±2 neighbors
- Expected 5-15% performance improvement

## GPU Compatibility

| GPU | Compute Capability | GPU_ARCH | Status |
|-----|-------------------|----------|--------|
| RTX 2080 Ti | 7.5 (Turing) | sm_75 | ✅ Tested |
| A100 | 8.0 (Ampere) | sm_80 | ✅ Compatible |
| H100 | 9.0 (Hopper) | sm_90 | ✅ Compatible |

## Performance Notes

- OpenACC performance is highly compiler-dependent (NVHPC 23.11+)
- RTX 2080 Ti: FP16 Tensor Cores available but limited benefit for memory-bound workloads
- Peak bandwidth: ~616 GB/s (2080 Ti), ~2000 GB/s (A100), ~3350 GB/s (H100)
- Shared memory tiling benefits increase with larger L1 cache (A100+)

## License

This project is provided as-is for research and educational purposes.

## References

- DEBUG.md: Detailed development history and bug fixes
- H100_README.md: H100-specific optimizations and pipelining
