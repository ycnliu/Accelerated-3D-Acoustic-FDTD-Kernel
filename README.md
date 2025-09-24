# FDTD Benchmark Suite - OpenACC vs CUDA Performance Comparison

A comprehensive benchmarking suite for comparing OpenACC and CUDA implementations of 3D Finite-Difference Time-Domain (FDTD) acoustic wave simulation.

## Overview

This repository contains three optimized implementations of a 3D acoustic FDTD solver:
- **OpenACC**: High-level directive-based GPU programming
- **CUDA Regular**: Hand-optimized CUDA kernels
- **CUDA Optimized**: CUDA with shared memory tiling optimizations

## Performance Results

### Complete Performance Comparison (256³ grid, 100 timesteps)

| Implementation | Total Time | Section0 (Compute) | Section1 (Source) | GFLOPS | GB/s |
|---------------|------------|-------------------|-------------------|--------|------|
| **CUDA Regular** | 0.170s | **0.052s** | 0.001s | **1138** | **2023** |
| **CUDA Optimized** | 0.171s | 0.052s | 0.001s | **1135** | **2018** |
| **OpenACC** | 0.174s | 0.056s | 0.001s | **1065** | **1893** |

### Key Findings
- **CUDA Regular** provides the best performance across all grid sizes
- **OpenACC** achieves 93% of CUDA performance with significantly less code complexity
- **CUDA Optimized** shows minimal improvement over regular CUDA for this workload
- **Section 0 (main compute)** dominates performance; Section 1 (source injection) is negligible

## Repository Structure

```
.
├── main.cpp              # Unified benchmark driver
├── openacc.cpp          # OpenACC implementation
├── cuda.cu              # Regular CUDA implementation
├── cuda_optimized.cu    # Optimized CUDA with shared memory
├── Makefile             # Build system for all implementations
└── README.md            # This file
```

## Build Requirements

- **NVIDIA HPC SDK** (for OpenACC): `nvc++` with `-acc` flag
- **CUDA Toolkit**: `nvcc` with compute capability 7.5+
- **GNU Make**
- **CUDA-capable GPU**

## Building and Running

### Build All Implementations
```bash
make all-impl          # Build all three executables
```

### Run Individual Benchmarks
```bash
make run-openacc-only   # OpenACC only
make run-cuda-only      # CUDA regular only
make run-cuda-opt-only  # CUDA optimized only
```

### Run Complete Comparison
```bash
make run-all-impl       # All three implementations
```

### Available Make Targets
- `all` - Build unified benchmark (OpenACC default)
- `all-impl` - Build all individual implementations
- `run-all-impl` - Run complete three-way comparison
- `clean` - Remove all generated files
- `show-results` - Display CSV benchmark results
- `help` - Show all available targets

## Benchmark Output

Results are saved to `benchmark.csv` with detailed timing breakdowns:
- **Total_Time**: Wall-clock time including all overhead
- **Section0_Time**: Main FDTD compute kernel time
- **Section1_Time**: Source injection kernel time
- **Device_Time**: Combined GPU kernel time
- **Overhead**: Memory transfers + kernel launch overhead
- **GFLOPS/GBps**: Performance metrics

## Implementation Details

### FDTD Algorithm
- **4th-order finite difference** spatial discretization
- **2nd-order leapfrog** time integration
- **3D acoustic wave equation** with variable velocity
- **Ricker wavelet source** injection
- **Perfectly matched layer** boundary conditions (implicit in padding)

### Grid Sizes Tested
- 64³, 128³, 256³, 512³, 768³ (limited by GPU memory)
- 100 timesteps per benchmark
- 8-point halo padding for boundary conditions

### Performance Metrics
- **36 FLOPs per grid point** per timestep (4th-order 3D Laplacian + leapfrog)
- **64 bytes I/O per grid point** per timestep (memory bandwidth model)
- **Device-only timing** excludes memory transfer overhead

## Code Highlights

### OpenACC Implementation
```cpp
#pragma acc parallel loop collapse(3) present(m,u)
for (int x = x_m; x <= x_M; x += 1) {
    for (int y = y_m; y <= y_M; y += 1) {
        for (int z = z_m; z <= z_M; z += 1) {
            // 4th-order stencil computation
        }
    }
}
```

### CUDA Implementation
```cpp
__global__ void stencil_update_kernel(
    const float* __restrict__ m,
    const float* __restrict__ u,
    float* __restrict__ u_out,
    // ... parameters
) {
    const int gx = x_m + blockIdx.x * blockDim.x + threadIdx.x;
    // ... CUDA kernel implementation
}
```

## License

This project is provided as-is for research and educational purposes.

## Performance Notes

- Results obtained on NVIDIA Tesla GPU (Compute Capability 7.5)
- Compiled with `-O3` optimization
- OpenACC performance is highly compiler-dependent
- CUDA optimizations may vary by architecture

## Contributing

Feel free to submit improvements, additional implementations, or performance optimizations via pull requests.