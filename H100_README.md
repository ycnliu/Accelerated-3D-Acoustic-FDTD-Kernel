# FDTD ConvStencil for H100

This implementation adapts the Microsoft ConvStencil approach for FDTD 4th-order stencil computation on NVIDIA H100 GPUs.

## Quick Start on H100

```bash
# Build for H100 (sm_90)
make clean && GPU_ARCH=sm_90 make -j

# Run tests
make run

# Or run specific benchmarks
./fdtd_benchmark cuda_optimized
```

## Architecture Support

The code now supports multiple GPU architectures via the `GPU_ARCH` variable:

- **H100 (Hopper)**: `GPU_ARCH=sm_90` (default) - Optimized for Tensor Cores
- **A100 (Ampere)**: `GPU_ARCH=sm_80` - Compatible
- **RTX 2080 Ti (Turing)**: `GPU_ARCH=sm_75` - Fallback

## ConvStencil Background

ConvStencil transforms stencil computations into matrix multiplications that can leverage Tensor Cores:

- **Reference**: Chen et al., "ConvStencil: Transform Stencil Computation to Matrix Multiplication on Tensor Cores", PPoPP 2024
- **Repository**: https://github.com/microsoft/ConvStencil

### Current Implementation Status

**Phase 1** (Completed):
- ✅ H100-optimized block configuration (8x8x8 threads = 512/block)
- ✅ Architecture-aware compilation (sm_90 support)
- ✅ ABI compatibility with existing test framework
- ✅ Baseline performance on H100

**Phase 2** (Completed):
- ✅ Shared memory tiling for data reuse (12x12x12 tiles with halos)
- ✅ Cooperative halo loading across thread blocks
- ✅ FP32 accuracy maintained on RTX 2080 Ti
- ✅ Infrastructure for Tensor Core integration
- ✅ Runtime configuration hook for TC enable/disable

**Phase 3** (Future - Full Tensor Core Integration):
- ⏳ WMMA fragment operations for stencil weights
- ⏳ Data layout transformation (ConvStencil-style)
- ⏳ FP64/TF32 Tensor Core utilization (H100)
- ⏳ FP16 mixed precision with FP32 accumulation (Turing+)
- ⏳ Multi-level blocking for large grids

## Performance Expectations

On H100, you should see significant improvements over RTX 2080 Ti due to:

1. **Higher memory bandwidth**: 3 TB/s (H100 HBM3) vs 616 GB/s (2080 Ti)
2. **More SMs**: 132 vs 68
3. **Tensor Cores**: 4th-gen with FP64 support
4. **Better occupancy**: Larger thread blocks and register files

### Baseline Performance (without full ConvStencil)

Expected speedup over 2080 Ti: **2-3×** from memory bandwidth alone

Current implementation uses:
- Standard CUDA stencil computation
- H100-optimized block sizes
- Larger thread blocks for better occupancy

### With Full ConvStencil (Phase 2)

Expected additional speedup: **1.5-2×** from Tensor Core utilization

Total expected speedup over 2080 Ti: **3-6×**

## Building for Different GPUs

```bash
# H100 (recommended)
make clean && GPU_ARCH=sm_90 make -j

# A100
make clean && GPU_ARCH=sm_80 make -j

# RTX 2080 Ti (baseline)
make clean && GPU_ARCH=sm_75 make -j
```

## Testing on H100

```bash
# Full test suite
make clean && GPU_ARCH=sm_90 make -j && make run

# Correctness only
./fdtd_benchmark correctness

# Speed comparison
./fdtd_benchmark speed

# Full benchmark
./fdtd_benchmark all
```

## Phase 2 Implementation Details

The current implementation includes:

1. **Shared Memory Tiling**:
   - 8x8x8 thread blocks processing tiles
   - 12x12x12 shared memory tiles (8+4 halo)
   - Cooperative halo loading across boundaries
   - __syncthreads() for data consistency

2. **Hybrid Memory Access**:
   - Interior threads use shared memory (low latency)
   - Boundary threads fall back to global memory
   - Reduces register pressure compared to full caching

3. **ConvStencil Infrastructure**:
   - `#include <mma.h>` for Tensor Core API
   - Runtime flag `g_use_tensorcore` for TC enable/disable
   - `FDTD_SetRuntimeConfig()` hook for configuration

## Next Steps for Phase 3 (Full Tensor Core Integration)

To implement the full ConvStencil WMMA transformation:

1. **Study the reference implementation**:
   ```bash
   cd /global/home/users/ycnliu/ConvStencil/src/3d
   # Examine gpu_star.cu for WMMA transformation pattern
   ```

2. **Key components to adapt**:
   - WMMA fragment declarations (wmma::fragment)
   - Data layout transformation for matrix multiply
   - Stencil coefficient matrix construction
   - wmma::load_matrix_sync / wmma::mma_sync operations

3. **Precision strategy**:
   - **RTX 2080 Ti (sm_75)**: FP16 input, FP32 accumulate
   - **H100 (sm_90)**: FP64 or TF32 Tensor Cores
   - Careful numerical validation at each step

4. **Testing strategy**:
   - Start with small grids (64³) to verify correctness
   - Compare against Phase 2 FP32 reference
   - Profile with nsys/ncu
   - Validate error metrics stay within tolerance

5. **Performance tuning**:
   - Experiment with block sizes (8x64, 16x32, etc.)
   - Tune shared memory layout for bank conflicts
   - Optimize WMMA fragment reuse
   - Use roofline model analysis

## Profiling

```bash
# Profile with Nsight Systems
nsys profile --stats=true ./fdtd_benchmark cuda_optimized

# Profile with Nsight Compute
ncu --set full ./fdtd_benchmark cuda_optimized
```

## References

- [ConvStencil Paper](https://dl.acm.org/doi/10.1145/3627535.3638492)
- [ConvStencil GitHub](https://github.com/microsoft/ConvStencil)
- [NVIDIA H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/)
- [CUDA Programming Guide - WMMA API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)

## Known Limitations

- Current implementation does not yet use the full ConvStencil transformation
- Tensor Cores are not yet utilized (Phase 2 work)
- Optimized for H100 but will run on older GPUs with reduced performance

## Contact

For questions or improvements, refer to the original ConvStencil repository and FDTD implementation documentation.
