# FDTD Optimized Kernel - Stability Fix Log
**Date:** 2025-10-07
**Commit:** 42d0be7
**Status:** ✅ RESOLVED - Production Ready

---

## Critical Bug Summary

### Symptoms
Numerical instability on small grids causing field value explosion:
- **32³ grid**: Max field = 570-575 (should be 0.1) → **5750× error**
- **64³ grid**: Max field = 38-39 → **390× error**
- **128³ grid**: Max field = 38-39 → **390× error**
- **256³ grid**: Max field = 29-35 → **350× error**
- **512³ grid**: Max field = 0.1 ✓ (only stable size)

**Non-deterministic**: Values varied between runs, indicating race condition.

---

## Root Cause Analysis

### 1. **Incorrect Halo Addressing** (Primary Bug)
**Location:** `load_plane` lambda, lines 111-127 (original)

**Problem:**
```cuda
// WRONG: Used per-thread gz/gy coordinates
if (threadIdx.x < R) {
  P[sy * pitchZ + (R + TZ + xh)] = __ldg(u_t0_f32 + (c + (xh + 1)));
  // c = gIndex(Xpad) uses Zpad = gz + HALO (varies per thread!)
}
```

Each thread's `gz` varies: thread 0 has `gz=z_m`, thread 1 has `gz=z_m+1`, etc.
→ Right halo column gets filled with data from **different Z slices** instead of block-wide Z+TZ
→ Neighboring threads read wrong data → Laplacian corrupted → instability

**Impact:** Worst on small grids where blocks have more out-of-bounds threads.

### 2. **Early Thread Return** (Amplifier Bug)
**Location:** Line 76 (original)

**Problem:**
```cuda
if (gz > z_M || gy > y_M) return;  // ❌ Exits before cooperative loading
```

Out-of-bounds threads exit early, but they're needed for:
- Participating in `__syncthreads()`
- The first R threads (threadIdx.x < R) load halos for the **entire block**

On small grids / tail blocks:
- Many threads have `gz > z_M`
- If halo-loader thread exits → halos uninitialized → NaN/garbage → explosion

### 3. **Over-Guarded Domain Checks** (Prevention of Valid Reads)
**Location:** Lines 103-127 (original)

**Problem:**
```cuda
if (threadIdx.x < R && gz - R + threadIdx.x >= z_m && gz <= z_M) {
  // Only load if within physical domain
}
```

Arrays are **padded** with `HALO=4` ghost cells. These domain checks:
- Prevented loading valid ghost cells near boundaries
- Left holes in shared memory
- Ghost cells exist precisely for boundary halos!

---

## Solution Implementation

### Fix 1: Block-Base Halo Coordinates ✅
**Lines 84-85:**
```cuda
const int Zbase = Zpad - threadIdx.x;  // Z of threadIdx.x=0
const int Ybase = Ypad - threadIdx.y;  // Y of threadIdx.y=0
```

**Lines 112-115:**
```cuda
const int zL = Zbase + threadIdx.x;           // left: [Zbase+0 .. Zbase+R-1]
const int zR = Zbase + TZ + R + threadIdx.x;  // right: [Zbase+TZ+R .. Zbase+TZ+2R-1]
P[sy*pitchZ + threadIdx.x] = __ldg(u_t0_f32 + row_base + zL);
P[sy*pitchZ + (R+TZ+threadIdx.x)] = __ldg(u_t0_f32 + row_base + zR);
```

**Result:** All threads in block now load consistent Z-slices for halos.

### Fix 2: Remove Early Return, Mask Output ✅
**Lines 77-78:**
```cuda
const bool active = (gz >= z_m && gz <= z_M && gy >= y_m && gy <= y_M);
// No return - all threads participate in loading
```

**Lines 172-175, 212-215:**
```cuda
if (active) {
  u_t2_f32[out_idx] = unew;  // Only write if in-bounds
}
```

**Result:** Out-of-bounds threads still load halos, participate in syncs, but don't corrupt output.

### Fix 3: Trust Padding, Remove Domain Guards ✅
**Lines 111-127:** Removed all `gz >= z_m`, `gy >= y_m` checks from `load_plane`

**Result:** Halos load ghost cells freely; padding ensures no out-of-bounds access.

---

## Verification Results

### Stability Test (15 runs across all sizes)
```
Grid    Max Field   Status
────────────────────────────
32³     0.1         ✅ STABLE
64³     0.1         ✅ STABLE
128³    0.1         ✅ STABLE
256³    0.1         ✅ STABLE
512³    0.1         ✅ STABLE
```

**Deterministic:** No variation across runs.

### Accuracy (vs OpenACC reference)
```
Grid    Max Rel Error   L2 Norm Error   Status
────────────────────────────────────────────────
32³     6.94e-4 (0.07%) 8.36e-5 (0.008%) ✅ PASS
64³     7.04e-4 (0.07%) 5.95e-5 (0.006%) ✅ PASS
128³    7.01e-4 (0.07%) 4.20e-5 (0.004%) ✅ PASS
256³    6.95e-4 (0.07%) 2.82e-5 (0.003%) ✅ PASS
512³    6.98e-4 (0.07%) 2.18e-5 (0.002%) ✅ PASS
```

**Consistent:** Error decreases with grid size (as expected for FP32).

### Performance (H100 80GB)
```
Grid    GFLOP/s  Memory BW   vs OpenACC   vs Plain CUDA
──────────────────────────────────────────────────────────
32³     29.2     0.3%        -48%         -52%
64³     126      1.3%        -69%         -66%
128³    616      6.1%        -61%         -35%
256³    2588     25.7%       -1.6%        +205%
512³    2842     28.3%       -1.1%        +290%  ✓ TARGET
```

**Analysis:**
- Small grids: Overhead-dominated (sync costs)
- Large grids: Near-optimal, competitive with OpenACC
- Plain CUDA: Much slower (no temporal blocking)

---

## Technical Insights

### Why Small Grids Failed
1. **XCHUNK=64** is larger than domain (32)
   - Many threads per block have `gz > z_M` or `gy > y_M`
   - With early return → no halo loading → explosion

2. **Higher sync-to-compute ratio**
   - 2× `__syncthreads()` per iteration
   - Small grids have fewer iterations to amortize sync cost

3. **More partial blocks**
   - Grid edges have incomplete tiles
   - Incorrect halo logic hits boundaries more often

### Why 512³ Was Stable
- XCHUNK=64 << domain (512)
- Most blocks fully interior (no out-of-bounds threads)
- Halo bugs only manifest at grid boundaries
- Enough iterations to hide occasional corrupted halos in noise

### Key Lesson
**Cooperative kernel launches require ALL threads to participate in collective operations**, even if they don't compute valid output. Guarding shared memory loads with domain checks breaks the cooperative model.

---

## Performance Trade-offs

### Before Fix (Unstable but Fast)
- Single `__syncthreads()` outside unrolled loop
- Minimal synchronization overhead
- 3150 GFLOP/s on 512³ (best case, when stable)

### After Fix (Stable, Slightly Slower)
- Two `__syncthreads()` per iteration (before/after load)
- Extra checks: `if (active)` for writes
- 2842 GFLOP/s on 512³ (-10% from unsafe peak)

**Verdict:** Acceptable. Correctness > 10% speed. Still matches OpenACC.

---

## Comparison with Reference Implementations

### Plain CUDA (stable baseline)
- **Accuracy:** 0.004% relative error (10× better than optimized)
- **Performance:** 728 GFLOP/s (4× slower)
- **Method:** Direct stencil, no temporal blocking
- **Issue:** Reloads all data every timestep

### OpenACC (compiler-optimized)
- **Accuracy:** Reference (FP64)
- **Performance:** 2873 GFLOP/s
- **Method:** Compiler-generated optimizations
- **Advantage:** 152% memory BW efficiency (compression?)

### CUDA_Optimized (our kernel)
- **Accuracy:** 0.07% relative error (acceptable)
- **Performance:** 2842 GFLOP/s (-1.1% vs OpenACC)
- **Method:** Manual temporal blocking with ring buffer
- **Advantage:** Explicit control, portable to other stencils

---

## Remaining Optimization Opportunities

### 1. Reduce Synchronization Overhead
**Current:** 2× sync per iteration (conservative)

**Possible:** Use double-buffering technique:
```cuda
load_plane((cur + 2*R) % ring_size, next_plane);  // Safe buffer
// No sync needed if cur advances properly
```

**Risk:** Complex indexing, potential for new bugs.

### 2. Increase Temporal Blocking Depth
**Current:** UNROLL_FACTOR=4

**Try:** 8, 16 to amortize loads further

**Risk:** Shared memory pressure, may reduce occupancy.

### 3. Vectorized Loads (float4)
**Current:** Scalar `__ldg(float*)`

**Try:**
```cuda
float4* src4 = (float4*)u_t0_f32;
float4 val = __ldg(&src4[idx/4]);
```

**Benefit:** 4× fewer load instructions.

### 4. Async Pipeline (H100 Feature)
```cuda
__pipeline_memcpy_async(&smem[...], &gmem[...], sizeof(plane));
__pipeline_commit();
__pipeline_wait_prior(0);
```

**Benefit:** Overlap compute with prefetch.

**Complexity:** High - requires careful pipeline depth tuning.

---

## Production Deployment Notes

### When to Use This Kernel
✅ Large 3D grids (128³+)
✅ Memory-bound stencils (low arithmetic intensity)
✅ Repeated timesteps (amortizes ring buffer overhead)
✅ Acceptable 0.07% relative error (most geophysics/imaging)

### When NOT to Use
❌ Small grids (<128³) - overhead-dominated, use plain CUDA
❌ Requires <0.01% error - use FP64 OpenACC instead
❌ Single timestep - no benefit from temporal blocking

### Recommended Settings
- **TZ=64, TY=16:** Good balance for H100
- **XCHUNK=64:** Matches L1 cache line behavior
- **UNROLL_FACTOR=4:** Conservative, stable
- **USE_FP32_ONLY=1:** Current stable configuration

### Testing Checklist Before Production
1. Run `make run` - verify all correctness tests pass
2. Check `Max field value: 0.1` for all grid sizes
3. Verify L2 norm error < 1e-4 across all tests
4. Profile with `nsys` to check for unexpected stalls
5. Test on target GPU architecture (rebuild with correct sm_XX)

---

## Files Modified

### cuda_optimized.cu
- **Lines 77-85:** Added `active` mask, block-base coordinates
- **Lines 96-128:** Rewrote `load_plane` with correct halo addressing
- **Lines 172-175, 212-215:** Guarded output writes with `if (active)`

### main.cpp
- **Line 536:** Increased test field values (×10000) to expose relative errors

---

## Acknowledgments

**Bug diagnosed by:** User insight on "blows up on small grids"

**Root cause analysis:** Identified per-thread vs block-base coordinate bug

**Fix design:** Three-part solution (block-base + no early return + trust padding)

**Verification:** 15 runs across 5 grid sizes, all stable ✅

---

## Commit History
```
42d0be7 - Fix critical stability bug in FDTD optimized kernel (HEAD)
ef6fe24 - Apply expert's halo-loading pattern and optimize FDTD kernel
c558263 - Implement FP16 storage + FP32 compute with shadow buffers
```

**GitHub:** https://github.com/ycnliu/Accelerated-3D-Acoustic-FDTD-Kernel

---

## Status: RESOLVED ✅
**Kernel is production-ready for large-scale 3D FDTD simulations.**

