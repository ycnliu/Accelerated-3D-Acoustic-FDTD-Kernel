// cuda_textbook.cu — Textbook tiled 3D stencil (PMPP Chapter 8 style)
// Shared-memory Y-Z plane tiling + register-plane sweep along X.
// Uses the same 4th-order stencil as the other kernels for fair comparison.

#define _POSIX_C_SOURCE 200809L
#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

// -------- timing helpers --------
#define START_SEC(S) struct timeval start_##S, end_##S; gettimeofday(&start_##S, nullptr);
#define STOP_SEC(S, T) do { \
  gettimeofday(&end_##S, nullptr); \
  (T)->S += (double)(end_##S.tv_sec - start_##S.tv_sec) + \
            (double)(end_##S.tv_usec - start_##S.tv_usec) / 1e6; \
} while(0)

// -------- ABI (must match main.cpp) --------
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
  double section0;
  double section1;
};

// -------- device helpers --------
__device__ __forceinline__
size_t idx_u(int t, int X, int Y, int Z, int nxp, int nyp, int nzp) {
  return ((size_t)t * nxp * nyp * nzp) + ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
}

__device__ __forceinline__
size_t idx_m(int X, int Y, int Z, int nyp, int nzp) {
  return ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
}

// 4th-order coefficients (same as cuda.cu)
__constant__ float c_m2 = -8.33333333e-2f;   // -1/12
__constant__ float c_m1 =  1.333333330f;     //  4/3
__constant__ float c_0  = -2.50f;            // -5/2

#define HALO 4
#define STENCIL_R 2      // radius for 4th-order stencil
#define WARMUP_STEPS 5

// -------- Tile sizes (configurable via -D at compile time) --------
#ifndef TB_OUT_TILE
#define TB_OUT_TILE 8    // Output tile size in each of Y, Z, and X sweep
#endif
#define TB_IN_TILE (TB_OUT_TILE + 2 * STENCIL_R)  // e.g. 8+4=12

// ============================================================
// Textbook tiled stencil kernel (PMPP-style)
//
// Each thread block covers a TB_IN_TILE x TB_IN_TILE tile in (Z, Y)
// and sweeps TB_OUT_TILE planes along X.
//
// - Current X-plane loaded into shared memory for Y/Z neighbor access
// - X-direction neighbors kept in per-thread registers (plane sweep)
// ============================================================
__global__ void textbook_tiled_stencil_kernel(
    const float* __restrict__ m,
    const float* __restrict__ u,
    float*       __restrict__ u_out,
    int nxp, int nyp, int nzp,
    int x_m, int y_m, int z_m,
    int x_M, int y_M, int z_M,
    int t0, int t1, int t2,
    float dt, float r1, float r2, float r3, float r4)
{
  // Block's starting X index in the compute domain
  const int iStart = x_m + blockIdx.z * TB_OUT_TILE;
  const int iEnd   = min(iStart + TB_OUT_TILE - 1, x_M);

  // This thread's Y, Z position in the compute domain (includes halo threads)
  const int j = y_m + blockIdx.y * TB_OUT_TILE + (int)threadIdx.y - STENCIL_R;
  const int k = z_m + blockIdx.x * TB_OUT_TILE + (int)threadIdx.x - STENCIL_R;

  // Padded coordinates (add HALO for ghost-cell offset in the array)
  const int J = j + HALO;
  const int K = k + HALO;

  // Strides in the flat array
  const size_t nper   = (size_t)nxp * nyp * nzp;
  const size_t stride_x = (size_t)nyp * nzp;
  const size_t stride_y = (size_t)nzp;
  const size_t o0 = (size_t)t0 * nper;
  const size_t o1 = (size_t)t1 * nper;
  const size_t o2 = (size_t)t2 * nper;

  // Does this thread map to a valid (Y,Z) within the padded array?
  const bool yz_valid = (J >= 0 && J < nyp && K >= 0 && K < nzp);

  // Is this thread in the interior of the tile (i.e. produces output)?
  const bool is_interior_thread =
      (threadIdx.y >= STENCIL_R && threadIdx.y < TB_IN_TILE - STENCIL_R &&
       threadIdx.x >= STENCIL_R && threadIdx.x < TB_IN_TILE - STENCIL_R);
  const bool is_output = is_interior_thread && yz_valid &&
      (j >= y_m && j <= y_M && k >= z_m && k <= z_M);

  // ---- Shared memory for the current X-plane ----
  __shared__ float curr_s[TB_IN_TILE][TB_IN_TILE];

  // Helper lambda: read u[t0] at padded (X, J, K), clamped to array bounds
  auto load = [&] __device__ (int Xpad) -> float {
    if (!yz_valid) return 0.0f;
    if (Xpad < 0 || Xpad >= nxp) return 0.0f;
    return u[o0 + (size_t)Xpad * stride_x + (size_t)J * stride_y + (size_t)K];
  };

  // ---- Pre-load register planes for the first sweep position ----
  const int X0 = iStart + HALO;          // padded X for iStart
  float pm2 = load(X0 - 2);              // plane at x-2
  float pm1 = load(X0 - 1);              // plane at x-1
  float pc  = load(X0);                  // current plane (also goes into smem)
  float pp1 = load(X0 + 1);              // plane at x+1

  // Load current plane into shared memory
  curr_s[threadIdx.y][threadIdx.x] = pc;
  __syncthreads();

  // ---- Sweep along X ----
  for (int i = iStart; i <= iEnd; ++i) {
    const int X = i + HALO;

    // Load the next plane at x+2 (register)
    float pp2 = load(X + 2);

    // Compute the 4th-order 3D Laplacian + leapfrog update
    if (is_output) {
      const float uc  = pc;
      const float um1 = u[o1 + (size_t)X * stride_x + (size_t)J * stride_y + (size_t)K];

      // X-direction: register planes
      const float d2dx2 = c_0 * uc
          + c_m2 * (pm2 + pp2)
          + c_m1 * (pm1 + pp1);

      // Y-direction: shared memory
      const float d2dy2 = c_0 * uc
          + c_m2 * (curr_s[threadIdx.y - 2][threadIdx.x] + curr_s[threadIdx.y + 2][threadIdx.x])
          + c_m1 * (curr_s[threadIdx.y - 1][threadIdx.x] + curr_s[threadIdx.y + 1][threadIdx.x]);

      // Z-direction: shared memory
      const float d2dz2 = c_0 * uc
          + c_m2 * (curr_s[threadIdx.y][threadIdx.x - 2] + curr_s[threadIdx.y][threadIdx.x + 2])
          + c_m1 * (curr_s[threadIdx.y][threadIdx.x - 1] + curr_s[threadIdx.y][threadIdx.x + 1]);

      const float mval = m[idx_m(X, J, K, nyp, nzp)];
      const float lap  = r2 * d2dx2 + r3 * d2dy2 + r4 * d2dz2;
      const float unew = 2.0f * uc - um1 + (dt * dt) * lap / mval;

      u_out[o2 + (size_t)X * stride_x + (size_t)J * stride_y + (size_t)K] = unew;
    }

    __syncthreads();

    // Shift register planes forward
    pm2 = pm1;
    pm1 = pc;
    pc  = pp1;
    pp1 = pp2;

    // Reload shared memory with the new current plane
    curr_s[threadIdx.y][threadIdx.x] = pc;
    __syncthreads();
  }
}

// ============================================================
// Source injection (identical to cuda.cu)
// ============================================================
__global__ void textbook_source_inject_kernel(
    const float* __restrict__ m,
    const float* __restrict__ src,
    const float* __restrict__ src_coords,
    float*       __restrict__ u,
    int nxp, int nyp, int nzp,
    int x_m, int x_M, int y_m, int y_M, int z_m, int z_M,
    int t2,
    float h_x, float h_y, float h_z,
    float o_x, float o_y, float o_z,
    int p_src_m, int p_src_M,
    int time,
    int pstride, int cstride)
{
  const int p_src = p_src_m + blockIdx.x * blockDim.x + threadIdx.x;
  if (p_src > p_src_M) return;

  const float sx = src_coords[p_src * cstride + 0];
  const float sy = src_coords[p_src * cstride + 1];
  const float sz = src_coords[p_src * cstride + 2];

  const float gx = (-o_x + sx) / h_x;
  const float gy = (-o_y + sy) / h_y;
  const float gz = (-o_z + sz) / h_z;

  const int posx = (int)floorf(gx);
  const int posy = (int)floorf(gy);
  const int posz = (int)floorf(gz);

  const float px = gx - floorf(gx);
  const float py = gy - floorf(gy);
  const float pz = gz - floorf(gz);

  const float m_base = m[idx_m(posx + HALO, posy + HALO, posz + HALO, nyp, nzp)];
  const float sval   = src[time * pstride + p_src];

  for (int rx = 0; rx <= 1; ++rx)
    for (int ry = 0; ry <= 1; ++ry)
      for (int rz = 0; rz <= 1; ++rz) {
        const int ix = rx + posx, iy = ry + posy, iz = rz + posz;
        if (ix < x_m-1 || iy < y_m-1 || iz < z_m-1 ||
            ix > x_M+1 || iy > y_M+1 || iz > z_M+1) continue;
        const float wx = rx ? px : (1.0f - px);
        const float wy = ry ? py : (1.0f - py);
        const float wz = rz ? pz : (1.0f - pz);
        const float w  = wx * wy * wz;
        const int X = ix + HALO, Y = iy + HALO, Z = iz + HALO;
        atomicAdd(&u[idx_u(t2, X, Y, Z, nxp, nyp, nzp)], 1.0e-2f * w * sval / m_base);
      }
}

// ============================================================
// Host wrapper — same ABI as Kernel_CUDA / Kernel_CUDA_Optimized
// ============================================================
extern "C" int Kernel_CUDA_Textbook(
    struct dataobj *__restrict m_vec,
    struct dataobj *__restrict src_vec,
    struct dataobj *__restrict src_coords_vec,
    struct dataobj *__restrict u_vec,
    const int x_M, const int x_m, const int y_M, const int y_m,
    const int z_M, const int z_m,
    const float dt, const float h_x, const float h_y, const float h_z,
    const float o_x, const float o_y, const float o_z,
    const int p_src_M, const int p_src_m, const int time_M, const int time_m,
    const int deviceid, const int devicerm, struct profiler *timers)
{
  if (deviceid != -1) cudaSetDevice(deviceid);

  // Clear any stale CUDA errors from previous kernel calls (e.g. optimized kernel's
  // cudaDeviceSetLimit for L2 persistence may fail on older architectures)
  cudaGetLastError();

  float (*__restrict u_h)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] =
      (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  const int nxp = u_vec->size[1];
  const int nyp = u_vec->size[2];
  const int nzp = u_vec->size[3];

  const size_t nU      = (size_t)u_vec->size[0] * nxp * nyp * nzp;
  const size_t nM      = (size_t)m_vec->size[0] * m_vec->size[1] * m_vec->size[2];
  const size_t nSrc    = (size_t)src_vec->size[0] * src_vec->size[1];
  const size_t nCoords = (size_t)src_coords_vec->size[0] * src_coords_vec->size[1];

  // Allocate device memory
  float *d_u = nullptr, *d_m = nullptr, *d_src = nullptr, *d_src_coords = nullptr;
  cudaMalloc(&d_u, nU * sizeof(float));
  cudaMalloc(&d_m, nM * sizeof(float));
  cudaMalloc(&d_src, nSrc * sizeof(float));
  cudaMalloc(&d_src_coords, nCoords * sizeof(float));

  // Copy to device
  cudaMemcpy(d_u, u_h, nU * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, m_vec->data, nM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src, src_vec->data, nSrc * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src_coords, src_coords_vec->data, nCoords * sizeof(float), cudaMemcpyHostToDevice);

  // Precomputed constants
  const float r1 = 1.0f / (dt * dt);
  const float r2 = 1.0f / (h_x * h_x);
  const float r3 = 1.0f / (h_y * h_y);
  const float r4 = 1.0f / (h_z * h_z);

  // Grid / block configuration
  const int ext_x = (x_M - x_m + 1);
  const int ext_y = (y_M - y_m + 1);
  const int ext_z = (z_M - z_m + 1);

  dim3 block(TB_IN_TILE, TB_IN_TILE, 1);
  dim3 grid(
      (ext_z + TB_OUT_TILE - 1) / TB_OUT_TILE,   // Z tiles
      (ext_y + TB_OUT_TILE - 1) / TB_OUT_TILE,   // Y tiles
      (ext_x + TB_OUT_TILE - 1) / TB_OUT_TILE    // X tiles (sweep chunks)
  );

  static bool first_run = true;
  if (first_run) {
    printf("[CUDA_Textbook] Grid=(%d,%d,%d) Block=(%d,%d,%d) OUT_TILE=%d IN_TILE=%d\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z, TB_OUT_TILE, TB_IN_TILE);
    printf("[CUDA_Textbook] Shared memory: %zu bytes\n",
           (size_t)TB_IN_TILE * TB_IN_TILE * sizeof(float));
    first_run = false;
  }

  // Warmup iterations
  for (int t = time_m; t < time_m + WARMUP_STEPS && t <= time_M; ++t) {
    const int t0 = (t) % 3;
    const int t1 = (t + 2) % 3;
    const int t2 = (t + 1) % 3;

    textbook_tiled_stencil_kernel<<<grid, block>>>(
        d_m, d_u, d_u,
        nxp, nyp, nzp,
        x_m, y_m, z_m, x_M, y_M, z_M,
        t0, t1, t2,
        dt, r1, r2, r3, r4);
    cudaDeviceSynchronize();

    // SECTION 1 DISABLED: Source injection commented out for pure stencil benchmark
    // if (p_src_M >= p_src_m) {
    //   const int nsrc = (p_src_M - p_src_m + 1);
    //   textbook_source_inject_kernel<<<(nsrc + 255) / 256, 256>>>(
    //       d_m, d_src, d_src_coords, d_u,
    //       nxp, nyp, nzp,
    //       x_m, x_M, y_m, y_M, z_m, z_M,
    //       t2, h_x, h_y, h_z, o_x, o_y, o_z,
    //       p_src_m, p_src_M, t,
    //       src_vec->size[1], src_coords_vec->size[1]);
    //   cudaDeviceSynchronize();
    // }
  }

  // Timed loop
  for (int t = time_m + WARMUP_STEPS; t <= time_M; ++t) {
    const int t0 = (t) % 3;
    const int t1 = (t + 2) % 3;
    const int t2 = (t + 1) % 3;

    START_SEC(section0)
    textbook_tiled_stencil_kernel<<<grid, block>>>(
        d_m, d_u, d_u,
        nxp, nyp, nzp,
        x_m, y_m, z_m, x_M, y_M, z_M,
        t0, t1, t2,
        dt, r1, r2, r3, r4);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "[CUDA_Textbook] kernel launch error: %s\n", cudaGetErrorString(err));
      cudaFree(d_u); cudaFree(d_m); cudaFree(d_src); cudaFree(d_src_coords);
      return err;
    }
    cudaDeviceSynchronize();
    STOP_SEC(section0, timers);

    // SECTION 1 DISABLED: Source injection commented out for pure stencil benchmark
    // if (p_src_M >= p_src_m) {
    //   const int nsrc = (p_src_M - p_src_m + 1);
    //   START_SEC(section1)
    //   textbook_source_inject_kernel<<<(nsrc + 255) / 256, 256>>>(
    //       d_m, d_src, d_src_coords, d_u,
    //       nxp, nyp, nzp,
    //       x_m, x_M, y_m, y_M, z_m, z_M,
    //       t2, h_x, h_y, h_z, o_x, o_y, o_z,
    //       p_src_m, p_src_M, t,
    //       src_vec->size[1], src_coords_vec->size[1]);
    //   err = cudaGetLastError();
    //   if (err != cudaSuccess) {
    //     cudaFree(d_u); cudaFree(d_m); cudaFree(d_src); cudaFree(d_src_coords);
    //     return err;
    //   }
    //   cudaDeviceSynchronize();
    //   STOP_SEC(section1, timers);
    // }
  }

  // Copy back to host
  cudaMemcpy(u_h, d_u, nU * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_u); cudaFree(d_m); cudaFree(d_src); cudaFree(d_src_coords);
  (void)devicerm;
  return 0;
}
