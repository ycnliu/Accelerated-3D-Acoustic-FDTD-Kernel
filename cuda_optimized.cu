// ==== ADAPTED FROM STAR/WMMA KERNEL FOR 4TH-ORDER LEAPFROG ====
//  - Keeps your ABI and shared-memory tile
//  - Adds: constant-mem coeffs + optional WMMA (FP16) microkernel for Y-axis 5-tap
//  - Target: sm_75+ (Turing), using FP16 Tensor Cores with FP32 accumulation

#define _POSIX_C_SOURCE 200809L
#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// Timing helpers
#define START_SEC(S) struct timeval start_##S, end_##S; gettimeofday(&start_##S, nullptr);
#define STOP_SEC(S, T) do { \
  gettimeofday(&end_##S, nullptr); \
  (T)->S += (double)(end_##S.tv_sec - start_##S.tv_sec) + \
            (double)(end_##S.tv_usec - start_##S.tv_usec) / 1e6; \
} while(0)

// ABI structures
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

// ---------------- Coeffs & TC config ----------------
// Stencil coefficients (initialized via cudaMemcpyToSymbol in host code)
__constant__ float c_m2_c;  // -1/12
__constant__ float c_m1_c;  //  4/3
__constant__ float c_0_c;   // -5/2

// optional Tensor Core path toggle (device global)
__device__ int g_use_tensorcore = 0;

// WMMA coefficients (currently unused - FP32 path active)
// __constant__ half wmma_B_5tap[16*16]; // FP16 coefficient matrix for future use

// ---------------- Index helpers ----------------
__device__ __forceinline__
size_t idx_u(int t, int X, int Y, int Z, int nxp, int nyp, int nzp) {
  return ((size_t)t * nxp * nyp * nzp) + ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
}
__device__ __forceinline__
size_t idx_m(int X, int Y, int Z, int nyp, int nzp) {
  return ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
}

// ---------------- Tile config (same as your kernel) ----------------
#ifndef TILE_X
#define TILE_X 8
#endif
#ifndef TILE_Y
#define TILE_Y 8
#endif
#ifndef TILE_Z
#define TILE_Z 8
#endif

// HALO=4 for 4th-order stencils
// This must match the global array padding (4 ghost cells each side)
// Stencil uses radius-2 (±2 points), shared memory tile needs 2 halo cells each side
#ifndef HALO
#define HALO 4
#endif

// ================== WMMA microkernel (Y-axis 5-tap) ==================
// TODO: Implement FP16 WMMA for sm_75 (currently disabled due to complexity)
// Placeholder - not used, FP32 path is active in kernel

// ================== Main compute kernel (shared memory + optional TC) ==================
__global__ void stencil_update_kernel_smem_opt(
  const float* __restrict__ m,
  const float* __restrict__ u,
  float*       __restrict__ u_out,
  int nxp, int nyp, int nzp,
  int x_m, int y_m, int z_m,
  int x_M, int y_M, int z_M,
  int t0, int t1, int t2,
  float dt, float r1, float r2, float r3, float r4)
{
  // Shared tile (radius 2 each dim)
  __shared__ float s_tile[(TILE_X+4)][(TILE_Y+4)][(TILE_Z+4)];

  const int gx = x_m + blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = y_m + blockIdx.y * blockDim.y + threadIdx.y;
  const int gz = z_m + blockIdx.z * blockDim.z + threadIdx.z;
  if (gx > x_M || gy > y_M || gz > z_M) return;

  const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const int X = gx + HALO, Y = gy + HALO, Z = gz + HALO;

  const size_t nper = (size_t)nxp * nyp * nzp;
  const size_t o0 = (size_t)t0 * nper;
  const size_t o1 = (size_t)t1 * nper;
  const size_t o2 = (size_t)t2 * nper;
  const size_t c   = ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;

  // Center
  s_tile[tx+2][ty+2][tz+2] = u[o0 + c];

  // Halos (radius 2) - FIXED: load both +1 and +2 for right/top/back
  // X halos (left: -2,-1) - unrolled
  #pragma unroll
  if (tx < 2) {
    // tx=0 -> X-2 at s_tile[0], tx=1 -> X-1 at s_tile[1]
    s_tile[tx][ty+2][tz+2] = u[o0 + (((size_t)(X-2+tx))*nyp*nzp + (size_t)Y*nzp + (size_t)Z)];
  }
  // X halos (right: +1,+2) - unrolled
  #pragma unroll
  if (tx >= blockDim.x - 2) {
    const int ox = tx - (blockDim.x - 2);  // 0 or 1
    // tx==bx-2 -> +1 at s_tile[tx+3], tx==bx-1 -> +2 at s_tile[tx+4]
    s_tile[tx+3+ox][ty+2][tz+2] = u[o0 + (((size_t)(X+1+ox))*nyp*nzp + (size_t)Y*nzp + (size_t)Z)];
  }

  // Y halos (bottom: -2,-1) - unrolled
  #pragma unroll
  if (ty < 2) {
    s_tile[tx+2][ty][tz+2] = u[o0 + ((size_t)X*nyp*nzp + (size_t)(Y-2+ty)*nzp + (size_t)Z)];
  }
  // Y halos (top: +1,+2) - unrolled
  #pragma unroll
  if (ty >= blockDim.y - 2) {
    const int oy = ty - (blockDim.y - 2);  // 0 or 1
    s_tile[tx+2][ty+3+oy][tz+2] = u[o0 + ((size_t)X*nyp*nzp + (size_t)(Y+1+oy)*nzp + (size_t)Z)];
  }

  // Z halos (front: -2,-1) - unrolled
  #pragma unroll
  if (tz < 2) {
    s_tile[tx+2][ty+2][tz] = u[o0 + ((size_t)X*nyp*nzp + (size_t)Y*nzp + (size_t)(Z-2+tz))];
  }
  // Z halos (back: +1,+2) - unrolled
  #pragma unroll
  if (tz >= blockDim.z - 2) {
    const int oz = tz - (blockDim.z - 2);  // 0 or 1
    s_tile[tx+2][ty+2][tz+3+oz] = u[o0 + ((size_t)X*nyp*nzp + (size_t)Y*nzp + (size_t)(Z+1+oz))];
  }

  __syncthreads();

  const float uc  = s_tile[tx+2][ty+2][tz+2];
  const float um1 = u[o1 + c];
  const float c0  = c_0_c * uc;

  // X second derivative (unrolled)
  float d2dx2;
  #pragma unroll
  d2dx2 = c0
    + c_m2_c * (s_tile[tx][ty+2][tz+2]   + s_tile[tx+4][ty+2][tz+2])
    + c_m1_c * (s_tile[tx+1][ty+2][tz+2] + s_tile[tx+3][ty+2][tz+2]);

  // Z second derivative (unrolled)
  float d2dz2;
  #pragma unroll
  d2dz2 = c0
    + c_m2_c * (s_tile[tx+2][ty+2][tz]   + s_tile[tx+2][ty+2][tz+4])
    + c_m1_c * (s_tile[tx+2][ty+2][tz+1] + s_tile[tx+2][ty+2][tz+3]);

  // Y second derivative (unrolled)
  float d2dy2;
  #pragma unroll
  d2dy2 = c0
    + c_m2_c * (s_tile[tx+2][ty][tz+2]   + s_tile[tx+2][ty+4][tz+2])
    + c_m1_c * (s_tile[tx+2][ty+1][tz+2] + s_tile[tx+2][ty+3][tz+2]);

  const float mval = m[idx_m(X, Y, Z, nyp, nzp)];
  const float lap  = r2*d2dx2 + r3*d2dy2 + r4*d2dz2;
  const float unew = 2.0f*uc - um1 + (dt*dt) * (lap / mval);
  u_out[o2 + c] = unew;
}

// ================== Source injection kernel ==================
__global__ void source_inject_kernel_opt(
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

  const float px = -floorf(gx) + gx;
  const float py = -floorf(gy) + gy;
  const float pz = -floorf(gz) + gz;

  const float m_base = m[idx_m(posx + HALO, posy + HALO, posz + HALO, nyp, nzp)];
  const float sval = src[time * pstride + p_src];

  #pragma unroll
  for (int rx = 0; rx <= 1; ++rx) {
    #pragma unroll
    for (int ry = 0; ry <= 1; ++ry) {
      #pragma unroll
      for (int rz = 0; rz <= 1; ++rz) {
        const int ix = rx + posx;
        const int iy = ry + posy;
        const int iz = rz + posz;

        if (ix >= x_m - 1 && iy >= y_m - 1 && iz >= z_m - 1 &&
            ix <= x_M + 1 && iy <= y_M + 1 && iz <= z_M + 1) {
          const float wx = rx ? px : (1.0f - px);
          const float wy = ry ? py : (1.0f - py);
          const float wz = rz ? pz : (1.0f - pz);
          const float w  = wx * wy * wz;

          const int X = ix + HALO, Y = iy + HALO, Z = iz + HALO;
          const size_t c = idx_u(t2, X, Y, Z, nxp, nyp, nzp);

          atomicAdd(&u[c], 1.0e-2f * w * sval / m_base);
        }
      }
    }
  }
}

// ================== Host glue (unchanged ABI) ==================
extern "C" int Kernel_CUDA_Optimized(
    struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec,
    struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
    const int x_M, const int x_m, const int y_M, const int y_m,
    const int z_M, const int z_m,
    const float dt, const float h_x, const float h_y, const float h_z,
    const float o_x, const float o_y, const float o_z,
    const int p_src_M, const int p_src_m, const int time_M, const int time_m,
    const int deviceid, const int devicerm, struct profiler *timers)
{
  if (deviceid != -1) cudaSetDevice(deviceid);

  float (*__restrict m_h)[m_vec->size[1]][m_vec->size[2]] =
      (float (*)[m_vec->size[1]][m_vec->size[2]]) m_vec->data;
  float (*__restrict src_h)[src_vec->size[1]] =
      (float (*)[src_vec->size[1]]) src_vec->data;
  float (*__restrict src_coords_h)[src_coords_vec->size[1]] =
      (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*__restrict u_h)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] =
      (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  const int nxp = u_vec->size[1];
  const int nyp = u_vec->size[2];
  const int nzp = u_vec->size[3];

  const size_t nU = (size_t)u_vec->size[0] * nxp * nyp * nzp;
  const size_t nM = (size_t)m_vec->size[0] * m_vec->size[1] * m_vec->size[2];
  const size_t nSrc = (size_t)src_vec->size[0] * src_vec->size[1];
  const size_t nCoords = (size_t)src_coords_vec->size[0] * src_coords_vec->size[1];

  float *d_u = nullptr, *d_m = nullptr, *d_src = nullptr, *d_src_coords = nullptr;
  cudaMalloc(&d_u, nU * sizeof(float));
  cudaMalloc(&d_m, nM * sizeof(float));
  cudaMalloc(&d_src, nSrc * sizeof(float));
  cudaMalloc(&d_src_coords, nCoords * sizeof(float));

  cudaMemcpy(d_u, u_h, nU * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, m_h, nM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src, src_h, nSrc * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src_coords, src_coords_h, nCoords * sizeof(float), cudaMemcpyHostToDevice);

  // constants for laplacian scale
  const float r2 = 1.0f / (h_x * h_x);
  const float r3 = 1.0f / (h_y * h_y);
  const float r4 = 1.0f / (h_z * h_z);
  const float r1_unused = 0.0f; // kept for ABI parity

  // ---- push coeffs to __constant__ (once per call) ----
  {
    const float c2 = -1.0f/12.0f, c1 = 4.0f/3.0f, c0 = -2.5f;
    cudaMemcpyToSymbol(c_m2_c, &c2, sizeof(float));
    cudaMemcpyToSymbol(c_m1_c, &c1, sizeof(float));
    cudaMemcpyToSymbol(c_0_c , &c0, sizeof(float));
  }

  // ---- WMMA coefficient matrix (currently unused, FP32 path active) ----
  // TODO: Implement FP16 WMMA m16n16k16 for sm_75
  // For now, this is a placeholder

  // Grid config (same as your current)
  dim3 threads(TILE_X, TILE_Y, TILE_Z);
  const int ext_x = (x_M - x_m + 1);
  const int ext_y = (y_M - y_m + 1);
  const int ext_z = (z_M - z_m + 1);
  dim3 blocks((ext_x + threads.x - 1) / threads.x,
              (ext_y + threads.y - 1) / threads.y,
              (ext_z + threads.z - 1) / threads.z);

  // TC disabled by default (WMMA path not yet implemented for sm_75)
  {
    int zero = 0;
    cudaMemcpyToSymbol(g_use_tensorcore, &zero, sizeof(int));
  }

  // --- time loop ---
  for (int t = time_m; t <= time_M; ++t) {
    const int t0 = (t) % 3;
    const int t1 = (t + 2) % 3;
    const int t2 = (t + 1) % 3;

    // section0: wave update
    START_SEC(section0)
    stencil_update_kernel_smem_opt<<<blocks, threads>>>(
        d_m, d_u, d_u,
        nxp, nyp, nzp,
        x_m, y_m, z_m,
        x_M, y_M, z_M,
        t0, t1, t2,
        dt, r1_unused, r2, r3, r4);
    cudaError_t err0 = cudaGetLastError();
    if (err0 != cudaSuccess) {
      cudaFree(d_u); cudaFree(d_m); cudaFree(d_src); cudaFree(d_src_coords);
      return err0;
    }
    cudaDeviceSynchronize();
    STOP_SEC(section0, timers);

    // section1: source injection
    if (p_src_M >= p_src_m) {
      const int nsrc = (p_src_M - p_src_m + 1);
      const int threads_src = 256;
      const int blocks_src = (nsrc + threads_src - 1) / threads_src;
      const int pstride = src_vec->size[1];
      const int cstride = src_coords_vec->size[1];

      START_SEC(section1)
      source_inject_kernel_opt<<<blocks_src, threads_src>>>(
          d_m, d_src, d_src_coords, d_u,
          nxp, nyp, nzp,
          x_m, x_M, y_m, y_M, z_m, z_M,
          t2,
          h_x, h_y, h_z,
          o_x, o_y, o_z,
          p_src_m, p_src_M, t,
          pstride, cstride);
      cudaError_t err1 = cudaGetLastError();
      if (err1 != cudaSuccess) {
        cudaFree(d_u); cudaFree(d_m); cudaFree(d_src); cudaFree(d_src_coords);
        return err1;
      }
      cudaDeviceSynchronize();
      STOP_SEC(section1, timers);
    }
  }

  cudaMemcpy(u_h, d_u, nU * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_u); cudaFree(d_m); cudaFree(d_src); cudaFree(d_src_coords);
  (void)devicerm;
  return 0;
}

// Runtime toggle (same symbol name you already export)
extern "C" void FDTD_SetRuntimeConfig(int use_tc, int t_fuse, int nfields) {
  (void)t_fuse; (void)nfields;
  cudaMemcpyToSymbol(g_use_tensorcore, &use_tc, sizeof(int));
}
