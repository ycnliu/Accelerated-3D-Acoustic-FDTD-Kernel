// fdtd_cuda_plain_4th.cu — Plain CUDA 4th-order stencil
// Timed like OpenACC (host wall clock around each section)

#define _POSIX_C_SOURCE 200809L
#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// -------- timing helpers (match OpenACC style) --------
#define START_SEC(S) struct timeval start_##S, end_##S; gettimeofday(&start_##S, nullptr);
#define STOP_SEC(S, T) do { \
  gettimeofday(&end_##S, nullptr); \
  (T)->S += (double)(end_##S.tv_sec - start_##S.tv_sec) + \
            (double)(end_##S.tv_usec - start_##S.tv_usec) / 1e6; \
} while(0)

// -------- no temporal fusion for standard 4th order --------

// ---------------- Shared ABI ----------------
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
  double section0; // main update
  double section1; // source inject
};

// Flattened indices for u: [t][X][Y][Z] with sizes [3][nxp][nyp][nzp]
__device__ __forceinline__
size_t idx_u(int t, int X, int Y, int Z, int nxp, int nyp, int nzp) {
  return ((size_t)t * nxp * nyp * nzp) + ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
}

// Flattened indices for m: [X][Y][Z] with sizes [nxp][nyp][nzp]
__device__ __forceinline__
size_t idx_m(int X, int Y, int Z, int nyp, int nzp) {
  return ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
}

// --------------- 4th-order coefficients --------------------
__constant__ float c_m2 = -8.33333333e-2f;   // -1/12
__constant__ float c_m1 = 1.333333330f;      // 4/3
__constant__ float c_0  = -2.50f;            // -5/2

#define HALO 4  // 4th order needs radius 2, so 2*2 = 4
#define WARMUP_STEPS 5  // GPU warmup iterations before timing

// ---------------- kernels ----------------

// Section 0: SINGLE-STEP 4th-order Laplacian + leapfrog update into t2
__global__ void stencil_update_kernel_1step(
  const float* __restrict__ m,
  const float* __restrict__ u,     // all 3 time levels in one array
  float*       __restrict__ u_out, // alias to same u buffer (we write t2)
  int nxp, int nyp, int nzp,
  int x_m, int y_m, int z_m,
  int x_M, int y_M, int z_M,
  int t0, int t1, int t2,
  float dt, float r1, float r2, float r3, float r4)
{
  const int gx = x_m + blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = y_m + blockIdx.y * blockDim.y + threadIdx.y;
  const int gz = z_m + blockIdx.z * blockDim.z + threadIdx.z;
  if (gx > x_M || gy > y_M || gz > z_M) return;

  const int X = gx + HALO, Y = gy + HALO, Z = gz + HALO;
  const size_t nper = (size_t)nxp * nyp * nzp;
  const size_t o0 = (size_t)t0 * nper;
  const size_t o1 = (size_t)t1 * nper;
  const size_t o2 = (size_t)t2 * nper;
  const size_t c  = ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;

  const float uc  = u[o0 + c];
  const float um1 = u[o1 + c];

  const float r5 = c_0 * uc;

  // 4th-order finite difference stencil in X direction
  const float d2dx2 = r5
    + c_m2*(u[o0 + ((size_t)(X-2)*nyp*nzp + (size_t)Y*nzp + (size_t)Z)] + u[o0 + ((size_t)(X+2)*nyp*nzp + (size_t)Y*nzp + (size_t)Z)])
    + c_m1*(u[o0 + ((size_t)(X-1)*nyp*nzp + (size_t)Y*nzp + (size_t)Z)] + u[o0 + ((size_t)(X+1)*nyp*nzp + (size_t)Y*nzp + (size_t)Z)]);

  // 4th-order finite difference stencil in Y direction
  const float d2dy2 = r5
    + c_m2*(u[o0 + ((size_t)X*nyp*nzp + (size_t)(Y-2)*nzp + (size_t)Z)] + u[o0 + ((size_t)X*nyp*nzp + (size_t)(Y+2)*nzp + (size_t)Z)])
    + c_m1*(u[o0 + ((size_t)X*nyp*nzp + (size_t)(Y-1)*nzp + (size_t)Z)] + u[o0 + ((size_t)X*nyp*nzp + (size_t)(Y+1)*nzp + (size_t)Z)]);

  // 4th-order finite difference stencil in Z direction
  const float d2dz2 = r5
    + c_m2*(u[o0 + ((size_t)X*nyp*nzp + (size_t)Y*nzp + (size_t)(Z-2))] + u[o0 + ((size_t)X*nyp*nzp + (size_t)Y*nzp + (size_t)(Z+2))])
    + c_m1*(u[o0 + ((size_t)X*nyp*nzp + (size_t)Y*nzp + (size_t)(Z-1))] + u[o0 + ((size_t)X*nyp*nzp + (size_t)Y*nzp + (size_t)(Z+1))]);

  const float mval = m[idx_m(X, Y, Z, nyp, nzp)];
  const float lap  = r2*d2dx2 + r3*d2dy2 + r4*d2dz2;
  const float unew  = 2.0f * uc - um1 + (dt*dt) * lap / mval;

  u_out[o2 + c] = unew;
}


// Section 1: trilinear 8-neighbor scatter of sources into u[t2] (unchanged)
__global__ void source_inject_kernel(
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

  for (int rx = 0; rx <= 1; ++rx) {
    for (int ry = 0; ry <= 1; ++ry) {
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

// --------------- Host wrapper --------------------
extern "C" int Kernel_CUDA(
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

  // Allocate device memory
  float *d_u = nullptr, *d_m = nullptr, *d_src = nullptr, *d_src_coords = nullptr;
  cudaMalloc(&d_u, nU * sizeof(float));
  cudaMalloc(&d_m, nM * sizeof(float));
  cudaMalloc(&d_src, nSrc * sizeof(float));
  cudaMalloc(&d_src_coords, nCoords * sizeof(float));

  // Copy to device
  cudaMemcpy(d_u, u_h, nU * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, m_h, nM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src, src_h, nSrc * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src_coords, src_coords_h, nCoords * sizeof(float), cudaMemcpyHostToDevice);

  // Constants
  const float r1 = 1.0f / (dt * dt);
  const float r2 = 1.0f / (h_x * h_x);
  const float r3 = 1.0f / (h_y * h_y);
  const float r4 = 1.0f / (h_z * h_z);

  // Grid configuration for stencil kernel
  dim3 threads(8, 8, 8);  // 512 threads per block
  const int ext_x = (x_M - x_m + 1);
  const int ext_y = (y_M - y_m + 1);
  const int ext_z = (z_M - z_m + 1);
  dim3 blocks((ext_x + threads.x - 1) / threads.x,
              (ext_y + threads.y - 1) / threads.y,
              (ext_z + threads.z - 1) / threads.z);

  // Warmup iterations to initialize GPU caches and ensure stable timing
  for (int t = time_m; t < time_m + WARMUP_STEPS && t <= time_M; ++t) {
    const int t0 = (t) % 3;
    const int t1 = (t + 2) % 3;
    const int t2 = (t + 1) % 3;

    stencil_update_kernel_1step<<<blocks, threads>>>(
        d_m, d_u, d_u,
        nxp, nyp, nzp,
        x_m, y_m, z_m,
        x_M, y_M, z_M,
        t0, t1, t2,
        dt, r1, r2, r3, r4);
    cudaDeviceSynchronize();

    if (p_src_M >= p_src_m) {
      const int nsrc = (p_src_M - p_src_m + 1);
      const int threads_src = 256;
      const int blocks_src = (nsrc + threads_src - 1) / threads_src;
      const int pstride = src_vec->size[1];
      const int cstride = src_coords_vec->size[1];

      source_inject_kernel<<<blocks_src, threads_src>>>(
          d_m, d_src, d_src_coords, d_u,
          nxp, nyp, nzp,
          x_m, x_M, y_m, y_M, z_m, z_M,
          t2,
          h_x, h_y, h_z,
          o_x, o_y, o_z,
          p_src_m, p_src_M, t,
          pstride, cstride);
      cudaDeviceSynchronize();
    }
  }

  // Timed loop - starts after warmup
  for (int t = time_m + WARMUP_STEPS; t <= time_M; ++t) {
    const int t0 = (t) % 3;
    const int t1 = (t + 2) % 3;
    const int t2 = (t + 1) % 3;

    START_SEC(section0)
    stencil_update_kernel_1step<<<blocks, threads>>>(
        d_m, d_u, d_u,
        nxp, nyp, nzp,
        x_m, y_m, z_m,
        x_M, y_M, z_M,
        t0, t1, t2,
        dt, r1, r2, r3, r4);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      cudaFree(d_u); cudaFree(d_m); cudaFree(d_src); cudaFree(d_src_coords);
      return err;
    }
    cudaDeviceSynchronize();
    STOP_SEC(section0, timers);

    if (p_src_M >= p_src_m) {
      const int nsrc = (p_src_M - p_src_m + 1);
      const int threads_src = 256;
      const int blocks_src = (nsrc + threads_src - 1) / threads_src;

      const int pstride = src_vec->size[1];
      const int cstride = src_coords_vec->size[1];

      START_SEC(section1)
      source_inject_kernel<<<blocks_src, threads_src>>>(
          d_m, d_src, d_src_coords, d_u,
          nxp, nyp, nzp,
          x_m, x_M, y_m, y_M, z_m, z_M,
          t2,
          h_x, h_y, h_z,
          o_x, o_y, o_z,
          p_src_m, p_src_M, t,
          pstride, cstride);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        cudaFree(d_u); cudaFree(d_m); cudaFree(d_src); cudaFree(d_src_coords);
        return err;
      }
      cudaDeviceSynchronize();
      STOP_SEC(section1, timers);
    }
  }

  // Copy back to host
  cudaMemcpy(u_h, d_u, nU * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_u); cudaFree(d_m); cudaFree(d_src); cudaFree(d_src_coords);
  (void)devicerm; // unused
  return 0;
}
