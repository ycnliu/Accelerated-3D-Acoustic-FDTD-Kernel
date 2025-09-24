// fdtd_cuda_optimized.cu — optimized CUDA with shared-memory tiling
// Matches the plain CUDA ABI and section timings for fair comparison.

#include <cuda_runtime.h>
#include <cmath>

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

// ---------------- Helpers ----------------
__device__ __forceinline__
size_t idx_u_txyz(int t, int X, int Y, int Z, int nxp, int nyp, int nzp) {
  return ((size_t)t * nxp * nyp * nzp) + ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
}
__device__ __forceinline__
size_t idx_m_xyz(int X, int Y, int Z, int nyp, int nzp) {
  return ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
}

static inline int divUp(int a, int b) { return (a + b - 1) / b; }

// ---------------- Tunables ----------------
#ifndef BX
#define BX 16
#endif
#ifndef BY
#define BY 8
#endif
#ifndef BZ
#define BZ 4
#endif
#define R 2       // stencil radius for 4th order
#define HALO 4    // pad in input arrays (already present)

// ---------------- Kernels ----------------

// Section 0: Shared-memory tiled 3D 4th-order update
__global__ void stencil_update_sm_kernel(
    const float* __restrict__ m,
    const float* __restrict__ u,   // all 3 time levels
    float*       __restrict__ u_out,
    int nxp, int nyp, int nzp,     // padded sizes
    int x_m, int x_M,              // interior ranges
    int y_m, int y_M,
    int z_m, int z_M,
    int t0, int t1, int t2,
    float dt, float r1, float r2, float r3, float r4)
{
  // Global interior coords (unshifted)
  const int gx = x_m + blockIdx.x * BX + threadIdx.x;
  const int gy = y_m + blockIdx.y * BY + threadIdx.y;
  const int gz = z_m + blockIdx.z * BZ + threadIdx.z;

  if (gx > x_M || gy > y_M || gz > z_M) return;

  // Shift to padded indices
  const int X = gx + HALO;
  const int Y = gy + HALO;
  const int Z = gz + HALO;

  const int sx = threadIdx.x + R;
  const int sy = threadIdx.y + R;
  const int sz = threadIdx.z + R;

  // Shared tile with halo R=2 in each dim
  __shared__ float tile[BZ + 2*R][BY + 2*R][BX + 2*R];

  // Precompute plane strides
  const int sX = nyp * nzp;
  const int sY = nzp;

  // center
  const size_t c_t0 = ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
  tile[sz][sy][sx] = u[(size_t)t0 * (size_t)nxp * nyp * nzp + c_t0];

  // Load halos cooperatively
  // X- halos
  if (threadIdx.x < R) {
    // left ±1,±2
    tile[sz][sy][sx - R]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 - R * sX)];
    tile[sz][sy][sx - 1]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 - 1 * sX)];
  }
  if (threadIdx.x >= BX - R) {
    // right ±1,±2
    tile[sz][sy][sx + 1]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 + 1 * sX)];
    tile[sz][sy][sx + R]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 + R * sX)];
  }
  // Y halos
  if (threadIdx.y < R) {
    tile[sz][sy - R][sx]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 - R * sY)];
    tile[sz][sy - 1][sx]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 - 1 * sY)];
  }
  if (threadIdx.y >= BY - R) {
    tile[sz][sy + 1][sx]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 + 1 * sY)];
    tile[sz][sy + R][sx]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 + R * sY)];
  }
  // Z halos
  if (threadIdx.z < R) {
    tile[sz - R][sy][sx]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 - R)];
    tile[sz - 1][sy][sx]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 - 1)];
  }
  if (threadIdx.z >= BZ - R) {
    tile[sz + 1][sy][sx]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 + 1)];
    tile[sz + R][sy][sx]     = u[(size_t)t0 * nxp * nyp * nzp + (c_t0 + R)];
  }

  __syncthreads();

  // 4th-order coefficients
  const float c_m2 = -8.33333333e-2f; // -1/12
  const float c_m1 =  1.333333330f;   //  4/3
  const float c_0  = -2.5f;           // -30/12

  const float uc   = tile[sz][sy][sx];
  const float um1  = u[(size_t)t1 * (size_t)nxp * nyp * nzp + c_t0];
  const float r5   = c_0 * uc;

  // 1D 4th order in each axis using shared tile
  float d2dx2 = r5
    + c_m2 * (tile[sz][sy][sx - R] + tile[sz][sy][sx + R])
    + c_m1 * (tile[sz][sy][sx - 1] + tile[sz][sy][sx + 1]);

  float d2dy2 = r5
    + c_m2 * (tile[sz][sy - R][sx] + tile[sz][sy + R][sx])
    + c_m1 * (tile[sz][sy - 1][sx] + tile[sz][sy + 1][sx]);

  float d2dz2 = r5
    + c_m2 * (tile[sz - R][sy][sx] + tile[sz + R][sy][sx])
    + c_m1 * (tile[sz - 1][sy][sx] + tile[sz + 1][sy][sx]);

  const float lap   = r2 * d2dx2 + r3 * d2dy2 + r4 * d2dz2;
  const float mval  = m[idx_m_xyz(X, Y, Z, nyp, nzp)];
  const float tterm = -2.0f * r1 * uc + r1 * um1;

  const float unew  = (dt * dt) * (lap - tterm * mval) / mval;

  u_out[(size_t)t2 * (size_t)nxp * nyp * nzp + c_t0] = unew;
}

// Section 1: trilinear source injection (same semantics)
// atomicAdd on float; identical math as your plain CUDA
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
  int time)
{
  const int p = p_src_m + blockIdx.x * blockDim.x + threadIdx.x;
  if (p > p_src_M) return;

  const float sx = src_coords[p * 3 + 0];
  const float sy = src_coords[p * 3 + 1];
  const float sz = src_coords[p * 3 + 2];

  const float gx = (-o_x + sx) / h_x;
  const float gy = (-o_y + sy) / h_y;
  const float gz = (-o_z + sz) / h_z;

  const int posx = (int)floorf(gx);
  const int posy = (int)floorf(gy);
  const int posz = (int)floorf(gz);

  const float px = gx - floorf(gx);
  const float py = gy - floorf(gy);
  const float pz = gz - floorf(gz);

  const float sval = src[time * (p_src_M + 1) + p];
  const float m_base = m[idx_m_xyz(posx + HALO, posy + HALO, posz + HALO, nyp, nzp)];

  for (int rx = 0; rx <= 1; ++rx) {
    for (int ry = 0; ry <= 1; ++ry) {
      for (int rz = 0; rz <= 1; ++rz) {
        const int ix = rx + posx;
        const int iy = ry + posy;
        const int iz = rz + posz;
        if (ix < x_m - 1 || iy < y_m - 1 || iz < z_m - 1 ||
            ix > x_M + 1 || iy > y_M + 1 || iz > z_M + 1) continue;

        const float wx = rx ? px : (1.0f - px);
        const float wy = ry ? py : (1.0f - py);
        const float wz = rz ? pz : (1.0f - pz);
        const float w  = wx * wy * wz;

        const int X = ix + HALO, Y = iy + HALO, Z = iz + HALO;
        const size_t c = idx_u_txyz(t2, X, Y, Z, nxp, nyp, nzp);
        atomicAdd(&u[c], 1.0e-2f * w * sval / m_base);
      }
    }
  }
}

// ---------------- Host Wrapper (same signature) ----------------
extern "C" int Kernel(
  struct dataobj *__restrict m_vec,
  struct dataobj *__restrict src_vec,
  struct dataobj *__restrict src_coords_vec,
  struct dataobj *__restrict u_vec,
  const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
  const float dt, const float h_x, const float h_y, const float h_z,
  const float o_x, const float o_y, const float o_z,
  const int p_src_M, const int p_src_m, const int time_M, const int time_m,
  const int deviceid, const int devicerm, struct profiler * timers)
{
  if (deviceid != -1) cudaSetDevice(deviceid);

  float (*__restrict m_h)[m_vec->size[1]][m_vec->size[2]] = (float (*)[m_vec->size[1]][m_vec->size[2]]) m_vec->data;
  float (*__restrict src_h)[src_vec->size[1]]             = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*__restrict src_coords_h)[src_coords_vec->size[1]]= (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*__restrict u_h)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] =
      (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  const int nxp = u_vec->size[1];
  const int nyp = u_vec->size[2];
  const int nzp = u_vec->size[3];

  float *d_u=nullptr, *d_m=nullptr, *d_src=nullptr, *d_src_coords=nullptr;

  const size_t nU = (size_t)u_vec->size[0] * nxp * nyp * nzp;
  const size_t nM = (size_t)m_vec->size[0] * m_vec->size[1] * m_vec->size[2];
  const size_t nSrc = (size_t)src_vec->size[0] * src_vec->size[1];
  const size_t nCoords = (size_t)src_coords_vec->size[0] * src_coords_vec->size[1];

  cudaMalloc(&d_u,           nU * sizeof(float));
  cudaMalloc(&d_m,           nM * sizeof(float));
  cudaMalloc(&d_src,         nSrc * sizeof(float));
  cudaMalloc(&d_src_coords,  nCoords * sizeof(float));

  cudaMemcpy(d_u,           u_h,   nU * sizeof(float),          cudaMemcpyHostToDevice);
  cudaMemcpy(d_m,           m_h,   nM * sizeof(float),          cudaMemcpyHostToDevice);
  cudaMemcpy(d_src,         src_h, nSrc * sizeof(float),        cudaMemcpyHostToDevice);
  cudaMemcpy(d_src_coords,  src_coords_h, nCoords * sizeof(float), cudaMemcpyHostToDevice);

  // constants
  const float r1 = 1.0f / (dt * dt);
  const float r2 = 1.0f / (h_x * h_x);
  const float r3 = 1.0f / (h_y * h_y);
  const float r4 = 1.0f / (h_z * h_z);

  // launch config
  const int ext_x = (x_M - x_m + 1);
  const int ext_y = (y_M - y_m + 1);
  const int ext_z = (z_M - z_m + 1);

  dim3 block(BX, BY, BZ);
  dim3 grid(divUp(ext_x, BX), divUp(ext_y, BY), divUp(ext_z, BZ));

  // Favor L1 cache for global mem heavy kernel
  cudaFuncSetCacheConfig(stencil_update_sm_kernel, cudaFuncCachePreferL1);

  // events for accumulating section timings
  cudaEvent_t ev0a, ev0b, ev1a, ev1b;
  cudaEventCreate(&ev0a); cudaEventCreate(&ev0b);
  cudaEventCreate(&ev1a); cudaEventCreate(&ev1b);
  float s0_ms_acc = 0.0f, s1_ms_acc = 0.0f;

  for (int time = time_m; time <= time_M; ++time) {
    const int t0 = (time)     % 3;
    const int t1 = (time + 2) % 3;
    const int t2 = (time + 1) % 3;

    // Section 0
    cudaEventRecord(ev0a);
    stencil_update_sm_kernel<<<grid, block>>>(
      d_m, d_u, d_u,
      nxp, nyp, nzp,
      x_m, x_M, y_m, y_M, z_m, z_M,
      t0, t1, t2,
      dt, r1, r2, r3, r4);
    cudaEventRecord(ev0b);
    cudaEventSynchronize(ev0b);
    float s0;
    cudaEventElapsedTime(&s0, ev0a, ev0b);
    s0_ms_acc += s0;

    // Section 1
    if (src_vec->size[0]*src_vec->size[1] > 0 && (p_src_M - p_src_m + 1) > 0) {
      const int nsrc = (p_src_M - p_src_m + 1);
      const int threads = 128;
      const int blocks = divUp(nsrc, threads);

      cudaEventRecord(ev1a);
      source_inject_kernel<<<blocks, threads>>>(
        d_m, d_src, d_src_coords, d_u,
        nxp, nyp, nzp,
        x_m, x_M, y_m, y_M, z_m, z_M,
        t2,
        h_x, h_y, h_z,
        o_x, o_y, o_z,
        p_src_m, p_src_M,
        time);
      cudaEventRecord(ev1b);
      cudaEventSynchronize(ev1b);
      float s1;
      cudaEventElapsedTime(&s1, ev1a, ev1b);
      s1_ms_acc += s1;
    }
  }

  // copy back u (parity with original)
  cudaMemcpy(u_h, d_u, nU * sizeof(float), cudaMemcpyDeviceToHost);

  if (timers) {
    timers->section0 += (double)s0_ms_acc / 1000.0;
    timers->section1 += (double)s1_ms_acc / 1000.0;
  }

  cudaEventDestroy(ev0a); cudaEventDestroy(ev0b);
  cudaEventDestroy(ev1a); cudaEventDestroy(ev1b);

  cudaFree(d_u);
  cudaFree(d_m);
  cudaFree(d_src);
  cudaFree(d_src_coords);

  return 0;
}
