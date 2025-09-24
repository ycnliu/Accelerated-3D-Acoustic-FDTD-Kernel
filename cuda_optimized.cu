// fdtd_cuda_optimized_final.cu
// Fully optimized Section0 with rolling XY-plane window, coalesced Z, __ldg(), unroll, optional float4.
// Section1 unchanged (separate kernel) to keep timing parity with baseline.
//
// Build suggestion (Turing/2080 Ti):
// nvcc -O3 -std=c++14 -arch=sm_75 -lineinfo -Xptxas=-O3,-dlcm=ca \
//      -DTZ=32 -DTY=8 -DXCHUNK=64 -c fdtd_cuda_optimized_final.cu -o fdtd_cuda_optimized_final.o

#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

// ----- Shared ABI -----
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
struct profiler { double section0, section1; };

// ----- Helpers -----
__device__ __forceinline__
size_t idx_u_txyz(int t, int X, int Y, int Z, int nxp, int nyp, int nzp) {
  return ((size_t)t * nxp * nyp * nzp) + ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
}
__device__ __forceinline__
size_t idx_m_xyz(int X, int Y, int Z, int nyp, int nzp) {
  return ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;
}
static inline int divUp(int a, int b) { return (a + b - 1) / b; }

// ----- Tunables -----
#ifndef TY
#define TY 8         // threads on Y
#endif
#ifndef TZ
#define TZ 32        // threads on Z (unit-stride)
#endif
#ifndef XCHUNK
#define XCHUNK 64    // X span per block (swept)
#endif
#define R 2
#define HALO 4

// ======== SECTION 0: rolling-plane optimized stencil ========
__global__ void stencil_update_plane_kernel(
    const float* __restrict__ m,
    const float* __restrict__ u,     // all 3 time levels [t][X][Y][Z]
    float*       __restrict__ u_out, // same layout
    int nxp, int nyp, int nzp,
    int x_m, int x_M, int y_m, int y_M, int z_m, int z_M,
    int t0, int t1, int t2,
    float dt, float r1, float r2, float r3, float r4)
{
  // Block maps to a YZ tile; sweep X with a 5-plane rolling window.
  const int gz = z_m + blockIdx.x * TZ + threadIdx.x;  // contiguous axis
  const int gy = y_m + blockIdx.y * TY + threadIdx.y;
  const int x0 = x_m + blockIdx.z * XCHUNK;
  const int x1 = min(x0 + XCHUNK - 1, x_M);
  if (gz > z_M || gy > y_M) return;

  const int Ypad = gy + HALO;
  const int Zpad = gz + HALO;
  if (gy < y_m || gz < z_m) return;

  const int sX = nyp * nzp;
  const int sY = nzp;

  // 5 rolling planes in SMEM, halo R in Y/Z, pad +1 on fastest dim to avoid bank conflicts
  __shared__ float plane[5][TY + 2*R][TZ + 2*R + 1];

  const size_t ofs_t0 = (size_t)t0 * (size_t)nxp * nyp * nzp;
  const size_t ofs_t1 = (size_t)t1 * (size_t)nxp * nyp * nzp;
  const size_t ofs_t2 = (size_t)t2 * (size_t)nxp * nyp * nzp;

  auto gIndex = [&](int Xpad) {
    return (size_t)Xpad * nyp * nzp + (size_t)Ypad * nzp + (size_t)Zpad;
  };

  // Optional vectorized Z loads when aligned (Z contiguous)
  const bool z_aligned = ((Zpad & 3) == 0) && (nzp % 4 == 0) && (TZ % 4 == 0);

  auto load_plane = [&](int buf, int Xpad) {
    const size_t c = gIndex(Xpad);

    // center
    if (z_aligned) {
      // Load a float4 chunk per 4-lane group; unpack in-lane
      // Lane-local offset inside group of 4
      const int lane4 = threadIdx.x & 3;
      const size_t base4 = (ofs_t0 + c) >> 2; // /4
      float4 v = __ldg(reinterpret_cast<const float4*>(&u[ofs_t0 + c - lane4]));
      float val = (lane4 == 0 ? v.x : lane4 == 1 ? v.y : lane4 == 2 ? v.z : v.w);
      plane[buf][threadIdx.y + R][threadIdx.x + R] = val;
    } else {
      plane[buf][threadIdx.y + R][threadIdx.x + R] = __ldg(&u[ofs_t0 + c]);
    }

    // Z halos (fast axis)
    if (threadIdx.x < R) {
      plane[buf][threadIdx.y + R][threadIdx.x]         = __ldg(&u[ofs_t0 + c - R]);
      plane[buf][threadIdx.y + R][threadIdx.x + (R-1)] = __ldg(&u[ofs_t0 + c - 1]);
    }
    if (threadIdx.x >= TZ - R) {
      plane[buf][threadIdx.y + R][threadIdx.x + 1]     = __ldg(&u[ofs_t0 + c + 1]);
      plane[buf][threadIdx.y + R][threadIdx.x + R]     = __ldg(&u[ofs_t0 + c + R]);
    }
    // Y halos
    if (threadIdx.y < R) {
      plane[buf][threadIdx.y][threadIdx.x + R]         = __ldg(&u[ofs_t0 + c - R * sY]);
      plane[buf][threadIdx.y + (R-1)][threadIdx.x + R] = __ldg(&u[ofs_t0 + c - 1 * sY]);
    }
    if (threadIdx.y >= TY - R) {
      plane[buf][threadIdx.y + 1][threadIdx.x + R]     = __ldg(&u[ofs_t0 + c + 1 * sY]);
      plane[buf][threadIdx.y + R][threadIdx.x + R]     = __ldg(&u[ofs_t0 + c + R * sY]);
    }
  };

  // Preload 5 planes for x0-2 .. x0+2 (padded indices)
  int Xpad_m2 = (x0 - 2) + HALO;
  int Xpad_m1 = (x0 - 1) + HALO;
  int Xpad_0  = (x0     ) + HALO;
  int Xpad_p1 = (x0 + 1 ) + HALO;
  int Xpad_p2 = (x0 + 2 ) + HALO;

  load_plane(0, Xpad_m2);
  load_plane(1, Xpad_m1);
  load_plane(2, Xpad_0 );
  load_plane(3, Xpad_p1);
  load_plane(4, Xpad_p2);
  __syncthreads();

  const float c_m2 = -8.33333333e-2f; // -1/12
  const float c_m1 =  1.333333330f;   //  4/3
  const float c_0  = -2.5f;           // -30/12

  int cur = 2; // buffer index for X=x
  for (int x = x0; x <= x1; ++x) {
    const int prev  = (cur + 4) % 5; // x-1
    const int prev2 = (cur + 3) % 5; // x-2
    const int next  = (cur + 1) % 5; // x+1
    const int next2 = (cur + 2) % 5; // x+2

    const int sy = threadIdx.y + R;
    const int sz = threadIdx.x + R;

    const float uc  = plane[cur][sy][sz];
    const size_t c1 = gIndex(x + HALO);
    const float um1 = __ldg(&u[ofs_t1 + c1]);
    const float r5  = c_0 * uc;

    // Unrolled 4th-order taps
    #pragma unroll
    float d2dx2 = r5
      + c_m2 * (plane[prev2][sy][sz] + plane[next2][sy][sz])
      + c_m1 * (plane[prev ][sy][sz] + plane[next ][sy][sz]);

    #pragma unroll
    float d2dy2 = r5
      + c_m2 * (plane[cur][sy - R][sz] + plane[cur][sy + R][sz])
      + c_m1 * (plane[cur][sy - 1][sz] + plane[cur][sy + 1][sz]);

    #pragma unroll
    float d2dz2 = r5
      + c_m2 * (plane[cur][sy][sz - R] + plane[cur][sy][sz + R])
      + c_m1 * (plane[cur][sy][sz - 1] + plane[cur][sy][sz + 1]);

    const float lap   = r2 * d2dx2 + r3 * d2dy2 + r4 * d2dz2;
    const float mval  = __ldg(&m[idx_m_xyz(x + HALO, Ypad, Zpad, nyp, nzp)]);
    const float tterm = -2.0f * r1 * uc + r1 * um1;
    const float unew  = (dt * dt) * (lap - tterm * mval) / mval;

    u_out[ofs_t2 + gIndex(x + HALO)] = unew;

    // Advance rolling window: load X+3 plane into "prev2" buffer for next iteration
    if (x + 3 <= x1 + 2) {
      const int Xpad_new = (x + 3) + HALO;
      load_plane(prev2, Xpad_new);
    }
    __syncthreads();
    cur = next;
  }
}

// ======== SECTION 1: unchanged trilinear source injection ========
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

  const float sval   = __ldg(&src[time * (p_src_M + 1) + p]);
  const float m_base = __ldg(&m[idx_m_xyz(posx + HALO, posy + HALO, posz + HALO, nyp, nzp)]);

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

        const size_t c = idx_u_txyz(t2, ix + HALO, iy + HALO, iz + HALO, nxp, nyp, nzp);
        atomicAdd(&u[c], 1.0e-2f * w * sval / m_base);
      }
    }
  }
}

// ======== Host wrapper (same ABI/timing) ========
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

  // extents
  const int ext_x = (x_M - x_m + 1);
  const int ext_y = (y_M - y_m + 1);
  const int ext_z = (z_M - z_m + 1);

  // launch: (Z,Y) tile; X chunk per block
  dim3 block(TZ, TY, 1);
  dim3 grid(divUp(ext_z, TZ), divUp(ext_y, TY), divUp(ext_x, XCHUNK));

  // cache config: rolling planes use SMEM heavily—prefer shared
  cudaFuncSetCacheConfig(stencil_update_plane_kernel, cudaFuncCachePreferShared);

  // timers
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
    stencil_update_plane_kernel<<<grid, block>>>(
      d_m, d_u, d_u, nxp, nyp, nzp,
      x_m, x_M, y_m, y_M, z_m, z_M,
      t0, t1, t2,
      dt, r1, r2, r3, r4);
    cudaEventRecord(ev0b);
    cudaEventSynchronize(ev0b);
    float s0; cudaEventElapsedTime(&s0, ev0a, ev0b);
    s0_ms_acc += s0;

    // Section 1 (unchanged)
    if (src_vec->size[0]*src_vec->size[1] > 0 && (p_src_M - p_src_m + 1) > 0) {
      const int nsrc = (p_src_M - p_src_m + 1);
      const int threads = 128;
      const int blocks  = divUp(nsrc, threads);

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
      float s1; cudaEventElapsedTime(&s1, ev1a, ev1b);
      s1_ms_acc += s1;
    }
  }

  // copy back
  cudaMemcpy(u_h, d_u, nU * sizeof(float), cudaMemcpyDeviceToHost);

  if (timers) {
    timers->section0 += (double)s0_ms_acc / 1000.0;
    timers->section1 += (double)s1_ms_acc / 1000.0;
  }

  cudaEventDestroy(ev0a); cudaEventDestroy(ev0b);
  cudaEventDestroy(ev1a); cudaEventDestroy(ev1b);

  cudaFree(d_u); cudaFree(d_m); cudaFree(d_src); cudaFree(d_src_coords);
  return 0;
}
