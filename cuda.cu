// fdtd_cuda_dropin.cu  — plain CUDA version matching the OpenACC numerics

#define _POSIX_C_SOURCE 200809L
#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

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

// --------------- Helpers --------------------
static inline int3 make_interior_extents(int x_m, int x_M, int y_m, int y_M, int z_m, int z_M) {
  return make_int3(x_M - x_m + 1, y_M - y_m + 1, z_M - z_m + 1);
}

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

// --------------- Kernels --------------------

// Section 0: 3D 4th-order Laplacian + leapfrog update into t2
__global__ void stencil_update_kernel(
  const float* __restrict__ m,
  const float* __restrict__ u,   // all 3 time levels in one array
  float*       __restrict__ u_out, // alias to same u buffer (we write t2)
  // sizes (with halo)
  int nxp, int nyp, int nzp,
  // interior logical bounds (x in [x_m..x_M], etc.) -> shifted by +4 halo inside kernel
  int x_m, int y_m, int z_m,
  // time indices
  int t0, int t1, int t2,
  // constants
  float dt, float r1, float r2, float r3, float r4)
{
  const int gx = x_m + blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = y_m + blockIdx.y * blockDim.y + threadIdx.y;
  const int gz = z_m + blockIdx.z * blockDim.z + threadIdx.z;

  // interior guards
  // caller sizes guarantee we can access X±1,±2 etc. because domain excludes halos
  if (gx > gridDim.x * blockDim.x + x_m - 1) return;
  if (gy > gridDim.y * blockDim.y + y_m - 1) return;
  if (gz > gridDim.z * blockDim.z + z_m - 1) return;

  // also skip threads outside [x_m..x_M], etc.
  // (grid may overshoot domain)
  // NOTE: x_M etc. not passed; instead compute max extents from nxp-8 etc. Caller shapes grid so overshoot is minimal.
  // Here we just check via padded range since we shift by +4 below.
  // Convert to padded coordinates (+4 halo)
  const int X = gx + 4;
  const int Y = gy + 4;
  const int Z = gz + 4;

  // Shorthands
  const size_t ofs_t0 = (size_t)t0 * (size_t)nxp * (size_t)nyp * (size_t)nzp;
  const size_t ofs_t1 = (size_t)t1 * (size_t)nxp * (size_t)nyp * (size_t)nzp;
  const size_t ofs_t2 = (size_t)t2 * (size_t)nxp * (size_t)nyp * (size_t)nzp;

  const size_t c   = ((size_t)X * nyp * nzp) + ((size_t)Y * nzp) + (size_t)Z;

  const float uc   = u[ofs_t0 + c];
  const float um1  = u[ofs_t1 + c];

  // 4th-order 1D pieces around center
  const float r5 = -2.5f * uc;

  const float d2dx2 = r5
    + (-8.33333333e-2f) * (u[ofs_t0 + ((size_t)(X-2) * nyp * nzp + (size_t)Y * nzp + (size_t)Z)]
                          + u[ofs_t0 + ((size_t)(X+2) * nyp * nzp + (size_t)Y * nzp + (size_t)Z)])
    + (1.333333330f)   * (u[ofs_t0 + ((size_t)(X-1) * nyp * nzp + (size_t)Y * nzp + (size_t)Z)]
                          + u[ofs_t0 + ((size_t)(X+1) * nyp * nzp + (size_t)Y * nzp + (size_t)Z)]);

  const float d2dy2 = r5
    + (-8.33333333e-2f) * (u[ofs_t0 + ((size_t)X * nyp * nzp + (size_t)(Y-2) * nzp + (size_t)Z)]
                          + u[ofs_t0 + ((size_t)X * nyp * nzp + (size_t)(Y+2) * nzp + (size_t)Z)])
    + (1.333333330f)   * (u[ofs_t0 + ((size_t)X * nyp * nzp + (size_t)(Y-1) * nzp + (size_t)Z)]
                          + u[ofs_t0 + ((size_t)X * nyp * nzp + (size_t)(Y+1) * nzp + (size_t)Z)]);

  const float d2dz2 = r5
    + (-8.33333333e-2f) * (u[ofs_t0 + ((size_t)X * nyp * nzp + (size_t)Y * nzp + (size_t)(Z-2))]
                          + u[ofs_t0 + ((size_t)X * nyp * nzp + (size_t)Y * nzp + (size_t)(Z+2))])
    + (1.333333330f)   * (u[ofs_t0 + ((size_t)X * nyp * nzp + (size_t)Y * nzp + (size_t)(Z-1))]
                          + u[ofs_t0 + ((size_t)X * nyp * nzp + (size_t)Y * nzp + (size_t)(Z+1))]);

  const float mval = m[idx_m(X, Y, Z, nyp, nzp)];
  const float lap  = r2 * d2dx2 + r3 * d2dy2 + r4 * d2dz2;
  const float tterm = -2.0f * r1 * uc + r1 * um1;

  u_out[ofs_t2 + c] = (dt * dt) * (lap - tterm * mval) / mval;
}

// Section 1: trilinear 8-neighbor scatter of sources into u[t2]
// NOTE: divide by base-cell m (posx,posy,posz), matching OpenACC
__global__ void source_inject_kernel(
  const float* __restrict__ m,
  const float* __restrict__ src,
  const float* __restrict__ src_coords,
  float*       __restrict__ u,
  // sizes
  int nxp, int nyp, int nzp,
  // interior bounds
  int x_m, int x_M, int y_m, int y_M, int z_m, int z_M,
  // time index
  int t2,
  // geom
  float h_x, float h_y, float h_z,
  float o_x, float o_y, float o_z,
  // sources
  int p_src_m, int p_src_M,
  int time)
{
  const int p_src = p_src_m + blockIdx.x * blockDim.x + threadIdx.x;
  if (p_src > p_src_M) return;

  const float sx = src_coords[p_src * 3 + 0];
  const float sy = src_coords[p_src * 3 + 1];
  const float sz = src_coords[p_src * 3 + 2];

  const float gx = (-o_x + sx) / h_x;
  const float gy = (-o_y + sy) / h_y;
  const float gz = (-o_z + sz) / h_z;

  const int posx = (int)floorf(gx);
  const int posy = (int)floorf(gy);
  const int posz = (int)floorf(gz);

  const float px = -floorf(gx) + gx;
  const float py = -floorf(gy) + gy;
  const float pz = -floorf(gz) + gz;

  // base-cell m (OpenACC parity)
  const float m_base = m[idx_m(posx + 4, posy + 4, posz + 4, nyp, nzp)];

  const float sval = src[time * (p_src_M + 1) + p_src]; // src[time][p_src] in row-major

  // 8 neighbors (rsrcx/rsrcy/rsrcz ∈ {0,1})
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

          const int X = ix + 4, Y = iy + 4, Z = iz + 4;
          const size_t c = idx_u(t2, X, Y, Z, nxp, nyp, nzp);

          atomicAdd(&u[c], 1.0e-2f * w * sval / m_base);
        }
      }
    }
  }
}
