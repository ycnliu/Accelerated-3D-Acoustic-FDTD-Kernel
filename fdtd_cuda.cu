// fdtd_cuda.cu  — Baseline CUDA implementation (correctness-first, efficient)

#include <cuda_runtime.h>
#include <sys/time.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define START(S) \
  struct timeval start_##S, end_##S; \
  gettimeofday(&start_##S, NULL);

#define STOP(S,T) \
  gettimeofday(&end_##S, NULL); \
  (T)->S += (double)(end_##S.tv_sec - start_##S.tv_sec) + \
            (double)(end_##S.tv_usec - start_##S.tv_usec)/1e6;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err__));                                    \
      exit(1);                                                               \
    }                                                                        \
  } while(0)

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
  double section0;       // main stencil
  double section1;       // source inject
  double cuda_malloc;
  double cuda_memcpy;
  double cuda_memcpy_h2d;
  double cuda_memcpy_d2h;
  double kernel_time;    // overall kernel time (time loop)
};

// ---------------------------------------------
// Device kernels
// ---------------------------------------------

// 4th-order Laplacian + leapfrog time update (baseline, FP32)
__global__ void wave_kernel(
    float *__restrict__ u_t2,
    const float *__restrict__ u_t1,
    const float *__restrict__ u_t0,
    const float *__restrict__ m,        // read-only
    int x_M, int x_m, int y_M, int y_m, int z_M, int z_m,
    int size_x, int size_y, int size_z,
    float dt, float r1, float r2, float r3, float r4)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x + x_m;
  const int y = blockIdx.y * blockDim.y + threadIdx.y + y_m;
  const int z = blockIdx.z * blockDim.z + threadIdx.z + z_m;

  if (x > x_M || y > y_M || z > z_M) return;

  // Base index with halo offset (+4 just like OpenACC code)
  const int idx = (x + 4) * size_y * size_z + (y + 4) * size_z + (z + 4);

  // Neighbor strides
  const int sx = size_y * size_z;
  const int sy = size_z;

  // Read center
  const float u0 = u_t0[idx];

  // 4th-order second derivatives (same coefficients as OpenACC)
  const float r5 = -2.5f * u0;

  // X
  const float d2x = r5
    + (-8.33333333e-2f) * (u_t0[idx - 2*sx] + u_t0[idx + 2*sx])
    + (1.33333333f)     * (u_t0[idx - sx]   + u_t0[idx + sx]);

  // Y
  const float d2y = r5
    + (-8.33333333e-2f) * (u_t0[idx - 2*sy] + u_t0[idx + 2*sy])
    + (1.33333333f)     * (u_t0[idx - sy]   + u_t0[idx + sy]);

  // Z
  const float d2z = r5
    + (-8.33333333e-2f) * (u_t0[idx - 2] + u_t0[idx + 2])
    + (1.33333333f)     * (u_t0[idx - 1] + u_t0[idx + 1]);

  // Medium (read-only cache hint)
  const float mval = __ldg(&m[idx]);

  // Laplacian and time-term (r1 = 1/(dt*dt))
  const float lap = r2 * d2x + r3 * d2y + r4 * d2z;
  const float time_term = -2.0f * r1 * u0 + r1 * u_t1[idx];

  // IMPORTANT: dt*dt factor matches OpenACC exactly
  const float u_new = (dt * dt) * (lap - time_term * mval) / mval;

  u_t2[idx] = u_new;
}

// Trilinear source injection (same math as OpenACC; FP32)
__global__ void source_injection_kernel(
    float *__restrict__ u_t2,
    const float *__restrict__ m,
    const float *__restrict__ src,
    const float *__restrict__ src_coords, // [num_src, 3]
    int time, int p_src_M, int p_src_m,
    int x_M, int x_m, int y_M, int y_m, int z_M, int z_m,
    int size_x, int size_y, int size_z, int src_size1,
    float h_x, float h_y, float h_z, float o_x, float o_y, float o_z)
{
  const int p_src = blockIdx.x * blockDim.x + threadIdx.x + p_src_m;
  if (p_src > p_src_M) return;

  const float sx = src_coords[p_src * 3 + 0];
  const float sy = src_coords[p_src * 3 + 1];
  const float sz = src_coords[p_src * 3 + 2];

  const int posx = (int)floorf((-o_x + sx) / h_x);
  const int posy = (int)floorf((-o_y + sy) / h_y);
  const int posz = (int)floorf((-o_z + sz) / h_z);

  const float px = -floorf((-o_x + sx) / h_x) + (-o_x + sx) / h_x;
  const float py = -floorf((-o_y + sy) / h_y) + (-o_y + sy) / h_y;
  const float pz = -floorf((-o_z + sz) / h_z) + (-o_z + sz) / h_z;

  const float sval = src[time * src_size1 + p_src];

  // 8-point trilinear scatter
  for (int rsrcx = 0; rsrcx <= 1; ++rsrcx) {
    for (int rsrcy = 0; rsrcy <= 1; ++rsrcy) {
      for (int rsrcz = 0; rsrcz <= 1; ++rsrcz) {

        const int gx = rsrcx + posx;
        const int gy = rsrcy + posy;
        const int gz = rsrcz + posz;

        if (gx >= x_m - 1 && gx <= x_M + 1 &&
            gy >= y_m - 1 && gy <= y_M + 1 &&
            gz >= z_m - 1 && gz <= z_M + 1) {

          const float wx = (rsrcx ? px : (1.0f - px));
          const float wy = (rsrcy ? py : (1.0f - py));
          const float wz = (rsrcz ? pz : (1.0f - pz));
          const float w  = wx * wy * wz;

          const int idx = (gx + 4) * size_y * size_z + (gy + 4) * size_z + (gz + 4);
          const float mval = __ldg(&m[idx]);
          const float add  = 1.0e-2f * w * sval / mval;

          atomicAdd(&u_t2[idx], add);
        }
      }
    }
  }
}

// ---------------------------------------------
// Host entry point (matches OpenACC signature)
// ---------------------------------------------
extern "C" int Kernel_CUDA(
    struct dataobj *__restrict m_vec,
    struct dataobj *__restrict src_vec,
    struct dataobj *__restrict src_coords_vec,
    struct dataobj *__restrict u_vec,
    const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
    const float dt, const float h_x, const float h_y, const float h_z,
    const float o_x, const float o_y, const float o_z,
    const int p_src_M, const int p_src_m, const int time_M, const int time_m,
    const int deviceid, const int devicerm, struct profiler *timers)
{
  if (deviceid != -1) {
    CUDA_CHECK(cudaSetDevice(deviceid));
  }

  // Dimensions
  const int size_x = u_vec->size[1];
  const int size_y = u_vec->size[2];
  const int size_z = u_vec->size[3];

  const size_t plane      = (size_t)size_y * (size_t)size_z;
  const size_t volume     = (size_t)size_x * plane;
  const size_t u_elems    = (size_t)u_vec->size[0] * volume;      // 3 * volume
  const size_t m_elems    = (size_t)m_vec->size[0] * (size_t)m_vec->size[1] * (size_t)m_vec->size[2];
  const size_t src_elems  = (size_t)src_vec->size[0] * (size_t)src_vec->size[1];
  const size_t sco_elems  = (size_t)src_coords_vec->size[0] * (size_t)src_coords_vec->size[1];

  // Host pointers
  float *h_u           = (float*)u_vec->data;
  const float *h_m     = (const float*)m_vec->data;
  const float *h_src   = (const float*)src_vec->data;
  const float *h_sco   = (const float*)src_coords_vec->data;

  // Device pointers
  float *d_u = nullptr, *d_m = nullptr, *d_src = nullptr, *d_sco = nullptr;

  START(cuda_malloc)
  CUDA_CHECK(cudaMalloc(&d_u,   u_elems  * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_m,   m_elems  * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_src, src_elems* sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sco, sco_elems* sizeof(float)));
  STOP(cuda_malloc, timers)

  // H2D copies
  START(cuda_memcpy_h2d)
  CUDA_CHECK(cudaMemcpy(d_u,   h_u,   u_elems  * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_m,   h_m,   m_elems  * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_src, h_src, src_elems* sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sco, h_sco, sco_elems* sizeof(float), cudaMemcpyHostToDevice));
  STOP(cuda_memcpy_h2d, timers)

  // Precompute constants
  const float r1 = 1.0f / (dt * dt);
  const float r2 = 1.0f / (h_x * h_x);
  const float r3 = 1.0f / (h_y * h_y);
  const float r4 = 1.0f / (h_z * h_z);

  // Launch geometry
  dim3 block(8, 8, 8);
  dim3 grid(
    (x_M - x_m + 1 + block.x - 1) / block.x,
    (y_M - y_m + 1 + block.y - 1) / block.y,
    (z_M - z_m + 1 + block.z - 1) / block.z
  );

  dim3 block_src(128, 1, 1);
  dim3 grid_src(
    (p_src_M - p_src_m + 1 + block_src.x - 1) / block_src.x, 1, 1
  );

  START(kernel_time)

  for (int time = time_m; time <= time_M; ++time) {
    const int t0 =  time      % 3;
    const int t1 = (time + 2) % 3;
    const int t2 = (time + 1) % 3;

    float *u_t0 = d_u + (size_t)t0 * volume;
    float *u_t1 = d_u + (size_t)t1 * volume;
    float *u_t2 = d_u + (size_t)t2 * volume;

    // Main stencil
    wave_kernel<<<grid, block>>>(
      u_t2, u_t1, u_t0, d_m,
      x_M, x_m, y_M, y_m, z_M, z_m,
      size_x, size_y, size_z,
      dt, r1, r2, r3, r4
    );
    CUDA_CHECK(cudaGetLastError());

    // Source injection (if any)
    if (src_vec->size[0] * src_vec->size[1] > 0 && (p_src_M - p_src_m + 1) > 0) {
      source_injection_kernel<<<grid_src, block_src>>>(
        u_t2, d_m, d_src, d_sco,
        time, p_src_M, p_src_m,
        x_M, x_m, y_M, y_m, z_M, z_m,
        size_x, size_y, size_z, src_vec->size[1],
        h_x, h_y, h_z, o_x, o_y, o_z
      );
      CUDA_CHECK(cudaGetLastError());
    }

    // Kernels run on default stream; no per-step synchronize needed.
  }

  // Sync once after the loop so kernel_time measures device work only
  CUDA_CHECK(cudaDeviceSynchronize());
  STOP(kernel_time, timers)

  // D2H copy of results
  START(cuda_memcpy_d2h)
  CUDA_CHECK(cudaMemcpy(h_u, d_u, u_elems * sizeof(float), cudaMemcpyDeviceToHost));
  STOP(cuda_memcpy_d2h, timers)

  if (devicerm) {
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_sco));
  }

  return 0;
}
