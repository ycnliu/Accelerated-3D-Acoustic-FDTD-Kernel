// fdtd_optimized_advanced.cu - Highly optimized CUDA implementation
// Optimizations:
// 1. Shared memory for better cache efficiency
// 2. Coalesced memory access patterns
// 3. Fused kernel (stencil + source injection)
// 4. Better register utilization
// 5. Thread block optimization

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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
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
  double section0{};
  double section1{};
  double cuda_malloc{};
  double cuda_memcpy{};
  double cuda_memcpy_h2d{};
  double cuda_memcpy_d2h{};
  double kernel_time{};
  double conversion_time{};
};

// Optimized constants
#define BLOCK_X 16
#define BLOCK_Y 8
#define BLOCK_Z 4
#define HALO 2

// Fused kernel: wave equation + source injection in one pass
__global__ void wave_kernel_fused_optimized(
    float *__restrict__ u_t2,
    const float *__restrict__ u_t1,
    const float *__restrict__ u_t0,
    const float *__restrict__ m,
    const float *__restrict__ src,
    const float *__restrict__ src_coords,
    int time, int p_src_M, int p_src_m,
    int x_M, int x_m, int y_M, int y_m, int z_M, int z_m,
    int size_x, int size_y, int size_z, int src_size1,
    float dt, float r1, float r2, float r3, float r4,
    float h_x, float h_y, float h_z, float o_x, float o_y, float o_z)
{
    // Shared memory for better cache efficiency
    __shared__ float s_u0[BLOCK_Z + 2*HALO][BLOCK_Y + 2*HALO][BLOCK_X + 2*HALO];
    __shared__ float s_m[BLOCK_Z][BLOCK_Y][BLOCK_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int x = blockIdx.x * BLOCK_X + tx + x_m;
    const int y = blockIdx.y * BLOCK_Y + ty + y_m;
    const int z = blockIdx.z * BLOCK_Z + tz + z_m;

    // Boundary check
    if (x > x_M || y > y_M || z > z_M) return;

    const int idx = (x + 4) * size_y * size_z + (y + 4) * size_z + (z + 4);
    const int sx = size_y * size_z;
    const int sy = size_z;

    // Load data into shared memory with halo
    // Load center point
    if (tx < BLOCK_X && ty < BLOCK_Y && tz < BLOCK_Z) {
        s_u0[tz + HALO][ty + HALO][tx + HALO] = u_t0[idx];
        s_m[tz][ty][tx] = m[idx];
    }

    // Load halo regions (collaborative loading)
    // X-direction halos
    if (tx < HALO && x - HALO >= x_m) {
        int halo_idx = idx - HALO * sx;
        s_u0[tz + HALO][ty + HALO][tx] = u_t0[halo_idx];
    }
    if (tx >= BLOCK_X - HALO && x + HALO <= x_M) {
        int halo_idx = idx + HALO * sx;
        s_u0[tz + HALO][ty + HALO][tx + 2*HALO] = u_t0[halo_idx];
    }

    // Y-direction halos
    if (ty < HALO && y - HALO >= y_m) {
        int halo_idx = idx - HALO * sy;
        s_u0[tz + HALO][ty][tx + HALO] = u_t0[halo_idx];
    }
    if (ty >= BLOCK_Y - HALO && y + HALO <= y_M) {
        int halo_idx = idx + HALO * sy;
        s_u0[tz + HALO][ty + 2*HALO][tx + HALO] = u_t0[halo_idx];
    }

    // Z-direction halos
    if (tz < HALO && z - HALO >= z_m) {
        int halo_idx = idx - HALO;
        s_u0[tz][ty + HALO][tx + HALO] = u_t0[halo_idx];
    }
    if (tz >= BLOCK_Z - HALO && z + HALO <= z_M) {
        int halo_idx = idx + HALO;
        s_u0[tz + 2*HALO][ty + HALO][tx + HALO] = u_t0[halo_idx];
    }

    __syncthreads();

    // Compute stencil using shared memory
    const float u0 = s_u0[tz + HALO][ty + HALO][tx + HALO];
    const float r5 = -2.5f * u0;

    // 4th-order stencil from shared memory
    const float d2x = r5
        + (-8.33333333e-2f) * (s_u0[tz + HALO][ty + HALO][tx] + s_u0[tz + HALO][ty + HALO][tx + 2*HALO])
        + (1.33333333f)     * (s_u0[tz + HALO][ty + HALO][tx + 1] + s_u0[tz + HALO][ty + HALO][tx + 2*HALO - 1]);

    const float d2y = r5
        + (-8.33333333e-2f) * (s_u0[tz + HALO][ty][tx + HALO] + s_u0[tz + HALO][ty + 2*HALO][tx + HALO])
        + (1.33333333f)     * (s_u0[tz + HALO][ty + 1][tx + HALO] + s_u0[tz + HALO][ty + 2*HALO - 1][tx + HALO]);

    const float d2z = r5
        + (-8.33333333e-2f) * (s_u0[tz][ty + HALO][tx + HALO] + s_u0[tz + 2*HALO][ty + HALO][tx + HALO])
        + (1.33333333f)     * (s_u0[tz + 1][ty + HALO][tx + HALO] + s_u0[tz + 2*HALO - 1][ty + HALO][tx + HALO]);

    // Medium value from shared memory
    const float mval = s_m[tz][ty][tx];

    // Time step calculation
    const float lap = r2 * d2x + r3 * d2y + r4 * d2z;
    const float time_term = -2.0f * r1 * u0 + r1 * u_t1[idx];
    float u_new = (dt * dt) * (lap - time_term * mval) / mval;

    // Fused source injection - check if this point is affected by any source
    if (src && p_src_M >= p_src_m) {
        for (int p_src = p_src_m; p_src <= p_src_M; ++p_src) {
            const float sx = src_coords[p_src * 3 + 0];
            const float sy = src_coords[p_src * 3 + 1];
            const float sz = src_coords[p_src * 3 + 2];

            const int posx = (int)floorf((-o_x + sx) / h_x);
            const int posy = (int)floorf((-o_y + sy) / h_y);
            const int posz = (int)floorf((-o_z + sz) / h_z);

            // Check if current thread is within source influence
            const int gx = x - 4;  // Convert from halo coordinates to interior
            const int gy = y - 4;
            const int gz = z - 4;

            if (gx >= posx && gx <= posx + 1 &&
                gy >= posy && gy <= posy + 1 &&
                gz >= posz && gz <= posz + 1) {

                const float px = -floorf((-o_x + sx) / h_x) + (-o_x + sx) / h_x;
                const float py = -floorf((-o_y + sy) / h_y) + (-o_y + sy) / h_y;
                const float pz = -floorf((-o_z + sz) / h_z) + (-o_z + sz) / h_z;

                const int rsrcx = gx - posx;
                const int rsrcy = gy - posy;
                const int rsrcz = gz - posz;

                const float wx = (rsrcx ? px : (1.0f - px));
                const float wy = (rsrcy ? py : (1.0f - py));
                const float wz = (rsrcz ? pz : (1.0f - pz));
                const float w = wx * wy * wz;

                const float sval = src[time * src_size1 + p_src];
                const float add = 1.0e-2f * w * sval / mval;
                u_new += add;
            }
        }
    }

    // Write result with coalesced access
    u_t2[idx] = u_new;
}

// Host entry point
extern "C" int Kernel_Optimized_Advanced(
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
    const size_t u_elems    = (size_t)u_vec->size[0] * volume;
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

    // Optimized launch geometry
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (x_M - x_m + 1 + BLOCK_X - 1) / BLOCK_X,
        (y_M - y_m + 1 + BLOCK_Y - 1) / BLOCK_Y,
        (z_M - z_m + 1 + BLOCK_Z - 1) / BLOCK_Z
    );

    START(kernel_time)

    for (int time = time_m; time <= time_M; ++time) {
        const int t0 =  time      % 3;
        const int t1 = (time + 2) % 3;
        const int t2 = (time + 1) % 3;

        float *u_t0 = d_u + (size_t)t0 * volume;
        float *u_t1 = d_u + (size_t)t1 * volume;
        float *u_t2 = d_u + (size_t)t2 * volume;

        // Single fused kernel call
        wave_kernel_fused_optimized<<<grid, block>>>(
            u_t2, u_t1, u_t0, d_m, d_src, d_sco,
            time, p_src_M, p_src_m,
            x_M, x_m, y_M, y_m, z_M, z_m,
            size_x, size_y, size_z, src_vec->size[1],
            dt, r1, r2, r3, r4,
            h_x, h_y, h_z, o_x, o_y, o_z
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    STOP(kernel_time, timers)

    // D2H copy
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