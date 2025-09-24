#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <sys/time.h>

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

// 3D indexing macros for u (4D: time, x, y, z)
#define U(t, x, y, z) u[(t) * x_size * y_size * z_size + (x) * y_size * z_size + (y) * z_size + (z)]
#define M(x, y, z) m[(x) * y_size * z_size + (y) * z_size + (z)]

// Single-step temporal blocking kernel with shared memory (safe, correct)
__global__ void wave_kernel_temporal_blocking_1step(
    float* __restrict__ u,
    const float* __restrict__ m,
    int x_size, int y_size, int z_size,
    float dt, float h_x, float h_y, float h_z,
    int t_start_base, int x_start, int y_start, int z_start
) {
    const int BLOCK_X = 8;
    const int BLOCK_Y = 8;
    const int BLOCK_Z = 6;
    const int HALO = 2;

    // Shared memory for temporal blocking
    __shared__ float s_u_prev[BLOCK_Z + 2*HALO][BLOCK_Y + 2*HALO][BLOCK_X + 2*HALO];
    __shared__ float s_u_curr[BLOCK_Z + 2*HALO][BLOCK_Y + 2*HALO][BLOCK_X + 2*HALO];
    __shared__ float s_m[BLOCK_Z + 2*HALO][BLOCK_Y + 2*HALO][BLOCK_X + 2*HALO];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Time indices
    int t0 = (t_start_base) % 3;
    int t1 = (t_start_base + 2) % 3;
    int t2 = (t_start_base + 1) % 3;

    // Precompute coefficients
    float r1 = 1.0f / (dt * dt);
    float r2 = 1.0f / (h_x * h_x);
    float r3 = 1.0f / (h_y * h_y);
    float r4 = 1.0f / (h_z * h_z);

    // Cooperative tile loading - each thread loads disjoint subset
    const int SX = BLOCK_X + 2*HALO;
    const int SY = BLOCK_Y + 2*HALO;
    const int SZ = BLOCK_Z + 2*HALO;

    for (int lz = threadIdx.z; lz < SZ; lz += blockDim.z) {
        int gz_load = (blockIdx.z*BLOCK_Z + lz - HALO) + z_start + 4;
        for (int ly = threadIdx.y; ly < SY; ly += blockDim.y) {
            int gy_load = (blockIdx.y*BLOCK_Y + ly - HALO) + y_start + 4;
            for (int lx = threadIdx.x; lx < SX; lx += blockDim.x) {
                int gx_load = (blockIdx.x*BLOCK_X + lx - HALO) + x_start + 4;
                float u_prev = 0.f, u_curr = 0.f, mval = 1.f;
                if (gx_load>=0 && gx_load<x_size &&
                    gy_load>=0 && gy_load<y_size &&
                    gz_load>=0 && gz_load<z_size) {
                    u_prev = U(t1, gx_load, gy_load, gz_load);
                    u_curr = U(t0, gx_load, gy_load, gz_load);
                    mval   = __ldg(&M(gx_load, gy_load, gz_load));
                }
                s_u_prev[lz][ly][lx] = u_prev;
                s_u_curr[lz][ly][lx] = u_curr;
                s_m     [lz][ly][lx] = mval;
            }
        }
    }
    __syncthreads();

    // Compute t2 on core region
    if (tx < BLOCK_X && ty < BLOCK_Y && tz < BLOCK_Z) {
        const int sx = tx + HALO, sy = ty + HALO, sz = tz + HALO;

        float center = s_u_curr[sz][sy][sx];
        float r5 = -2.5f * center;

        // 4th-order stencil in X direction
        float d2dx2 = r5 + (-8.33333333e-2f) * (s_u_curr[sz][sy][sx-2] + s_u_curr[sz][sy][sx+2]) +
                            1.333333330f * (s_u_curr[sz][sy][sx-1] + s_u_curr[sz][sy][sx+1]);

        // 4th-order stencil in Y direction
        float d2dy2 = r5 + (-8.33333333e-2f) * (s_u_curr[sz][sy-2][sx] + s_u_curr[sz][sy+2][sx]) +
                            1.333333330f * (s_u_curr[sz][sy-1][sx] + s_u_curr[sz][sy+1][sx]);

        // 4th-order stencil in Z direction
        float d2dz2 = r5 + (-8.33333333e-2f) * (s_u_curr[sz-2][sy][sx] + s_u_curr[sz+2][sy][sx]) +
                            1.333333330f * (s_u_curr[sz-1][sy][sx] + s_u_curr[sz+1][sy][sx]);

        float m_val = s_m[sz][sy][sx];
        float laplacian = r2 * d2dx2 + r3 * d2dy2 + r4 * d2dz2;
        float time_term = -2.0f * r1 * center + r1 * s_u_prev[sz][sy][sx];

        float u_t2_core = (dt * dt * (laplacian - time_term * m_val)) / m_val;

        // Global coordinates
        const int gx = blockIdx.x*BLOCK_X + tx + x_start + 4;
        const int gy = blockIdx.y*BLOCK_Y + ty + y_start + 4;
        const int gz = blockIdx.z*BLOCK_Z + tz + z_start + 4;

        // Guard writes to valid interior
        if (gx>=4 && gx<x_size-4 && gy>=4 && gy<y_size-4 && gz>=4 && gz<z_size-4) {
            U(t2, gx, gy, gz) = u_t2_core;
        }
    }
}

// Source injection kernel (unchanged from optimized version)
__global__ void source_kernel(float* __restrict__ u,
                             const float* __restrict__ src,
                             const float* __restrict__ src_coords,
                             const float* __restrict__ m,
                             int x_size, int y_size, int z_size,
                             int time, int p_src_m, int p_src_M,
                             float h_x, float h_y, float h_z,
                             float o_x, float o_y, float o_z) {
    int p_src = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_src > p_src_M - p_src_m) return;

    p_src += p_src_m;

    int t2 = (time + 1) % 3;

    for (int rsrcx = 0; rsrcx <= 1; rsrcx++) {
        for (int rsrcy = 0; rsrcy <= 1; rsrcy++) {
            for (int rsrcz = 0; rsrcz <= 1; rsrcz++) {
                int posx = static_cast<int>(floorf((-o_x + src_coords[p_src * 3 + 0]) / h_x));
                int posy = static_cast<int>(floorf((-o_y + src_coords[p_src * 3 + 1]) / h_y));
                int posz = static_cast<int>(floorf((-o_z + src_coords[p_src * 3 + 2]) / h_z));

                float px = -floorf((-o_x + src_coords[p_src * 3 + 0]) / h_x) + (-o_x + src_coords[p_src * 3 + 0]) / h_x;
                float py = -floorf((-o_y + src_coords[p_src * 3 + 1]) / h_y) + (-o_y + src_coords[p_src * 3 + 1]) / h_y;
                float pz = -floorf((-o_z + src_coords[p_src * 3 + 2]) / h_z) + (-o_z + src_coords[p_src * 3 + 2]) / h_z;

                int gx = rsrcx + posx + 4;
                int gy = rsrcy + posy + 4;
                int gz = rsrcz + posz + 4;

                if (gx >= 3 && gx < x_size - 3 && gy >= 3 && gy < y_size - 3 && gz >= 3 && gz < z_size - 3) {
                    float weight = (rsrcx * px + (1 - rsrcx) * (1 - px)) *
                                  (rsrcy * py + (1 - rsrcy) * (1 - py)) *
                                  (rsrcz * pz + (1 - rsrcz) * (1 - pz));

                    float contribution = 1.0e-2f * weight * src[time * (p_src_M - p_src_m + 1) + (p_src - p_src_m)] /
                                       __ldg(&M(gx, gy, gz));

                    atomicAdd(&U(t2, gx, gy, gz), contribution);
                }
            }
        }
    }
}

extern "C" int Kernel_Temporal_Blocking(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec,
                                        struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec,
                                        const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
                                        const float dt, const float h_x, const float h_y, const float h_z,
                                        const float o_x, const float o_y, const float o_z,
                                        const int p_src_M, const int p_src_m, const int time_M, const int time_m,
                                        const int deviceid, const int devicerm, struct profiler *timers) {

    // Set device
    if (deviceid != -1) {
        CUDA_CHECK(cudaSetDevice(deviceid));
    }

    // GPU memory allocation and setup
    float *d_u, *d_m, *d_src, *d_src_coords;

    int x_size = u_vec->size[1];
    int y_size = u_vec->size[2];
    int z_size = u_vec->size[3];

    size_t u_bytes = u_vec->size[0] * x_size * y_size * z_size * sizeof(float);
    size_t m_bytes = m_vec->size[0] * m_vec->size[1] * m_vec->size[2] * sizeof(float);
    size_t src_bytes = src_vec->size[0] * src_vec->size[1] * sizeof(float);
    size_t src_coords_bytes = src_coords_vec->size[0] * src_coords_vec->size[1] * sizeof(float);

    START(cuda_malloc)
    CUDA_CHECK(cudaMalloc(&d_u, u_bytes));
    CUDA_CHECK(cudaMalloc(&d_m, m_bytes));
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    CUDA_CHECK(cudaMalloc(&d_src_coords, src_coords_bytes));
    STOP(cuda_malloc, timers)

    START(cuda_memcpy)
    CUDA_CHECK(cudaMemcpy(d_u, u_vec->data, u_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, m_vec->data, m_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src, src_vec->data, src_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_coords, src_coords_vec->data, src_coords_bytes, cudaMemcpyHostToDevice));
    STOP(cuda_memcpy, timers)

    // Grid configuration for temporal blocking
    dim3 blockSize(8, 8, 6);
    dim3 gridSize((x_M - x_m + 1 + blockSize.x - 1) / blockSize.x,
                  (y_M - y_m + 1 + blockSize.y - 1) / blockSize.y,
                  (z_M - z_m + 1 + blockSize.z - 1) / blockSize.z);

    dim3 src_blockSize(64);
    dim3 src_gridSize((p_src_M - p_src_m + src_blockSize.x) / src_blockSize.x);

    START(kernel_time)

    // Main temporal blocking loop - process 1 time step per iteration (safe)
    for (int time = time_m; time <= time_M; time++) {

        // Launch temporal blocking kernel (processes 1 time step)
        wave_kernel_temporal_blocking_1step<<<gridSize, blockSize>>>(
            d_u, d_m, x_size, y_size, z_size, dt, h_x, h_y, h_z,
            time, x_m, y_m, z_m);

        CUDA_CHECK(cudaGetLastError());

        // Source injection
        if (src_vec->size[0] * src_vec->size[1] > 0 && p_src_M - p_src_m + 1 > 0) {
            source_kernel<<<src_gridSize, src_blockSize>>>(
                d_u, d_src, d_src_coords, d_m, x_size, y_size, z_size,
                time, p_src_m, p_src_M, h_x, h_y, h_z, o_x, o_y, o_z);

            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    STOP(kernel_time, timers)

    // Copy results back
    struct timeval start_copyback, end_copyback;
    gettimeofday(&start_copyback, NULL);
    CUDA_CHECK(cudaMemcpy(u_vec->data, d_u, u_bytes, cudaMemcpyDeviceToHost));
    gettimeofday(&end_copyback, NULL);
    timers->cuda_memcpy += (double)(end_copyback.tv_sec-start_copyback.tv_sec)
                         + (double)(end_copyback.tv_usec-start_copyback.tv_usec)/1e6;

    // Cleanup
    if (devicerm) {
        cudaFree(d_u);
        cudaFree(d_m);
        cudaFree(d_src);
        cudaFree(d_src_coords);
    }

    return 0;
}