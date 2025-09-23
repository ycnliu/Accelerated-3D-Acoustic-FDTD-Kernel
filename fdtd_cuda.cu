#include <cuda_runtime.h>
#include <sys/time.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

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
    double kernel_time{};
    double conversion_time{};  // required by CUDA/OpenACC impls
};

__global__ void wave_kernel(
    float* __restrict__ u_t2,
    const float* __restrict__ u_t1,
    const float* __restrict__ u_t0,
    const float* __restrict__ m,
    int x_M, int x_m, int y_M, int y_m, int z_M, int z_m,
    int size_x, int size_y, int size_z,
    float dt, float h_x, float h_y, float h_z,
    float r1, float r2, float r3, float r4
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + x_m;
    int y = blockIdx.y * blockDim.y + threadIdx.y + y_m;
    int z = blockIdx.z * blockDim.z + threadIdx.z + z_m;

    if (x <= x_M && y <= y_M && z <= z_M) {
        int idx = (x + 4) * size_y * size_z + (y + 4) * size_z + (z + 4);

        // 4th order finite difference stencil
        float r5 = -2.5f * u_t0[idx];

        // X direction stencil
        float d2u_dx2 = r5 +
            (-8.33333333e-2f) * (u_t0[idx - 2*size_y*size_z] + u_t0[idx + 2*size_y*size_z]) +
            1.333333330f * (u_t0[idx - size_y*size_z] + u_t0[idx + size_y*size_z]);

        // Y direction stencil
        float d2u_dy2 = r5 +
            (-8.33333333e-2f) * (u_t0[idx - 2*size_z] + u_t0[idx + 2*size_z]) +
            1.333333330f * (u_t0[idx - size_z] + u_t0[idx + size_z]);

        // Z direction stencil
        float d2u_dz2 = r5 +
            (-8.33333333e-2f) * (u_t0[idx - 2] + u_t0[idx + 2]) +
            1.333333330f * (u_t0[idx - 1] + u_t0[idx + 1]);

        float m_val = __ldg(&m[idx]);
        float laplacian = r2 * d2u_dx2 + r3 * d2u_dy2 + r4 * d2u_dz2;
        float time_term = -2.0f * r1 * u_t0[idx] + r1 * u_t1[idx];

        u_t2[idx] = (dt * dt * (laplacian - time_term * m_val)) / m_val;
    }
}

__global__ void source_injection_kernel(
    float* __restrict__ u_t2,
    const float* __restrict__ m,
    const float* __restrict__ src,
    const float* __restrict__ src_coords,
    int time, int p_src_M, int p_src_m,
    int x_M, int x_m, int y_M, int y_m, int z_M, int z_m,
    int size_x, int size_y, int size_z, int src_size1,
    float h_x, float h_y, float h_z, float o_x, float o_y, float o_z
) {
    int p_src = blockIdx.x * blockDim.x + threadIdx.x + p_src_m;
    int rsrcx = blockIdx.y * blockDim.y + threadIdx.y;
    int rsrcy = blockIdx.z * blockDim.z + threadIdx.z;

    if (p_src <= p_src_M && rsrcx <= 1 && rsrcy <= 1) {
        for (int rsrcz = 0; rsrcz <= 1; rsrcz++) {
            int posx = (int)floorf((-o_x + src_coords[p_src * 3 + 0]) / h_x);
            int posy = (int)floorf((-o_y + src_coords[p_src * 3 + 1]) / h_y);
            int posz = (int)floorf((-o_z + src_coords[p_src * 3 + 2]) / h_z);

            float px = -floorf((-o_x + src_coords[p_src * 3 + 0]) / h_x) + (-o_x + src_coords[p_src * 3 + 0]) / h_x;
            float py = -floorf((-o_y + src_coords[p_src * 3 + 1]) / h_y) + (-o_y + src_coords[p_src * 3 + 1]) / h_y;
            float pz = -floorf((-o_z + src_coords[p_src * 3 + 2]) / h_z) + (-o_z + src_coords[p_src * 3 + 2]) / h_z;

            if (rsrcx + posx >= x_m - 1 && rsrcy + posy >= y_m - 1 && rsrcz + posz >= z_m - 1 &&
                rsrcx + posx <= x_M + 1 && rsrcy + posy <= y_M + 1 && rsrcz + posz <= z_M + 1) {

                float weight = (rsrcx * px + (1 - rsrcx) * (1 - px)) *
                              (rsrcy * py + (1 - rsrcy) * (1 - py)) *
                              (rsrcz * pz + (1 - rsrcz) * (1 - pz));

                int m_idx = (posx + 4) * size_y * size_z + (posy + 4) * size_z + (posz + 4);
                int u_idx = (rsrcx + posx + 4) * size_y * size_z + (rsrcy + posy + 4) * size_z + (rsrcz + posz + 4);

                float r0 = 1.0e-2f * weight * __ldg(&src[time * src_size1 + p_src]) / __ldg(&m[m_idx]);
                atomicAdd(&u_t2[u_idx], r0);
            }
        }
    }
}

extern "C" int Kernel_CUDA(
    struct dataobj *__restrict m_vec,
    struct dataobj *__restrict src_vec,
    struct dataobj *__restrict src_coords_vec,
    struct dataobj *__restrict u_vec,
    const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
    const float dt, const float h_x, const float h_y, const float h_z,
    const float o_x, const float o_y, const float o_z,
    const int p_src_M, const int p_src_m, const int time_M, const int time_m,
    const int deviceid, const int devicerm, struct profiler *timers
) {
    // Set device
    if (deviceid != -1) {
        CUDA_CHECK(cudaSetDevice(deviceid));
    }

    // Get array dimensions
    int size_x = u_vec->size[1];
    int size_y = u_vec->size[2];
    int size_z = u_vec->size[3];
    int u_size = u_vec->size[0] * size_x * size_y * size_z;
    int m_size = m_vec->size[0] * m_vec->size[1] * m_vec->size[2];
    int src_size = src_vec->size[0] * src_vec->size[1];
    int src_coords_size = src_coords_vec->size[0] * src_coords_vec->size[1];

    // Host arrays
    float *h_u = (float*)u_vec->data;
    float *h_m = (float*)m_vec->data;
    float *h_src = (float*)src_vec->data;
    float *h_src_coords = (float*)src_coords_vec->data;

    // Device arrays
    float *d_u, *d_m, *d_src, *d_src_coords;

    START(cuda_malloc)
    CUDA_CHECK(cudaMalloc(&d_u, u_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, m_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src, src_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_coords, src_coords_size * sizeof(float)));
    STOP(cuda_malloc, timers)

    START(cuda_memcpy)
    CUDA_CHECK(cudaMemcpy(d_u, h_u, u_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, h_m, m_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, src_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_coords, h_src_coords, src_coords_size * sizeof(float), cudaMemcpyHostToDevice));
    STOP(cuda_memcpy, timers)

    // Pre-compute constants
    float r1 = 1.0f / (dt * dt);
    float r2 = 1.0f / (h_x * h_x);
    float r3 = 1.0f / (h_y * h_y);
    float r4 = 1.0f / (h_z * h_z);

    // Grid dimensions for wave kernel
    dim3 block_wave(8, 8, 8);
    dim3 grid_wave(
        (x_M - x_m + 1 + block_wave.x - 1) / block_wave.x,
        (y_M - y_m + 1 + block_wave.y - 1) / block_wave.y,
        (z_M - z_m + 1 + block_wave.z - 1) / block_wave.z
    );

    // Grid dimensions for source injection
    dim3 block_src(32, 2, 2);
    dim3 grid_src(
        (p_src_M - p_src_m + 1 + block_src.x - 1) / block_src.x,
        1, 1
    );

    START(kernel_time)

    // Time stepping loop
    for (int time = time_m; time <= time_M; time++) {
        int t0 = time % 3;
        int t1 = (time + 2) % 3;
        int t2 = (time + 1) % 3;

        // Pointers to current time levels
        float *u_t0 = d_u + t0 * size_x * size_y * size_z;
        float *u_t1 = d_u + t1 * size_x * size_y * size_z;
        float *u_t2 = d_u + t2 * size_x * size_y * size_z;

        // Wave propagation kernel
        wave_kernel<<<grid_wave, block_wave>>>(
            u_t2, u_t1, u_t0, d_m,
            x_M, x_m, y_M, y_m, z_M, z_m,
            size_x, size_y, size_z,
            dt, h_x, h_y, h_z, r1, r2, r3, r4
        );

        CUDA_CHECK(cudaGetLastError());

        // Source injection kernel
        if (src_vec->size[0] * src_vec->size[1] > 0 && p_src_M - p_src_m + 1 > 0) {
            source_injection_kernel<<<grid_src, block_src>>>(
                u_t2, d_m, d_src, d_src_coords,
                time, p_src_M, p_src_m,
                x_M, x_m, y_M, y_m, z_M, z_m,
                size_x, size_y, size_z, src_vec->size[1],
                h_x, h_y, h_z, o_x, o_y, o_z
            );

            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    STOP(kernel_time, timers)

    // Copy result back
    struct timeval start_copyback, end_copyback;
    gettimeofday(&start_copyback, NULL);
    CUDA_CHECK(cudaMemcpy(h_u, d_u, u_size * sizeof(float), cudaMemcpyDeviceToHost));
    gettimeofday(&end_copyback, NULL);
    timers->cuda_memcpy += (double)(end_copyback.tv_sec-start_copyback.tv_sec)+(double)(end_copyback.tv_usec-start_copyback.tv_usec)/1000000;

    // Cleanup
    if (devicerm) {
        cudaFree(d_u);
        cudaFree(d_m);
        cudaFree(d_src);
        cudaFree(d_src_coords);
    }

    return 0;
}