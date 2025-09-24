#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
    double section0;
    double section1;
    double cuda_malloc;
    double cuda_memcpy;
    double cuda_memcpy_h2d;
    double cuda_memcpy_d2h;
    double kernel_time;
    double conversion_time;
};

// Device kernel for FP32 to FP16 conversion
__global__ void convert_fp32_to_fp16_kernel(
    const float *__restrict__ input_fp32,
    __half *__restrict__ output_fp16,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output_fp16[idx] = __float2half(input_fp32[idx]);
    }
}

// Simple mixed-precision wave kernel
__global__ void wave_kernel_mixed_precision(
    __half *__restrict__ u_t2,
    __half *__restrict__ u_t1,
    __half *__restrict__ u_t0,
    float *__restrict__ m,  // Keep m in FP32 for accuracy
    int x_M, int x_m, int y_M, int y_m, int z_M, int z_m,
    int size_x, int size_y, int size_z,
    float dt, float h_x, float h_y, float h_z
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + x_m;
    int y = blockIdx.y * blockDim.y + threadIdx.y + y_m;
    int z = blockIdx.z * blockDim.z + threadIdx.z + z_m;

    if (x <= x_M && y <= y_M && z <= z_M) {
        int idx = (x + 4) * size_y * size_z + (y + 4) * size_z + (z + 4);

        // Stencil computation - bounds safety ensured by halo regions
        // Load values using read-only cache and convert to float for computation
        float u_center = __half2float(__ldg(&u_t0[idx]));

        // 4th order finite difference stencil - load via read-only cache and convert from half
        float u_xm2 = __half2float(__ldg(&u_t0[idx - 2*size_y*size_z]));
        float u_xm1 = __half2float(__ldg(&u_t0[idx - size_y*size_z]));
        float u_xp1 = __half2float(__ldg(&u_t0[idx + size_y*size_z]));
        float u_xp2 = __half2float(__ldg(&u_t0[idx + 2*size_y*size_z]));

        float u_ym2 = __half2float(__ldg(&u_t0[idx - 2*size_z]));
        float u_ym1 = __half2float(__ldg(&u_t0[idx - size_z]));
        float u_yp1 = __half2float(__ldg(&u_t0[idx + size_z]));
        float u_yp2 = __half2float(__ldg(&u_t0[idx + 2*size_z]));

        float u_zm2 = __half2float(__ldg(&u_t0[idx - 2]));
        float u_zm1 = __half2float(__ldg(&u_t0[idx - 1]));
        float u_zp1 = __half2float(__ldg(&u_t0[idx + 1]));
        float u_zp2 = __half2float(__ldg(&u_t0[idx + 2]));

        // Load previous time step and medium property via read-only cache
        float u_prev = __half2float(__ldg(&u_t1[idx]));
        float m_val = __ldg(&m[idx]);

        // 4th order finite differences
        float d2u_dx2 = (-u_xm2 + 16.0f*u_xm1 - 30.0f*u_center + 16.0f*u_xp1 - u_xp2) / (12.0f * h_x * h_x);
        float d2u_dy2 = (-u_ym2 + 16.0f*u_ym1 - 30.0f*u_center + 16.0f*u_yp1 - u_yp2) / (12.0f * h_y * h_y);
        float d2u_dz2 = (-u_zm2 + 16.0f*u_zm1 - 30.0f*u_center + 16.0f*u_zp1 - u_zp2) / (12.0f * h_z * h_z);

        // Wave equation: u_new = 2*u_center - u_prev + (dt^2/m) * (m * laplacian)
        float laplacian = d2u_dx2 + d2u_dy2 + d2u_dz2;
        float u_new = 2.0f * u_center - u_prev + (dt * dt / m_val) * (m_val * laplacian);

        // Convert result back to half and store
        u_t2[idx] = __float2half(u_new);
    }
}

// Source injection kernel for mixed precision
__global__ void source_injection_kernel_mixed(
    __half *__restrict__ u_t2,
    float *__restrict__ m,
    float *__restrict__ src,
    float *__restrict__ src_coords,
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
            // Load source coordinates via read-only cache
            float src_x = __ldg(&src_coords[p_src * 3 + 0]);
            float src_y = __ldg(&src_coords[p_src * 3 + 1]);
            float src_z = __ldg(&src_coords[p_src * 3 + 2]);

            int posx = (int)floorf((-o_x + src_x) / h_x);
            int posy = (int)floorf((-o_y + src_y) / h_y);
            int posz = (int)floorf((-o_z + src_z) / h_z);

            float px = -floorf((-o_x + src_x) / h_x) + (-o_x + src_x) / h_x;
            float py = -floorf((-o_y + src_y) / h_y) + (-o_y + src_y) / h_y;
            float pz = -floorf((-o_z + src_z) / h_z) + (-o_z + src_z) / h_z;

            if (rsrcx + posx >= x_m - 1 && rsrcy + posy >= y_m - 1 && rsrcz + posz >= z_m - 1 &&
                rsrcx + posx <= x_M + 1 && rsrcy + posy <= y_M + 1 && rsrcz + posz <= z_M + 1) {

                float weight = (rsrcx * px + (1 - rsrcx) * (1 - px)) *
                              (rsrcy * py + (1 - rsrcy) * (1 - py)) *
                              (rsrcz * pz + (1 - rsrcz) * (1 - pz));

                int m_idx = (posx + 4) * size_y * size_z + (posy + 4) * size_z + (posz + 4);
                int u_idx = (rsrcx + posx + 4) * size_y * size_z + (rsrcy + posy + 4) * size_z + (rsrcz + posz + 4);

                float r0 = 1.0e-2f * weight * __ldg(&src[time * src_size1 + p_src]) / __ldg(&m[m_idx]);

                // Add to existing value using atomic operation for race-free source injection
                // Bounds check for safe memory access
                if (u_idx < size_x * size_y * size_z) {
                    // Use float* alias for atomic operation since __half doesn't support atomics directly
                    float* u_t2_float = (float*)u_t2;
                    int half_idx = u_idx / 2;
                    int half_offset = u_idx % 2;

                    // Ensure half_idx is within bounds for half2 access
                    int max_half2_idx = (size_x * size_y * size_z + 1) / 2;
                    if (half_idx < max_half2_idx) {
                        // Read current half2 pair, update our half, write back atomically
                        float old_val, new_val;
                        do {
                            old_val = u_t2_float[half_idx];
                            half2 pair = __floats2half2_rn(0.0f, 0.0f);  // Initialize
                            // Convert float to half2 manually
                            unsigned int old_uint = __float_as_uint(old_val);
                            pair.x = __ushort_as_half((unsigned short)(old_uint & 0xFFFF));
                            pair.y = __ushort_as_half((unsigned short)((old_uint >> 16) & 0xFFFF));

                            if (half_offset == 0) {
                                pair.x = __float2half(__half2float(pair.x) + r0);
                            } else {
                                pair.y = __float2half(__half2float(pair.y) + r0);
                            }

                            // Convert half2 back to float manually
                            unsigned int new_uint = ((unsigned int)__half_as_ushort(pair.y) << 16) |
                                                   (unsigned int)__half_as_ushort(pair.x);
                            new_val = __uint_as_float(new_uint);
                        } while (atomicCAS((unsigned int*)&u_t2_float[half_idx],
                                          __float_as_uint(old_val),
                                          __float_as_uint(new_val)) != __float_as_uint(old_val));
                    }
                }
            }
        }
    }
}

extern "C" int Kernel_Mixed_Precision(
    struct dataobj *__restrict m_vec,
    struct dataobj *__restrict src_vec,
    struct dataobj *__restrict src_coords_vec,
    struct dataobj *__restrict u_vec,
    const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
    const float dt, const float h_x, const float h_y, const float h_z,
    const float o_x, const float o_y, const float o_z,
    const int p_src_M, const int p_src_m, const int time_M, const int time_m,
    const int deviceid, const int devicerm, struct profiler *timers, int use_vectorized = 0
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
    __half *d_u_fp16;
    float *d_m_fp32, *d_src_fp32, *d_src_coords_fp32;

    START(cuda_malloc)
    // Allocate FP16 arrays for wavefield with half2 alignment, FP32 for others
    // Ensure u_size is even for proper half2 pairing
    size_t aligned_u_size = (u_size + 1) & ~1;  // Round up to even number
    CUDA_CHECK(cudaMalloc(&d_u_fp16, aligned_u_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_m_fp32, m_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_fp32, src_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_coords_fp32, src_coords_size * sizeof(float)));
    STOP(cuda_malloc, timers)

    START(conversion_time)
    // Copy FP32 data directly
    CUDA_CHECK(cudaMemcpy(d_m_fp32, h_m, m_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_fp32, h_src, src_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_coords_fp32, h_src_coords, src_coords_size * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate temporary FP32 array on device for conversion
    float *d_u_fp32_temp;
    CUDA_CHECK(cudaMalloc(&d_u_fp32_temp, u_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_u_fp32_temp, h_u, u_size * sizeof(float), cudaMemcpyHostToDevice));

    // Convert wavefield from FP32 to FP16 on GPU for better performance
    int block_size = 256;
    int grid_size = (u_size + block_size - 1) / block_size;
    convert_fp32_to_fp16_kernel<<<grid_size, block_size>>>(d_u_fp32_temp, d_u_fp16, u_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary array
    CUDA_CHECK(cudaFree(d_u_fp32_temp));
    STOP(conversion_time, timers)

    // Constants computed inside kernel for better accuracy

    // Grid dimensions
    dim3 block_wave(8, 8, 8);
    int grid_x = max(1, (x_M - x_m + 1 + (int)block_wave.x - 1) / (int)block_wave.x);
    int grid_y = max(1, (y_M - y_m + 1 + (int)block_wave.y - 1) / (int)block_wave.y);
    int grid_z = max(1, (z_M - z_m + 1 + (int)block_wave.z - 1) / (int)block_wave.z);
    dim3 grid_wave(grid_x, grid_y, grid_z);

    // Grid dimensions for source injection
    dim3 block_src(32, 2, 2);
    int src_grid_x = max(1, (p_src_M - p_src_m + 1 + (int)block_src.x - 1) / (int)block_src.x);
    dim3 grid_src(src_grid_x, 1, 1);

    START(kernel_time)

    // Time stepping loop
    for (int time = time_m; time <= time_M; time++) {
        int t0 = time % 3;
        int t1 = (time + 2) % 3;
        int t2 = (time + 1) % 3;

        // Pointers to current time levels
        __half *u_t0 = d_u_fp16 + t0 * size_x * size_y * size_z;
        __half *u_t1 = d_u_fp16 + t1 * size_x * size_y * size_z;
        __half *u_t2 = d_u_fp16 + t2 * size_x * size_y * size_z;

        // Wave propagation kernel
        wave_kernel_mixed_precision<<<grid_wave, block_wave>>>(
            u_t2, u_t1, u_t0, d_m_fp32,
            x_M, x_m, y_M, y_m, z_M, z_m,
            size_x, size_y, size_z,
            dt, h_x, h_y, h_z
        );

        CUDA_CHECK(cudaGetLastError());

        // Source injection
        if (src_vec->size[0] * src_vec->size[1] > 0 && p_src_M - p_src_m + 1 > 0) {
            source_injection_kernel_mixed<<<grid_src, block_src>>>(
                u_t2, d_m_fp32, d_src_fp32, d_src_coords_fp32,
                time, p_src_M, p_src_m,
                x_M, x_m, y_M, y_m, z_M, z_m,
                size_x, size_y, size_z, src_vec->size[1],
                h_x, h_y, h_z, o_x, o_y, o_z
            );

            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    STOP(kernel_time, timers)

    // Convert result back to FP32 and copy to host
    START(cuda_memcpy)
    __half *h_u_half_result = new __half[u_size];
    CUDA_CHECK(cudaMemcpy(h_u_half_result, d_u_fp16, u_size * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < u_size; i++) {
        h_u[i] = __half2float(h_u_half_result[i]);
    }
    delete[] h_u_half_result;
    STOP(cuda_memcpy, timers)

    // Cleanup
    if (devicerm) {
        cudaFree(d_u_fp16);
        cudaFree(d_m_fp32);
        cudaFree(d_src_fp32);
        cudaFree(d_src_coords_fp32);
    }

    return 0;
}