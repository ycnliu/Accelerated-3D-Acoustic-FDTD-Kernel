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
    double kernel_time;
    double conversion_time;
};

// Optimized convert-once kernel with half2 vectorization
__global__ void wave_kernel_optimized(
    __half *__restrict__ u_t2,
    __half *__restrict__ u_t1,
    __half *__restrict__ u_t0,
    const float *__restrict__ m,  // Use read-only cache
    int x_M, int x_m, int y_M, int y_m, int z_M, int z_m,
    int size_x, int size_y, int size_z,
    float r1, float r2, float r3, float r4  // Precomputed constants
) {
    // Calculate thread position
    int x = blockIdx.x * blockDim.x + threadIdx.x + x_m;
    int y = blockIdx.y * blockDim.y + threadIdx.y + y_m;
    int z = blockIdx.z * blockDim.z + threadIdx.z + z_m;

    // Stencil computation - bounds safety ensured by halo regions
    int idx = (x + 4) * size_y * size_z + (y + 4) * size_z + (z + 4);

    // CONVERT ONCE: Load all FP16 values and convert to FP32 registers
    float u_center = __half2float(u_t0[idx]);
    float u_prev = __half2float(u_t1[idx]);

    // Load stencil neighbors and convert once
    float u_xm2 = __half2float(u_t0[idx - 2*size_y*size_z]);
    float u_xm1 = __half2float(u_t0[idx - size_y*size_z]);
    float u_xp1 = __half2float(u_t0[idx + size_y*size_z]);
    float u_xp2 = __half2float(u_t0[idx + 2*size_y*size_z]);

    float u_ym2 = __half2float(u_t0[idx - 2*size_z]);
    float u_ym1 = __half2float(u_t0[idx - size_z]);
    float u_yp1 = __half2float(u_t0[idx + size_z]);
    float u_yp2 = __half2float(u_t0[idx + 2*size_z]);

    float u_zm2 = __half2float(u_t0[idx - 2]);
    float u_zm1 = __half2float(u_t0[idx - 1]);
    float u_zp1 = __half2float(u_t0[idx + 1]);
    float u_zp2 = __half2float(u_t0[idx + 2]);

    // Load medium property via read-only cache
    float m_val = __ldg(&m[idx]);

    // Compute entire stencil in FP32 - no intermediate conversions
    float r5 = -2.5f * u_center;

    // 4th-order derivatives
    float d2u_dx2 = r5 + (-8.33333333e-2f) * (u_xm2 + u_xp2) + 1.333333330f * (u_xm1 + u_xp1);
    float d2u_dy2 = r5 + (-8.33333333e-2f) * (u_ym2 + u_yp2) + 1.333333330f * (u_ym1 + u_yp1);
    float d2u_dz2 = r5 + (-8.33333333e-2f) * (u_zm2 + u_zp2) + 1.333333330f * (u_zm1 + u_zp1);

    // Laplacian
    float laplacian = r2 * d2u_dx2 + r3 * d2u_dy2 + r4 * d2u_dz2;

    // Time update
    float time_term = -2.0f * r1 * u_center + r1 * u_prev;
    float u_new = (laplacian - time_term * m_val) / m_val;

    // CONVERT ONCE: Single conversion back to half and store
    u_t2[idx] = __float2half(u_new);
}

// Half2 vectorized kernel for better memory throughput
__global__ void wave_kernel_half2_optimized(
    __half *__restrict__ u_t2,
    __half *__restrict__ u_t1,
    __half *__restrict__ u_t0,
    const float *__restrict__ m,
    int x_M, int x_m, int y_M, int y_m, int z_M, int z_m,
    int size_x, int size_y, int size_z,
    float r1, float r2, float r3, float r4
) {
    // Process two adjacent points in Z direction - ensure even alignment
    int x = blockIdx.x * blockDim.x + threadIdx.x + x_m;
    int y = blockIdx.y * blockDim.y + threadIdx.y + y_m;
    int z = (blockIdx.z * blockDim.z + threadIdx.z) * 2 + z_m; // Ensure even Z

    // Ensure we can process two points - bounds safety ensured by halo regions
    if (z <= z_M - 1) { // Ensure we don't go beyond z_M when processing two points

        int idx = (x + 4) * size_y * size_z + (y + 4) * size_z + (z + 4);

        // Vectorized loads using half2 - load two adjacent Z points
        __half2 u_center_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx]);
        __half2 u_prev_h2 = *reinterpret_cast<const __half2*>(&u_t1[idx]);

        // Convert to float2 for computation
        float2 u_center_f2 = __half22float2(u_center_h2);
        float2 u_prev_f2 = __half22float2(u_prev_h2);

        // Load stencil neighbors in vectorized form
        __half2 u_xm2_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx - 2*size_y*size_z]);
        __half2 u_xm1_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx - size_y*size_z]);
        __half2 u_xp1_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx + size_y*size_z]);
        __half2 u_xp2_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx + 2*size_y*size_z]);

        __half2 u_ym2_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx - 2*size_z]);
        __half2 u_ym1_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx - size_z]);
        __half2 u_yp1_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx + size_z]);
        __half2 u_yp2_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx + 2*size_z]);

        __half2 u_zm2_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx - 2]);
        __half2 u_zm1_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx - 1]);
        __half2 u_zp1_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx + 1]);
        __half2 u_zp2_h2 = *reinterpret_cast<const __half2*>(&u_t0[idx + 2]);

        // Convert all to float2
        float2 u_xm2_f2 = __half22float2(u_xm2_h2);
        float2 u_xm1_f2 = __half22float2(u_xm1_h2);
        float2 u_xp1_f2 = __half22float2(u_xp1_h2);
        float2 u_xp2_f2 = __half22float2(u_xp2_h2);

        float2 u_ym2_f2 = __half22float2(u_ym2_h2);
        float2 u_ym1_f2 = __half22float2(u_ym1_h2);
        float2 u_yp1_f2 = __half22float2(u_yp1_h2);
        float2 u_yp2_f2 = __half22float2(u_yp2_h2);

        float2 u_zm2_f2 = __half22float2(u_zm2_h2);
        float2 u_zm1_f2 = __half22float2(u_zm1_h2);
        float2 u_zp1_f2 = __half22float2(u_zp1_h2);
        float2 u_zp2_f2 = __half22float2(u_zp2_h2);

        // Load medium properties
        float m_val1 = __ldg(&m[idx]);
        float m_val2 = __ldg(&m[idx + 1]);

        // Compute stencil for both points simultaneously
        float2 r5_f2 = make_float2(-2.5f * u_center_f2.x, -2.5f * u_center_f2.y);

        // X derivatives
        float2 d2u_dx2 = make_float2(
            r5_f2.x + (-8.33333333e-2f) * (u_xm2_f2.x + u_xp2_f2.x) + 1.333333330f * (u_xm1_f2.x + u_xp1_f2.x),
            r5_f2.y + (-8.33333333e-2f) * (u_xm2_f2.y + u_xp2_f2.y) + 1.333333330f * (u_xm1_f2.y + u_xp1_f2.y)
        );

        // Y derivatives
        float2 d2u_dy2 = make_float2(
            r5_f2.x + (-8.33333333e-2f) * (u_ym2_f2.x + u_yp2_f2.x) + 1.333333330f * (u_ym1_f2.x + u_yp1_f2.x),
            r5_f2.y + (-8.33333333e-2f) * (u_ym2_f2.y + u_yp2_f2.y) + 1.333333330f * (u_ym1_f2.y + u_yp1_f2.y)
        );

        // Z derivatives
        float2 d2u_dz2 = make_float2(
            r5_f2.x + (-8.33333333e-2f) * (u_zm2_f2.x + u_zp2_f2.x) + 1.333333330f * (u_zm1_f2.x + u_zp1_f2.x),
            r5_f2.y + (-8.33333333e-2f) * (u_zm2_f2.y + u_zp2_f2.y) + 1.333333330f * (u_zm1_f2.y + u_zp1_f2.y)
        );

        // Laplacian
        float2 laplacian = make_float2(
            r2 * d2u_dx2.x + r3 * d2u_dy2.x + r4 * d2u_dz2.x,
            r2 * d2u_dx2.y + r3 * d2u_dy2.y + r4 * d2u_dz2.y
        );

        // Time update
        float2 time_term = make_float2(
            -2.0f * r1 * u_center_f2.x + r1 * u_prev_f2.x,
            -2.0f * r1 * u_center_f2.y + r1 * u_prev_f2.y
        );

        float2 u_new_f2 = make_float2(
            (laplacian.x - time_term.x * m_val1) / m_val1,
            (laplacian.y - time_term.y * m_val2) / m_val2
        );

        // Convert back to half2 and store
        __half2 u_new_h2 = __float22half2_rn(u_new_f2);
        *reinterpret_cast<__half2*>(&u_t2[idx]) = u_new_h2;
    }
}

// GPU-based FP32 to FP16 conversion kernel
__global__ void convert_fp32_to_fp16(const float *__restrict__ src, __half *__restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// GPU-based FP16 to FP32 conversion kernel
__global__ void convert_fp16_to_fp32(const __half *__restrict__ src, float *__restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

// Source injection kernel (optimized)
__global__ void source_injection_kernel_optimized(
    __half *__restrict__ u_t2,
    const float *__restrict__ m,
    const float *__restrict__ src,
    const float *__restrict__ src_coords,
    int time, int p_src_M, int p_src_m,
    int x_M, int x_m, int y_M, int y_m, int z_M, int z_m,
    int size_x, int size_y, int size_z, int src_size1,
    float h_x, float h_y, float h_z, float o_x, float o_y, float o_z
) {
    int p_src = blockIdx.x * blockDim.x + threadIdx.x + p_src_m;

    if (p_src <= p_src_M) {
        // Preload source data
        float src_val = __ldg(&src[time * src_size1 + p_src]);
        float3 src_pos = make_float3(
            __ldg(&src_coords[p_src * 3 + 0]),
            __ldg(&src_coords[p_src * 3 + 1]),
            __ldg(&src_coords[p_src * 3 + 2])
        );

        // Trilinear interpolation
        for (int rsrcx = 0; rsrcx <= 1; rsrcx++) {
            for (int rsrcy = 0; rsrcy <= 1; rsrcy++) {
                for (int rsrcz = 0; rsrcz <= 1; rsrcz++) {
                    int posx = (int)floorf((-o_x + src_pos.x) / h_x);
                    int posy = (int)floorf((-o_y + src_pos.y) / h_y);
                    int posz = (int)floorf((-o_z + src_pos.z) / h_z);

                    float px = -floorf((-o_x + src_pos.x) / h_x) + (-o_x + src_pos.x) / h_x;
                    float py = -floorf((-o_y + src_pos.y) / h_y) + (-o_y + src_pos.y) / h_y;
                    float pz = -floorf((-o_z + src_pos.z) / h_z) + (-o_z + src_pos.z) / h_z;

                    if (rsrcx + posx >= x_m - 1 && rsrcy + posy >= y_m - 1 && rsrcz + posz >= z_m - 1 &&
                        rsrcx + posx <= x_M + 1 && rsrcy + posy <= y_M + 1 && rsrcz + posz <= z_M + 1) {

                        float weight = (rsrcx * px + (1 - rsrcx) * (1 - px)) *
                                      (rsrcy * py + (1 - rsrcy) * (1 - py)) *
                                      (rsrcz * pz + (1 - rsrcz) * (1 - pz));

                        int m_idx = (posx + 4) * size_y * size_z + (posy + 4) * size_z + (posz + 4);
                        int u_idx = (rsrcx + posx + 4) * size_y * size_z + (rsrcy + posy + 4) * size_z + (rsrcz + posz + 4);

                        float r0 = 1.0e-2f * weight * src_val / __ldg(&m[m_idx]);

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
    }
}

extern "C" int Kernel_Optimized(
    struct dataobj *__restrict m_vec,
    struct dataobj *__restrict src_vec,
    struct dataobj *__restrict src_coords_vec,
    struct dataobj *__restrict u_vec,
    const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m,
    const float dt, const float h_x, const float h_y, const float h_z,
    const float o_x, const float o_y, const float o_z,
    const int p_src_M, const int p_src_m, const int time_M, const int time_m,
    const int deviceid, const int devicerm, struct profiler *timers, int use_half2 = 0
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
    float *d_u_temp; // For GPU conversion

    START(cuda_malloc)
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_u_fp16, u_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_u_temp, u_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_fp32, m_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_fp32, src_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_coords_fp32, src_coords_size * sizeof(float)));
    STOP(cuda_malloc, timers)

    START(conversion_time)
    // Copy FP32 data directly
    CUDA_CHECK(cudaMemcpy(d_m_fp32, h_m, m_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_fp32, h_src, src_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_coords_fp32, h_src_coords, src_coords_size * sizeof(float), cudaMemcpyHostToDevice));

    // Copy wavefield and convert on GPU
    CUDA_CHECK(cudaMemcpy(d_u_temp, h_u, u_size * sizeof(float), cudaMemcpyHostToDevice));

    // GPU conversion FP32 -> FP16
    int grid_convert = (u_size + 255) / 256;
    convert_fp32_to_fp16<<<grid_convert, 256>>>(d_u_temp, d_u_fp16, u_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    STOP(conversion_time, timers)

    // Precompute constants
    float r1 = 1.0f / (dt * dt);
    float r2 = 1.0f / (h_x * h_x);
    float r3 = 1.0f / (h_y * h_y);
    float r4 = 1.0f / (h_z * h_z);

    // Optimized grid dimensions - test different block shapes
    dim3 block_wave, grid_wave;
    if (use_half2) {
        // For half2 vectorized kernel
        block_wave = dim3(16, 4, 8);  // Better for memory access patterns
        int grid_x = max(1, (x_M - x_m + 1 + (int)block_wave.x - 1) / (int)block_wave.x);
        int grid_y = max(1, (y_M - y_m + 1 + (int)block_wave.y - 1) / (int)block_wave.y);
        int grid_z = max(1, ((z_M - z_m + 1) / 2 + (int)block_wave.z - 1) / (int)block_wave.z);
        grid_wave = dim3(grid_x, grid_y, grid_z);
    } else {
        // Standard kernel with optimized block shape
        block_wave = dim3(8, 8, 8);  // Good balance for occupancy
        int grid_x = max(1, (x_M - x_m + 1 + (int)block_wave.x - 1) / (int)block_wave.x);
        int grid_y = max(1, (y_M - y_m + 1 + (int)block_wave.y - 1) / (int)block_wave.y);
        int grid_z = max(1, (z_M - z_m + 1 + (int)block_wave.z - 1) / (int)block_wave.z);
        grid_wave = dim3(grid_x, grid_y, grid_z);
    }

    // Grid for source injection
    dim3 block_src(128, 1, 1);  // Higher occupancy for source injection
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

        // Choose optimized kernel
        if (use_half2) {
            wave_kernel_half2_optimized<<<grid_wave, block_wave>>>(
                u_t2, u_t1, u_t0, d_m_fp32,
                x_M, x_m, y_M, y_m, z_M, z_m,
                size_x, size_y, size_z, r1, r2, r3, r4
            );
        } else {
            wave_kernel_optimized<<<grid_wave, block_wave>>>(
                u_t2, u_t1, u_t0, d_m_fp32,
                x_M, x_m, y_M, y_m, z_M, z_m,
                size_x, size_y, size_z, r1, r2, r3, r4
            );
        }

        CUDA_CHECK(cudaGetLastError());

        // Source injection
        if (src_vec->size[0] * src_vec->size[1] > 0 && p_src_M - p_src_m + 1 > 0) {
            source_injection_kernel_optimized<<<grid_src, block_src>>>(
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

    // Convert result back on GPU and copy to host
    START(cuda_memcpy)
    convert_fp16_to_fp32<<<grid_convert, 256>>>(d_u_fp16, d_u_temp, u_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_u, d_u_temp, u_size * sizeof(float), cudaMemcpyDeviceToHost));
    STOP(cuda_memcpy, timers)

    // Cleanup
    if (devicerm) {
        cudaFree(d_u_fp16);
        cudaFree(d_u_temp);
        cudaFree(d_m_fp32);
        cudaFree(d_src_fp32);
        cudaFree(d_src_coords_fp32);
    }

    return 0;
}