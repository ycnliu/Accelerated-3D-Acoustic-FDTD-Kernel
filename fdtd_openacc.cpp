// fdtd_openacc.cpp — cleaned single-definition OpenACC kernel
#define _POSIX_C_SOURCE 200809L
#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include <openacc.h>

// ===== Timing helpers (accumulate into profiler fields) =====
#define START(S) struct timeval start_##S, end_##S; gettimeofday(&start_##S, nullptr)
#define STOP(S, T) do { \
  gettimeofday(&end_##S, nullptr); \
  (T)->S += (double)(end_##S.tv_sec - start_##S.tv_sec) + \
            (double)(end_##S.tv_usec - start_##S.tv_usec) / 1e6; \
} while(0)

// ===== Project structs (match CUDA side for ABI compatibility) =====
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
  double section0 = 0.0;        // main update
  double section1 = 0.0;        // source injection
  double cuda_malloc = 0.0;     // unused here (kept for API compatibility)
  double cuda_memcpy = 0.0;     // unused here
  double kernel_time = 0.0;     // whole time loop
  double conversion_time = 0.0; // unused here
};

// ===== Single exported entry point =====
extern "C" int Kernel_OpenACC(
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
  // Device selection
  acc_init(acc_device_nvidia);
  if (deviceid != -1) acc_set_device_num(deviceid, acc_device_nvidia);

  // Alias with shapes (halo already included in sizes)
  float (*__restrict m)[m_vec->size[1]][m_vec->size[2]] =
      (float (*)[m_vec->size[1]][m_vec->size[2]]) m_vec->data;
  float (*__restrict src)[src_vec->size[1]] =
      (float (*)[src_vec->size[1]]) src_vec->data;
  float (*__restrict src_coords)[src_coords_vec->size[1]] =
      (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*__restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] =
      (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  // Precompute constants
  const float r1  = 1.0f / (dt * dt);
  const float r2  = 1.0f / (h_x * h_x);
  const float r3  = 1.0f / (h_y * h_y);
  const float r4  = 1.0f / (h_z * h_z);
  const float dt2 = dt * dt;

  // Single data region (copyin all inputs, copyout u at the end)
  #pragma acc data copyin( \
      u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]], \
      m[0:m_vec->size[0]][0:m_vec->size[1]][0:m_vec->size[2]], \
      src[0:src_vec->size[0]][0:src_vec->size[1]], \
      src_coords[0:src_coords_vec->size[0]][0:src_coords_vec->size[1]] ) \
      copyout(u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])
  {
    START(kernel_time);

    for (int time = time_m; time <= time_M; ++time) {
      const int t0 = (time)     % 3;
      const int t1 = (time + 2) % 3;
      const int t2 = (time + 1) % 3;

      // ===== Main update =====
      START(section0);
      #pragma acc parallel loop collapse(3) present(m,u)
      for (int x = x_m; x <= x_M; ++x) {
        for (int y = y_m; y <= y_M; ++y) {
          for (int z = z_m; z <= z_M; ++z) {
            const int X = x + 4, Y = y + 4, Z = z + 4;

            float uc = u[t0][X][Y][Z];
            float r5 = -2.5f * uc;

            // 4th-order stencil (±2)
            float d2dx2 = r5
              + (-8.33333333e-2f) * (u[t0][X-2][Y][Z] + u[t0][X+2][Y][Z])
              + (1.333333330f)   * (u[t0][X-1][Y][Z] + u[t0][X+1][Y][Z]);

            float d2dy2 = r5
              + (-8.33333333e-2f) * (u[t0][X][Y-2][Z] + u[t0][X][Y+2][Z])
              + (1.333333330f)   * (u[t0][X][Y-1][Z] + u[t0][X][Y+1][Z]);

            float d2dz2 = r5
              + (-8.33333333e-2f) * (u[t0][X][Y][Z-2] + u[t0][X][Y][Z+2])
              + (1.333333330f)   * (u[t0][X][Y][Z-1] + u[t0][X][Y][Z+1]);

            float mval = m[X][Y][Z];
            float lap  = r2*d2dx2 + r3*d2dy2 + r4*d2dz2;
            float tterm= -2.0f * r1 * uc + r1 * u[t1][X][Y][Z];

            u[t2][X][Y][Z] = dt2 * (lap - tterm * mval) / mval;
          }
        }
      }
      STOP(section0, timers);

      // ===== Source injection =====
      START(section1);
      if (src_vec->size[0]*src_vec->size[1] > 0 && p_src_M - p_src_m + 1 > 0) {
        #pragma acc parallel loop collapse(4) present(m,src,src_coords,u)
        for (int p_src = p_src_m; p_src <= p_src_M; ++p_src) {
          for (int rsrcx = 0; rsrcx <= 1; ++rsrcx) {
            for (int rsrcy = 0; rsrcy <= 1; ++rsrcy) {
              for (int rsrcz = 0; rsrcz <= 1; ++rsrcz) {

                int posx = (int)floorf((-o_x + src_coords[p_src][0]) / h_x);
                int posy = (int)floorf((-o_y + src_coords[p_src][1]) / h_y);
                int posz = (int)floorf((-o_z + src_coords[p_src][2]) / h_z);

                float px = -floorf((-o_x + src_coords[p_src][0]) / h_x)
                           + ((-o_x + src_coords[p_src][0]) / h_x);
                float py = -floorf((-o_y + src_coords[p_src][1]) / h_y)
                           + ((-o_y + src_coords[p_src][1]) / h_y);
                float pz = -floorf((-o_z + src_coords[p_src][2]) / h_z)
                           + ((-o_z + src_coords[p_src][2]) / h_z);

                if (rsrcx + posx >= x_m - 1 && rsrcy + posy >= y_m - 1 && rsrcz + posz >= z_m - 1 &&
                    rsrcx + posx <= x_M + 1 && rsrcy + posy <= y_M + 1 && rsrcz + posz <= z_M + 1) {

                  float w = (rsrcx*px + (1 - rsrcx)*(1 - px)) *
                            (rsrcy*py + (1 - rsrcy)*(1 - py)) *
                            (rsrcz*pz + (1 - rsrcz)*(1 - pz));

                  float r0 = 1.0e-2f * w * src[time][p_src] / m[posx+4][posy+4][posz+4];

                  #pragma acc atomic update
                  u[t2][rsrcx+posx+4][rsrcy+posy+4][rsrcz+posz+4] += r0;
                }
              }
            }
          }
        }
      }
      STOP(section1, timers);
    } // time loop

    acc_wait_all();
    STOP(kernel_time, timers);
  } // acc data

  (void)devicerm; // nothing persistent allocated here
  return 0;
}
