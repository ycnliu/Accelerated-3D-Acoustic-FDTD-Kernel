#ifndef FDTD_COMMON_H
#define FDTD_COMMON_H

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

#endif // FDTD_COMMON_H