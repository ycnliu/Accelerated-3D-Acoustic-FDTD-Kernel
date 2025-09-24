# ===============================
# Makefile: FDTD Benchmark Suite
# ===============================

# --- Paths ---
CUDA_PATH ?= /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-10.5.0/cuda-11.8.0-ky3sqqqaat26kya2ceeszhk4pcyd7owp
NVCC      ?= $(CUDA_PATH)/bin/nvcc
NSYS      ?= $(CUDA_PATH)/bin/nsys
CXX       ?= g++
NVCXX     ?= nvc++

# --- Flags ---
NVCC_FLAGS ?= -O3 --std=c++14 -lineinfo
CXX_FLAGS  ?= -O3 -std=c++14 -fopenmp
ACC_FLAGS  ?= -acc -O3 -std=c++14

INCLUDES   ?= -I. -I$(CUDA_PATH)/include
LDFLAGS    ?= -L$(CUDA_PATH)/lib64
LIBS       ?= -lcudart -lm -lgomp

# --- Sources / Objects ---
CUDA_SOURCES := fdtd_cuda.cu fdtd_mixed_precision_simple.cu fdtd_temporal_blocking.cu fdtd_optimized_advanced.cu
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)

OPENACC_SRC  := fdtd_openacc.cpp
OPENACC_OBJ  := $(OPENACC_SRC:.cpp=.o)

BENCHMARK_MAIN := fdtd_benchmark.cpp
BENCHMARK_OBJ  := fdtd_benchmark.o

# --- Phony targets ---
.PHONY: all clean fdtd_benchmark help

# Default build
all: fdtd_benchmark

# Complete benchmark (CUDA + OpenACC)
fdtd_benchmark: $(BENCHMARK_OBJ) $(CUDA_OBJECTS) $(OPENACC_OBJ)
	$(NVCXX) $(ACC_FLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)

# --- Object rules ---
# Benchmark compiled with OpenACC support
$(BENCHMARK_OBJ): $(BENCHMARK_MAIN)
	$(NVCXX) $(ACC_FLAGS) $(INCLUDES) -DUSE_OPENACC -c $< -o $@

# CUDA objects
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# OpenACC object
$(OPENACC_OBJ): $(OPENACC_SRC)
	$(NVCXX) $(ACC_FLAGS) $(INCLUDES) -c $< -o $@

# --- Cleaning ---
clean:
	rm -f *.o fdtd_benchmark

clean-all: clean
	rm -f *.qdrep *.nsys-rep *.sqlite *.txt *.csv

# --- Help ---
help:
	@echo "FDTD Benchmark Suite"
	@echo "Targets:"
	@echo "  all            - Build complete benchmark (CUDA + OpenACC)"
	@echo "  fdtd_benchmark - Build complete benchmark"
	@echo "  clean          - Clean build artifacts"
	@echo "  clean-all      - Clean all artifacts including profiles/results"
	@echo ""
	@echo "Usage:"
	@echo "  make           - Build benchmark"
	@echo "  ./fdtd_benchmark - Run comprehensive 4-GPU benchmark"
	@echo ""
	@echo "Requirements:"
	@echo "  module load nvhpc  (for OpenACC support)"
