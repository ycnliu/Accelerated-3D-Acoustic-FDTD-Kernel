# FDTD Unified Benchmark Makefile (Revised for NVHPC + 2080 Ti)

# ---------------- Toolchains ----------------
NVHPC_PATH = /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11/compilers/bin
NVC  = $(NVHPC_PATH)/nvc++
NVCC = $(NVHPC_PATH)/nvcc

# ---------------- Architectures ----------------
# H100 = Hopper (sm_90 / cc90) - default for ConvStencil optimization
# RTX 2080 Ti = Turing (sm_75 / cc75)
# Set GPU_ARCH=sm_75 for 2080 Ti, GPU_ARCH=sm_90 for H100
GPU_ARCH      ?= sm_75
CUDA_ARCH_SM   = $(GPU_ARCH)
NVHPC_CC       = $(subst sm_,cc,$(GPU_ARCH))

# ---------------- Flags ----------------
# NVHPC compile flags (host/OpenACC/CUDA-interoperable objs)
NVHPC_FLAGS      = -acc -O3 -std=c++17 -I. -gpu=$(NVHPC_CC)
# NVHPC link flags (pull CUDA libs here, not during .cu compilation)
NVHPC_LINK_FLAGS = -cuda -cudalib=cublas,curand

# NVCC compile flags for .cu -> .o
CUDA_FLAGS  = -O3 --std=c++17 -arch=$(CUDA_ARCH_SM) -lineinfo \
              -Xptxas=-O3,-dlcm=ca -use_fast_math

# ---------------- Files ----------------
TARGET        = fdtd_benchmark
TEST_TARGET   = test_correctness
MAIN_SRC      = main.cpp
TEST_SRC      = test_correctness.cpp
OPENACC_SRC   = openacc.cpp
CUDA_SRC      = cuda.cu
CUDA_OPT_SRC  = cuda_optimized.cu

# Object list (do not remove any objects)
OBJS := $(MAIN_SRC:.cpp=.o) $(OPENACC_SRC:.cpp=.o) $(CUDA_SRC:.cu=.o) $(CUDA_OPT_SRC:.cu=.o)
KERNEL_OBJS := $(OPENACC_SRC:.cpp=.o) $(CUDA_SRC:.cu=.o) $(CUDA_OPT_SRC:.cu=.o)

# ---------------- Default Target ----------------
all: $(TARGET)

# ---------------- Main Build Rule ----------------
$(TARGET): $(OBJS)
	@echo "Linking executable..."
	$(NVC) $(NVHPC_FLAGS) $(NVHPC_LINK_FLAGS) -o $(TARGET) $(OBJS) -lm
	@echo "Built unified benchmark. Run 'make run' to execute."

# ---------------- Pattern Rules for Compilation ----------------
# C++ -> .o with nvc++
%.o: %.cpp
	$(NVC) $(NVHPC_FLAGS) -c -o $@ $<

# CUDA -> .o with nvcc (do NOT link CUDA libs here)
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c -o $@ $<


# ---------------- Run helpers ----------------
run: $(TARGET)
	@echo "Running unified benchmark (correctness + performance)..."
	@echo ""
	./$(TARGET)

# ---------------- Results ----------------
show-results:
	@echo "=== Benchmark Results ==="
	@if [ -f benchmark.csv ]; then \
		head -1 benchmark.csv; \
		tail -n +2 benchmark.csv | sort -t, -k1,1; \
	else \
		echo "No benchmark.csv found. Run 'make run' first."; \
	fi

# ---------------- Clean ----------------
clean:
	rm -f *.o $(TARGET) $(TEST_TARGET) benchmark.csv

# ---------------- Help ----------------
help:
	@echo "FDTD Unified Benchmark Build System"
	@echo "===================================="
	@echo ""
	@echo "GPU Architecture (set GPU_ARCH for your GPU):"
	@echo "  GPU_ARCH=sm_90 (default)  - H100 (Hopper) with ConvStencil optimization"
	@echo "  GPU_ARCH=sm_75            - RTX 2080 Ti (Turing)"
	@echo "  GPU_ARCH=sm_80            - A100 (Ampere)"
	@echo ""
	@echo "Example: make clean && GPU_ARCH=sm_90 make -j && make run"
	@echo ""
	@echo "Build targets:"
	@echo "  all (default)    - Build unified benchmark executable"
	@echo ""
	@echo "Test targets:"
	@echo "  run              - Run unified benchmark (correctness + performance)"
	@echo "                     Tests all implementations and generates benchmark.csv"
	@echo ""
	@echo "Utility:"
	@echo "  show-results     - Display benchmark CSV results"
	@echo "  clean            - Remove generated files and benchmark.csv"

# ---------------- Phony Targets ----------------
.PHONY: all run clean show-results help
