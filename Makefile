# FDTD Unified Benchmark Makefile (Revised for Efficiency)

# ---------------- Toolchains ----------------
NVHPC_PATH = /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11/compilers/bin
NVC  = $(NVHPC_PATH)/nvc++
NVCC = $(NVHPC_PATH)/nvcc

# ---------------- Flags ----------------
NVHPC_FLAGS = -acc -O3 -std=c++17 -I. -gpu=cc90
CUDA_FLAGS  = -O3 --std=c++17 -arch=sm_90 -lineinfo -DTY=16 -DTZ=64 -DXCHUNK=32 -DROUNDUP_X=64 -Xptxas=-O3,-dlcm=ca -use_fast_math -DUSE_PIPELINE -DUSE_TENSORCORES=1 -DUSE_FP8_TC --extended-lambda

# ---------------- Files ----------------
TARGET        = fdtd_benchmark
MAIN_SRC      = main.cpp
OPENACC_SRC   = openacc.cpp
CUDA_SRC      = cuda.cu
CUDA_OPT_SRC  = cuda_optimized.cu

# Define object files for clarity and reuse
# The := operator is a "simply expanded" variable, good practice here.
OBJS := $(MAIN_SRC:.cpp=.o) $(OPENACC_SRC:.cpp=.o) $(CUDA_SRC:.cu=.o) $(CUDA_OPT_SRC:.cu=.o)

# ---------------- Default Target ----------------
all: $(TARGET)

# ---------------- Main Build Rule ----------------
# The final executable now depends on the object files.
# This link command will only run if one of the object files has been updated.
$(TARGET): $(OBJS)
	@echo "Linking executable..."
	$(NVC) $(NVHPC_FLAGS) -cuda -o $(TARGET) $(OBJS) -lm
	@echo "Built unified version with all implementations. Usage: ./$(TARGET) [all|openacc|cuda|cuda_optimized]"

# ---------------- Pattern Rules for Compilation ----------------
# These rules teach 'make' how to build object files efficiently.
# They will only run if the source file is newer than the object file.
# $< is an automatic variable for the first prerequisite (the source file).
# $@ is an automatic variable for the target (the object file).

# Rule to build .o from a .cpp file using nvc++
%.o: %.cpp
	$(NVC) $(NVHPC_FLAGS) -c -o $@ $<

# Rule to build .o from a .cu file using nvcc
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c -o $@ $<

# The 'unified' target is now just an alias for the main target
unified: $(TARGET)

# ---------------- Simplified Standalone Builds ----------------
# These are kept for convenience but are less efficient than the main build.
openacc:
	$(NVC) $(NVHPC_FLAGS) -o $(TARGET) $(MAIN_SRC) $(OPENACC_SRC) -lm
	@echo "Built OpenACC version. Usage: ./$(TARGET) openacc"

cuda:
	$(NVCC) $(CUDA_FLAGS) -o $(TARGET) $(MAIN_SRC) $(CUDA_SRC)
	@echo "Built CUDA version. Usage: ./$(TARGET) cuda"

cuda-opt:
	$(NVCC) $(CUDA_FLAGS) -o $(TARGET) $(MAIN_SRC) $(CUDA_OPT_SRC)
	@echo "Built CUDA optimized version. Usage: ./$(TARGET) cuda_optimized"

# ---------------- Run helpers (dangling target removed) ----------------
run: $(TARGET)
	./$(TARGET)

run-unified: $(TARGET)
	./$(TARGET) all

run-openacc: openacc
	./$(TARGET) openacc

run-cuda: cuda
	./$(TARGET) cuda

run-cuda-opt: cuda-opt
	./$(TARGET) cuda_optimized

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
	rm -f *.o $(TARGET) benchmark.csv

# ---------------- Help ----------------
help:
	@echo "FDTD Unified Benchmark Build System"
	@echo "=================================="
	@echo ""
	@echo "Build targets:"
	@echo "  all (default)    - Build unified version with all implementations"
	@echo "  unified          - An alias for 'all'"
	@echo "  openacc          - Build OpenACC only (less efficient, single command)"
	@echo "  cuda             - Build CUDA only (less efficient, single command)"
	@echo "  cuda-opt         - Build CUDA optimized only (less efficient, single command)"
	@echo ""
	@echo "Run targets:"
	@echo "  run              - Run current executable"
	@echo "  run-unified      - Build & run unified version (all implementations)"
	@echo "  run-openacc      - Build & run OpenACC only"
	@echo "  run-cuda         - Build & run CUDA only"
	@echo "  run-cuda-opt     - Build & run CUDA optimized only"
	@echo ""
	@echo "Unified executable options:"
	@echo "  ./fdtd_benchmark all             - Run all implementations"
	@echo "  ./fdtd_benchmark openacc         - Run OpenACC only"
	@echo "  ./fdtd_benchmark cuda            - Run CUDA only"
	@echo "  ./fdtd_benchmark cuda_optimized  - Run CUDA optimized only"
	@echo ""
	@echo "Utility:"
	@echo "  show-results     - Display benchmark CSV results"
	@echo "  clean            - Remove generated files"

# ---------------- Phony Targets ----------------
.PHONY: all unified openacc cuda cuda-opt \
        run run-unified run-openacc run-cuda run-cuda-opt \
        clean show-results help