# FDTD Unified Benchmark Makefile
# Builds single executable that can run any implementation

# Compiler paths (adjust for your system)
NVHPC_PATH = /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11/compilers/bin
NVC = $(NVHPC_PATH)/nvc++
NVCC = $(NVHPC_PATH)/nvcc

# Compiler flags
NVHPC_FLAGS = -acc -O3 -std=c++14 -I. -gpu=cc75
CUDA_FLAGS = -O3 --std=c++14 -arch=sm_75 -lineinfo -DTY=8 -DTZ=32 -DXCHUNK=64 -Xptxas=-O3,-dlcm=ca -use_fast_math

# Files
TARGET = fdtd_benchmark
MAIN = main.cpp
OPENACC_SRC = openacc.cpp
CUDA_SRC = cuda.cu
CUDA_OPT_SRC = cuda_optimized.cu

# Default target - OpenACC version (most portable)
all: openacc

# Build OpenACC version
openacc: $(MAIN) $(OPENACC_SRC)
	$(NVC) $(NVHPC_FLAGS) -o $(TARGET) $(MAIN) $(OPENACC_SRC) -lm
	@echo "Built OpenACC version. Usage: ./$(TARGET) [openacc]"

# Build CUDA version
cuda: $(MAIN) $(CUDA_SRC)
	$(NVCC) $(CUDA_FLAGS) -o $(TARGET) $(MAIN) $(CUDA_SRC)
	@echo "Built CUDA version. Usage: ./$(TARGET) [cuda]"

# Build CUDA optimized version
cuda-opt: $(MAIN) $(CUDA_OPT_SRC)
	$(NVCC) $(CUDA_FLAGS) -o $(TARGET) $(MAIN) $(CUDA_OPT_SRC)
	@echo "Built CUDA optimized version. Usage: ./$(TARGET) [cuda_optimized]"

# Run benchmarks
run: $(TARGET)
	./$(TARGET)

run-openacc: openacc
	./$(TARGET) openacc

run-cuda: cuda
	./$(TARGET) cuda

run-cuda-opt: cuda-opt
	./$(TARGET) cuda_optimized

# Show results
show-results:
	@echo "=== Benchmark Results ==="
	@if [ -f benchmark.csv ]; then \
		head -1 benchmark.csv; \
		tail -n +2 benchmark.csv | sort -t, -k1,1; \
	else \
		echo "No benchmark.csv found. Run 'make run' first."; \
	fi

# Clean up
clean:
	rm -f *.o $(TARGET) benchmark.csv

# Help
help:
	@echo "FDTD Unified Benchmark Build System"
	@echo "=================================="
	@echo ""
	@echo "Build targets:"
	@echo "  all (default)  - Build OpenACC version"
	@echo "  openacc        - Build OpenACC version"
	@echo "  cuda           - Build CUDA version"
	@echo "  cuda-opt       - Build CUDA optimized version"
	@echo ""
	@echo "Run targets:"
	@echo "  run            - Run current executable"
	@echo "  run-openacc    - Build and run OpenACC"
	@echo "  run-cuda       - Build and run CUDA"
	@echo "  run-cuda-opt   - Build and run CUDA optimized"
	@echo ""
	@echo "Utility:"
	@echo "  show-results   - Display benchmark CSV results"
	@echo "  clean          - Remove generated files"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Note: Only one implementation at a time. Rebuild to switch."

.PHONY: all openacc cuda cuda-opt run run-openacc run-cuda run-cuda-opt clean show-results help