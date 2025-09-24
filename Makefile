# OpenACC/CUDA FDTD Makefile
# Compiler paths
NVHPC_PATH = /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11/compilers/bin
NVC = $(NVHPC_PATH)/nvc++
NVCC = $(NVHPC_PATH)/nvcc

# Compiler flags
NVHPC_FLAGS = -acc -O3 -std=c++14 -I.
CUDA_FLAGS = -O3 --std=c++14 -arch=sm_75 -lineinfo -Xptxas=-O3 -DTY=8 -DTZ=32 -DXCHUNK=64

# Targets
TARGET_UNIFIED = fdtd_benchmark
TARGET_ACC = fdtd_openacc
TARGET_CUDA = fdtd_cuda
TARGET_CUDA_OPT = fdtd_cuda_optimized
OBJS_ACC = openacc.o
OBJS_CUDA = cuda.o
OBJS_CUDA_OPT = cuda_optimized.o
MAIN = main.cpp

# Default target (OpenACC version)
all: $(TARGET_UNIFIED)

# All implementations
all-impl: fdtd_openacc_exe fdtd_cuda_exe fdtd_cuda_opt_exe

# Unified benchmark executable (OpenACC only for now)
$(TARGET_UNIFIED): $(OBJS_ACC) $(MAIN)
	$(NVC) $(NVHPC_FLAGS) -o $@ $(MAIN) $(OBJS_ACC) -lm

# OpenACC executable
$(TARGET_ACC): $(OBJS_ACC) $(MAIN)
	$(NVC) $(NVHPC_FLAGS) -o $@ $(MAIN) $(OBJS_ACC) -lm

# CUDA executable
$(TARGET_CUDA): $(OBJS_CUDA) main_cuda.cpp
	$(NVCC) $(CUDA_FLAGS) -o $@ main_cuda.cpp $(OBJS_CUDA)

# OpenACC object file
openacc.o: openacc.cpp
	$(NVC) $(NVHPC_FLAGS) -c -o $@ $<

# CUDA object file
cuda.o: cuda.cu
	$(NVCC) $(CUDA_FLAGS) -c -o $@ $<

# CUDA optimized object file
cuda_optimized.o: cuda_optimized.cu
	$(NVCC) $(CUDA_FLAGS) -c -o $@ $<

# Individual executables for each implementation
fdtd_openacc_exe: $(OBJS_ACC) $(MAIN)
	$(NVC) $(NVHPC_FLAGS) -o $@ $(MAIN) $(OBJS_ACC) -lm

fdtd_cuda_exe: $(OBJS_CUDA) $(MAIN)
	$(NVCC) $(CUDA_FLAGS) -o $@ $(MAIN) $(OBJS_CUDA)

fdtd_cuda_opt_exe: $(OBJS_CUDA_OPT) $(MAIN)
	$(NVCC) $(CUDA_FLAGS) -o $@ $(MAIN) $(OBJS_CUDA_OPT)

# Run programs
run: $(TARGET_UNIFIED)
	./$(TARGET_UNIFIED)

run-acc: $(TARGET_UNIFIED)
	./$(TARGET_UNIFIED) openacc

run-cuda: $(TARGET_UNIFIED)
	./$(TARGET_UNIFIED) cuda

run-both: run

run-old-acc: $(TARGET_ACC)
	./$(TARGET_ACC)

run-old-cuda: $(TARGET_CUDA)
	./$(TARGET_CUDA)

# Run specific implementations
run-openacc-only: fdtd_openacc_exe
	./fdtd_openacc_exe openacc

run-cuda-only: fdtd_cuda_exe
	./fdtd_cuda_exe cuda

run-cuda-opt-only: fdtd_cuda_opt_exe
	./fdtd_cuda_opt_exe cuda_optimized

run-all-impl: fdtd_openacc_exe fdtd_cuda_exe fdtd_cuda_opt_exe
	@echo "=== Running All Implementations ==="
	./fdtd_openacc_exe openacc
	./fdtd_cuda_exe cuda
	./fdtd_cuda_opt_exe cuda_optimized

# Clean up
clean:
	rm -f *.o $(TARGET_UNIFIED) $(TARGET_ACC) $(TARGET_CUDA) $(TARGET_CUDA_OPT) benchmark.csv main_cuda.cpp fdtd_openacc_exe fdtd_cuda_exe fdtd_cuda_opt_exe

# Rebuild everything
rebuild: clean all

# Show benchmark results
show-results:
	@echo "=== Benchmark Results ==="
	@if [ -f benchmark.csv ]; then \
		echo "Method,Total_Time(s),Section0_Time(s),Section1_Time(s),GFLOPS,NX,NY,NZ,Timesteps"; \
		tail -n +2 benchmark.csv; \
	else \
		echo "No benchmark.csv found. Run 'make run' first."; \
	fi

# Help
help:
	@echo "Available targets:"
	@echo "  all               - Build unified benchmark executable (default - OpenACC)"
	@echo "  all-impl          - Build all individual implementation executables"
	@echo "  run               - Run unified benchmark (OpenACC by default)"
	@echo "  run-acc           - Run OpenACC benchmark via unified executable"
	@echo "  run-cuda          - Run CUDA benchmark via unified executable"
	@echo "  run-openacc-only  - Run OpenACC-only executable"
	@echo "  run-cuda-only     - Run CUDA-only executable"
	@echo "  run-cuda-opt-only - Run optimized CUDA-only executable"
	@echo "  run-all-impl      - Run all three implementations"
	@echo "  clean             - Remove all generated files"
	@echo "  rebuild           - Clean and build"
	@echo "  show-results      - Display benchmark results from CSV"
	@echo "  help              - Show this help message"

.PHONY: all run clean rebuild show-results help