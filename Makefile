NVHPC_PATH = /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11
NVCC = $(NVHPC_PATH)/compilers/bin/nvcc
NVC = $(NVHPC_PATH)/compilers/bin/nvc++
CXX = g++

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_75 -std=c++11 -lineinfo
CXX_FLAGS = -O3 -std=c++11 -fopenmp

# Include paths
INCLUDES = -I$(NVHPC_PATH)/cuda/include

# Library paths and libraries
LDFLAGS = -L$(NVHPC_PATH)/cuda/lib64
LIBS = -lcuda -lcudart

# Source files
CUDA_SRC = fdtd_cuda.cu
CPP_SRC = benchmark.cpp
OPENACC_SRC = fdtd_openacc_dummy.cpp

# Object files
CUDA_OBJ = fdtd_cuda.o
CPP_OBJ = benchmark.o
OPENACC_OBJ = fdtd_openacc_dummy.o

# Targets
all: benchmark comprehensive_benchmark validation_suite

# Validation targets
validation_suite: simple_validation_test cuda_openacc_validator

# Standard CUDA implementation
benchmark: $(CUDA_OBJ) $(CPP_OBJ) $(OPENACC_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

# Compile CUDA source
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ benchmark
$(CPP_OBJ): $(CPP_SRC)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile OpenACC source (dummy implementation)
$(OPENACC_OBJ): $(OPENACC_SRC)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Comprehensive benchmark with mixed precision - use nvc++ for linking to get OpenACC libs
comprehensive_benchmark: fdtd_cuda.o fdtd_mixed_precision_simple.o fdtd_optimized.o fdtd_openacc.o fdtd_temporal_blocking.o comprehensive_benchmark.o
	$(NVC) -acc -gpu=cc75 $(LDFLAGS) -o $@ $^ $(LIBS)

fdtd_temporal_blocking.o: fdtd_temporal_blocking.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

fdtd_mixed_precision_simple.o: fdtd_mixed_precision_simple.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

fdtd_optimized.o: fdtd_optimized.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

fdtd_openacc.o: fdtd_openacc.cpp
	$(NVC) -acc -gpu=cc75 -O3 -c $< -o $@

comprehensive_benchmark.o: comprehensive_benchmark.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Legacy tensor version (if needed)
tensor_benchmark: fdtd_tensor.o benchmark_tensor.o
	$(NVCC) $(NVCC_FLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

fdtd_tensor.o: fdtd_tensor.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Simple validation test (no solver dependency)
simple_validation_test: simple_validation_test.o
	$(CXX) $(CXX_FLAGS) $(LDFLAGS) -o $@ $^

simple_validation_test.o: simple_validation_test.cpp solver_validation.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c simple_validation_test.cpp -o $@

# Validation driver executable (full integration)
validation_driver: validation_driver.o fdtd_cuda.o fdtd_openacc.o
	$(NVC) -acc -gpu=cc75 $(LDFLAGS) -o $@ $^ $(LIBS)

validation_driver.o: validation_driver.cpp solver_validation.cpp fdtd_common.h
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c validation_driver.cpp -o $@

# CUDA vs OpenACC validator
cuda_openacc_validator: cuda_openacc_validator.o
	$(CXX) $(CXX_FLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

cuda_openacc_validator.o: cuda_openacc_validator.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Run validation suite
validate: validation_suite
	@echo "Running validation suite..."
	./simple_validation_test
	./cuda_openacc_validator default 64 64 64 100
	@echo "Validation complete. Check validation reports for results."

# Quick validation (faster subset)
validate-quick: validation_suite
	@echo "Running quick validation tests..."
	./simple_validation_test
	@echo "Quick validation complete."

# Clean
clean:
	rm -f *.o benchmark tensor_benchmark validation_driver cuda_openacc_validator simple_validation_test *.csv *.png *.pdf *.md ncu_report.csv validation_report.md

# Install dependencies (if needed)
install-deps:
	# This would install CUDA, cuBLAS, etc. if not already available

.PHONY: all clean install-deps