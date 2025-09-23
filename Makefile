# Streamlined Makefile for 3D Acoustic FDTD CUDA Kernels
# Single validation executable for all implementations

# Compiler paths (adjust for your system)
CUDA_PATH = /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-10.5.0/cuda-11.8.0-ky3sqqqaat26kya2ceeszhk4pcyd7owp
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# Compiler flags
NVCC_FLAGS = -O3 --std=c++14 -lineinfo
CXX_FLAGS = -O3 -std=c++14

# Include and library paths
INCLUDES = -I. -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64
LIBS = -lcudart -lm

# CUDA source files
CUDA_SOURCES = fdtd_cuda.cu fdtd_mixed_precision_simple.cu fdtd_temporal_blocking.cu
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Main targets
.PHONY: all test clean help

all: quick_test

# Single validation executable
quick_test: quick_test.o $(CUDA_OBJECTS)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)

# Build quick_test object
quick_test.o: quick_test.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Build CUDA objects
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@


# Run validation test
test: quick_test
	@echo "=== Running CUDA Validation Test ==="
	@export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$$LD_LIBRARY_PATH && ./quick_test

# Quick test via script
test-script: quick_test
	@echo "=== Running Test Script ==="
	./run_test.sh

# Clean build artifacts
clean:
	rm -f *.o quick_test

# Show help
help:
	@echo "Available targets:"
	@echo "  all         - Build quick_test (default)"
	@echo "  quick_test  - Build single validation executable"
	@echo "  test        - Run validation tests"
	@echo "  test-script - Run via test script"
	@echo "  clean       - Remove build artifacts"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Quick start:"
	@echo "  make && make test"