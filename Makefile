# Streamlined Makefile for 3D Acoustic FDTD CUDA Kernels
# Single validation executable for all implementations

# Compiler paths (adjust for your system)
CUDA_PATH = /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-10.5.0/cuda-11.8.0-ky3sqqqaat26kya2ceeszhk4pcyd7owp
NVCC = $(CUDA_PATH)/bin/nvcc
NSYS = $(CUDA_PATH)/bin/nsys
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
.PHONY: all test clean help profile profile-kernels profile-report

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

# Profile with nsys (full timeline)
profile: quick_test
	@echo "=== Running NSYS Profiling ==="
	@export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$$LD_LIBRARY_PATH && \
	$(NSYS) profile --trace=cuda,nvtx --output=fdtd_profile --force-overwrite=true ./quick_test
	@echo "✓ Profile saved as fdtd_profile.qdrep"

# Profile kernels only (lightweight)
profile-kernels: quick_test
	@echo "=== Running NSYS Kernel Profiling ==="
	@export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$$LD_LIBRARY_PATH && \
	$(NSYS) profile --trace=cuda --stats=true --output=fdtd_kernels --force-overwrite=true ./quick_test
	@echo "✓ Kernel profile saved as fdtd_kernels.qdrep"

# Generate text report from profile
profile-report: fdtd_kernels.nsys-rep
	@echo "=== Generating Profile Report ==="
	$(NSYS) stats --report gputrace,gpukernsum,gpumemtimesum,gpumemsum fdtd_kernels.nsys-rep > fdtd_profile_report.txt
	@echo "✓ Text report saved as fdtd_profile_report.txt"

# Clean build artifacts
clean:
	rm -f *.o quick_test

# Clean all artifacts including profiles
clean-all: clean
	rm -f *.qdrep *.sqlite *.txt

# Show help
help:
	@echo "Available targets:"
	@echo "  all            - Build quick_test (default)"
	@echo "  quick_test     - Build single validation executable"
	@echo "  test           - Run validation tests"
	@echo "  test-script    - Run via test script"
	@echo "  profile        - Run nsys profiling (full timeline)"
	@echo "  profile-kernels- Run nsys profiling (kernels only)"
	@echo "  profile-report - Generate text report from profile"
	@echo "  clean          - Remove build artifacts"
	@echo "  clean-all      - Remove all artifacts including profiles"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Quick start:"
	@echo "  make && make test"
	@echo "  make profile        # Profile all implementations"
	@echo "  make profile-report # Generate readable report"