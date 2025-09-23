#!/bin/bash

# CUDA Validation Test Runner
echo "=== CUDA FDTD Implementation Validator ==="
echo "Setting up environment..."

# Set CUDA library path
export LD_LIBRARY_PATH=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-10.5.0/cuda-11.8.0-ky3sqqqaat26kya2ceeszhk4pcyd7owp/lib64:$LD_LIBRARY_PATH

# Check if executable exists
if [ ! -f "./quick_test" ]; then
    echo "❌ quick_test executable not found!"
    echo "Please run: make quick_test"
    exit 1
fi

echo "✓ Environment ready"
echo "Running validation tests..."
echo ""

# Run the tests
./quick_test

exit_code=$?

echo ""
echo "=== Test Complete ==="
if [ $exit_code -eq 0 ]; then
    echo "✅ All tests passed successfully!"
else
    echo "❌ Some tests failed (exit code: $exit_code)"
fi

exit $exit_code