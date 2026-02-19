#!/bin/bash
#SBATCH --job-name=fdtd_2080ti
#SBATCH --account=lr_amos
#SBATCH --partition=es0
#SBATCH --qos=es_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=slurm_fdtd_2080ti_%j.out
#SBATCH --chdir=/clusterfs/csd/amos/ycnliu/check_gpu/fdtd_bench

echo "=== FDTD Benchmark on RTX 2080 Ti (sm_75) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
date

nvidia-smi

# Build in arch-specific directory to avoid races with other GPU jobs
BUILDDIR="_build_sm75"
rm -rf "$BUILDDIR" && mkdir -p "$BUILDDIR"
cp *.cu *.cpp *.h Makefile "$BUILDDIR"/ 2>/dev/null
cd "$BUILDDIR"
GPU_ARCH=sm_75 make -j4
if [ $? -ne 0 ]; then
    echo "BUILD FAILED"
    exit 1
fi

echo ""
echo "=== Running Benchmark ==="
./fdtd_benchmark 2>&1

echo ""
echo "=== Done ==="
date
