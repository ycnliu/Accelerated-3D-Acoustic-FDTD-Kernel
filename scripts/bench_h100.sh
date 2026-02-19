#!/bin/bash
#SBATCH --job-name=fdtd_h100
#SBATCH --account=lr_amos
#SBATCH --partition=es2
#SBATCH --qos=es_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=1:00:00
#SBATCH --output=slurm_fdtd_h100_%j.out
#SBATCH --chdir=/clusterfs/csd/amos/ycnliu/check_gpu/fdtd_bench

echo "=== FDTD Benchmark on H100 (sm_90) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
date

nvidia-smi

BUILDDIR="_build_sm90"
rm -rf "$BUILDDIR" && mkdir -p "$BUILDDIR"
cp *.cu *.cpp *.h Makefile "$BUILDDIR"/ 2>/dev/null
cd "$BUILDDIR"
GPU_ARCH=sm_90 make -j4
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
