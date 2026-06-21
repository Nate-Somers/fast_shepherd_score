#!/usr/bin/env bash
# Build ROSHAMBO (open-source GPU shape+color, GPL-3.0) for the fig5 head-to-head.
# ROSHAMBO wraps PAPER's CUDA kernels, so it needs the CUDA *toolkit* (nvcc), not
# just the runtime. The fast_shepherd_score env (SimModelEnv) has the CUDA runtime
# but NOT nvcc/cmake, so install those first.
#
# Run this on a machine with an NVIDIA GPU. A datacenter GPU (L40S/H100) is
# preferable to a 6 GB laptop for a fair throughput comparison.
set -euo pipefail
ENV="${1:-SimModelEnv}"

echo ">> installing build tools (nvcc matching CUDA 12.4, cmake) into env: $ENV"
conda install -n "$ENV" -y -c nvidia cuda-nvcc=12.4 cuda-toolkit=12.4
conda install -n "$ENV" -y -c conda-forge cmake eigen

echo ">> cloning + building ROSHAMBO (+ its PAPER submodule)"
git clone --recursive https://github.com/molecularinformatics/roshambo.git
cd roshambo
# Point the build at the GPU's compute capability, e.g. 8.9 (Ada/RTX 4050/L40S),
# 9.0 (H100/H200). Set CUDA_HOME to the env's toolkit.
export CUDA_HOME="$(conda info --base)/envs/$ENV"
export PATH="$CUDA_HOME/bin:$PATH"
# follow the repo README: set the right -arch=sm_XX for your GPU, then:
conda run -n "$ENV" pip install -e .

echo ">> sanity check"
conda run -n "$ENV" python -c "import roshambo; print('roshambo OK')"
echo ">> done. Now: PYTHONPATH=. python paper/fig5_roshambo_headtohead/run.py"
