#!/bin/bash
#SBATCH --job-name=build_roshambo2
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# Build ROSHAMBO2 (molecularinformatics/roshambo2) into its OWN conda env on Engaging.
# ROSHAMBO2 deps are light (rdkit/numpy/scipy/pybind11/cmake, NO torch/cupy); the CUDA
# kernels are built with nvcc from the cuda module. Built for sm_89 (L40S) AND sm_90 (H100)
# so the .so runs on both, letting fig5 run on either card.
#   sbatch paper/_engaging/build_roshambo2.sbatch
set -x
exec 2>&1
SRC="$HOME/roshambo2_src"
ENVNAME="roshambo2"

for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do [ -f "$f" ] && source "$f" 2>/dev/null; done
module load cuda/12.9.1 2>/dev/null || module load cuda 2>/dev/null
module load gcc/12.2.0 2>/dev/null
module load cmake 2>/dev/null
module load miniforge/25.11.0-0 2>/dev/null || module load miniforge 2>/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "===== TOOLCHAIN ====="
which nvcc && nvcc --version | tail -2
which gcc && gcc --version | head -1
which cmake && cmake --version | head -1
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export CMAKE_CUDA_ARCHITECTURES="89;90"
export TORCH_CUDA_ARCH_LIST="8.9;9.0"
echo "CUDA_HOME=$CUDA_HOME  ARCH=$CMAKE_CUDA_ARCHITECTURES"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "===== CLONE ====="
if [ ! -d "$SRC" ]; then
  git clone https://github.com/molecularinformatics/roshambo2.git "$SRC" || { echo "CLONE FAILED"; exit 1; }
fi
cd "$SRC"
git log --oneline -1
echo "--- environment.yaml ---"; cat environment.yaml 2>/dev/null | head -40

echo "===== CONDA ENV ====="
if ! conda env list | grep -q "envs/$ENVNAME"; then
  # force the env name regardless of the name inside environment.yaml
  conda env create -n "$ENVNAME" -f environment.yaml || { echo "ENV CREATE FAILED"; exit 1; }
fi
conda activate "$ENVNAME"
echo "env python: $(which python)"; python --version

echo "===== PIP INSTALL (build CUDA extension) ====="
pip install --no-build-isolation -v . 2>&1 | tail -60 || pip install -v . 2>&1 | tail -60

echo "===== SANITY ====="
python - <<'PY'
import inspect
try:
    from roshambo2 import Roshambo2
    print("ROSHAMBO2 import OK")
    print("__init__:", inspect.signature(Roshambo2.__init__))
    try:
        print("compute :", inspect.signature(Roshambo2.compute))
    except Exception as e:
        print("compute sig err:", e)
    print("methods:", [m for m in dir(Roshambo2) if not m.startswith('_')])
except Exception as e:
    import traceback; traceback.print_exc()
    print("ROSHAMBO2 IMPORT FAILED:", e)
PY
echo "===== DONE ($(date)) ====="
