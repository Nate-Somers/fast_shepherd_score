#!/usr/bin/env bash
# Shared environment setup for the Engaging sbatch jobs. Sourced by the job
# scripts. Adaptive + loud: it detects the CUDA module and a conda env that can
# import torch+rdkit+shepherd_score, and prints what it found so a failed first
# run still tells us how to fix the next one.
echo "===== ENV SETUP ($(date)) on $(hostname) ====="

# Lmod / modules in a non-login shell
for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /etc/profile.d/z00_lmod.sh \
         /usr/share/lmod/lmod/init/bash; do [ -f "$f" ] && source "$f" 2>/dev/null; done

echo "--- CUDA module ---"
for m in cuda/12.4 cuda/12.4.0 cuda/12.2 cuda cuda/12.1; do
  module load "$m" 2>/dev/null && { echo "loaded module $m"; break; }
done
command -v nvcc && nvcc --version | tail -1 || echo "WARN: no nvcc after module load"
command -v gcc && gcc --version | head -1
command -v cmake || module load cmake 2>/dev/null || echo "WARN: no cmake"

echo "--- conda ---"
# Find a conda
for c in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniforge3/etc/profile.d/conda.sh"; do
  [ -f "$c" ] && { source "$c"; echo "sourced $c"; break; }
done
command -v conda >/dev/null || { module load miniforge 2>/dev/null || module load anaconda 2>/dev/null; }
command -v conda >/dev/null && conda env list

# Pick an env that imports the stack. Override by exporting FSS_ENV before sbatch.
pick_env() {
  for e in "$FSS_ENV" SimModelEnv fss shepherd shepherd_score base; do
    [ -z "$e" ] && continue
    conda activate "$e" 2>/dev/null || continue
    if python -c "import torch, rdkit, shepherd_score" 2>/dev/null; then
      echo "USING conda env: $e"; return 0
    fi
  done
  return 1
}
if ! pick_env; then
  echo "ERROR: no conda env imports torch+rdkit+shepherd_score."
  echo "Set FSS_ENV=<envname> (sbatch --export=ALL,FSS_ENV=...) or create one."
  conda env list
fi

echo "--- python / GPU ---"
which python; python -c "import sys; print('py', sys.version.split()[0])"
python - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e:
    print("torch import ERR", e)
PY
# xtb on PATH (for ESP charges in fig6)
export PATH="$(dirname "$(which python)"):$PATH"
command -v xtb >/dev/null && echo "xtb: $(command -v xtb)" || echo "WARN: xtb not found (fig6 ESP charges need it)"
echo "===== ENV SETUP DONE ====="
