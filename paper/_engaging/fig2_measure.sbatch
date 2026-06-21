#!/bin/bash
#SBATCH --job-name=fss_fig2
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=fig2_%x_%j.out
#SBATCH --error=fig2_%x_%j.out
# Measure fig2 throughput (per-rep mean +/- SD) on ONE GPU.  Submit once per GPU type:
#   sbatch --gres=gpu:l40s:1 --job-name=fss_fig2_l40s paper/_engaging/fig2_measure.sbatch
#   sbatch --gres=gpu:h100:1 --job-name=fss_fig2_h100 paper/_engaging/fig2_measure.sbatch
#   sbatch --gres=gpu:a100:1 --job-name=fss_fig2_a100 paper/_engaging/fig2_measure.sbatch
#   sbatch --gres=gpu:h200:1 --job-name=fss_fig2_h200 paper/_engaging/fig2_measure.sbatch
set -x
exec 2>&1
REPO="${SLURM_SUBMIT_DIR:-$HOME/fss_run}"
cd "$REPO" || exit 1
bash ~/fss_env.sh fss python -u paper/fig2_throughput_scaling/measure.py --reps 7
echo "===== DONE ($(date)) ====="
