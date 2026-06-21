#!/bin/bash
#SBATCH --job-name=fss_figs134
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=figs134_%j.out
#SBATCH --error=figs134_%j.out
# Refresh figs 1, 3, 4 on one datacenter GPU node (same machine = honest fig3; jax+triton
# on one card = fig1; xTB on this node = fig4).  sbatch paper/_engaging/figs134_refresh.sbatch
set -x
exec 2>&1
REPO="${SLURM_SUBMIT_DIR:-$HOME/fss_run}"
cd "$REPO" || exit 1

# fig1 needs jax (reference backend); fig3 needs espsim + a diverse library (DUD-E decoys)
bash ~/fss_env.sh fss pip install -q "jax[cpu]" espsim 2>&1 | tail -3
DATA="$REPO/paper/fig3_speed_vs_baselines/data"; mkdir -p "$DATA"
[ -s "$DATA/lib.smi" ] || curl -fsSL "https://dude.docking.org/targets/aces/decoys_final.ism" -o "$DATA/lib.smi"
wc -l "$DATA/lib.smi"

echo "===== FIG 1 — backend parity (aligned + fixed-pose) ====="
bash ~/fss_env.sh fss python -u paper/fig1_backend_parity/run.py

echo "===== FIG 3 — same-machine speed vs baselines ====="
bash ~/fss_env.sh fss python -u paper/fig3_speed_vs_baselines/run.py --lib "$DATA/lib.smi" --n-lib 300

echo "===== FIG 4 — ESP value (UQ + polarity) ====="
bash ~/fss_env.sh fss python -u paper/fig4_esp_value/run.py

echo "===== DONE ($(date)) ====="
