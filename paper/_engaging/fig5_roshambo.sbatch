#!/bin/bash
#SBATCH --job-name=fss_fig5
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=fig5_%x_%j.out
#SBATCH --error=fig5_%x_%j.out
# Fig 5 — fast_shepherd_score vs ROSHAMBO2 head-to-head, unattended.
#   sbatch paper/_engaging/fig5_roshambo.sbatch
#   sbatch --gres=gpu:h100:1 --job-name=fss_fig5_h100 paper/_engaging/fig5_roshambo.sbatch
# Runs both tools on the SAME GPU over the SAME conformers; fss in env `fss`, ROSHAMBO2 in
# env `roshambo2` (built by build_roshambo2.sbatch).  N must saturate the GPU.
set -x
exec 2>&1
REPO="${SLURM_SUBMIT_DIR:-$HOME/fss_run}"
cd "$REPO" || exit 1
LIBTGT="${LIBTGT:-fa10}"     # DUD-E target whose decoys form the diverse library + actives the queries
N="${N:-4000}"
QUERIES="${QUERIES:-8}"
DATA="$REPO/paper/fig5_roshambo_headtohead/data"
mkdir -p "$DATA"

echo "===== FETCH DUD-E library ($LIBTGT) ====="
[ -s "$DATA/lib.smi" ] || curl -fsSL "https://dude.docking.org/targets/$LIBTGT/decoys_final.ism" -o "$DATA/lib.smi"
[ -s "$DATA/queries.smi" ] || curl -fsSL "https://dude.docking.org/targets/$LIBTGT/actives_final.ism" -o "$DATA/queries.smi"
wc -l "$DATA/lib.smi" "$DATA/queries.smi"

echo "===== PREPARE (fss env): embed conformers, write SDFs, build fss cache ====="
bash ~/fss_env.sh fss python -u paper/fig5_roshambo_headtohead/run.py --prepare \
  --lib "$DATA/lib.smi" --queries-smi "$DATA/queries.smi" --n "$N" --queries "$QUERIES"

echo "===== FSS SIDE (fss env) ====="
bash ~/fss_env.sh fss python -u paper/fig5_roshambo_headtohead/run.py --side fss --modes vol surf

echo "===== ROSHAMBO2 SIDE (roshambo2 env) ====="
bash ~/fss_env.sh roshambo2 python -u paper/fig5_roshambo_headtohead/run.py --side roshambo2 \
  || echo "WARN: roshambo2 side failed (see trace); combine will still emit the fss side"

echo "===== COMBINE (fss env) ====="
bash ~/fss_env.sh fss python -u paper/fig5_roshambo_headtohead/run.py --combine
echo "===== DONE ($(date)) ====="
echo "result: $REPO/paper/fig5_roshambo_headtohead/results.json"
