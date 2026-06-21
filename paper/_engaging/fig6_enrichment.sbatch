#!/bin/bash
#SBATCH --job-name=fss_fig6
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=72G
#SBATCH --time=08:00:00
#SBATCH --output=fig6_%x_%j.out
#SBATCH --error=fig6_%x_%j.out
# Fig 6 — retrospective enrichment + ESP ablation on ONE DUD-E target, unattended.
#   sbatch --job-name=fss_fig6_ACES --export=ALL,TARGET=aces paper/_engaging/fig6_enrichment.sbatch
# DUD-E gives clean per-target SMILES (actives_final.ism / decoys_final.ism). We keep ALL
# actives and subsample decoys to N_DECOYS, average enrichment over QUERIES query actives,
# and sweep ESP lam.  The molecule cache makes precompute resumable.
set -x
exec 2>&1
REPO="${SLURM_SUBMIT_DIR:-$HOME/fss_run}"
cd "$REPO" || exit 1
TARGET="${TARGET:-aces}"
N_DECOYS="${N_DECOYS:-3000}"
QUERIES="${QUERIES:-8}"
DATA="$REPO/paper/fig6_enrichment/data/$TARGET"
mkdir -p "$DATA"

echo "===== FETCH DUD-E target: $TARGET ====="
for f in actives decoys; do
  out="$DATA/${f}.smi"
  [ -s "$out" ] && { echo "$out present ($(wc -l < "$out") lines)"; continue; }
  url="https://dude.docking.org/targets/$TARGET/${f}_final.ism"
  echo "GET $url"
  curl -fsSL "$url" -o "$out" && echo "  -> $(wc -l < "$out") lines" || echo "  WARN: fetch failed for $out"
done

echo "===== ENRICHMENT (surf vs esp ablation, lam sweep) ====="
bash ~/fss_env.sh fss python -u paper/fig6_enrichment/run.py \
  --target "$TARGET" \
  --actives "$DATA/actives.smi" --decoys "$DATA/decoys.smi" \
  --n-decoys "$N_DECOYS" --queries "$QUERIES" \
  --modes surf esp --lams 0.3 0.1 0.03 0.01 0.003 \
  --workers "${SLURM_CPUS_PER_TASK:-16}"
echo "===== DONE ($(date)) ====="
echo "result: $REPO/paper/fig6_enrichment/enrichment_${TARGET}.json"
