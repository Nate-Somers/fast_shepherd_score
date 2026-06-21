# Running figs 5 & 6 on MIT Engaging (`pi_melkin`, node3615)

Unattended SLURM jobs so a flaky SSH link doesn't matter: submit once, the job
runs for hours on the GPU node and writes a small result JSON, retrieve it later.

## Files
- `_env.sh` — adaptive env setup (finds CUDA module + a conda env that imports
  torch+rdkit+shepherd_score; prints what it found). Override the env with
  `FSS_ENV=<name>`.
- `fig5_roshambo.sbatch` — builds ROSHAMBO (CUDA toolkit from the `cuda` module),
  runs the FSS-vs-ROSHAMBO2 head-to-head → `paper/fig5_roshambo_headtohead/results.json`.
- `fig6_enrichment.sbatch` — fetches a DUDE-Z target, runs the enrichment + ESP
  ablation → `paper/fig6_enrichment/enrichment.json`.
- `probe.sh` — one-shot cluster recon (run first to confirm module/env/partition names).

## Workflow (on Engaging, in any working session)
```bash
cd ~/Software/Github/fast_shepherd_score
git pull                                   # get the fig5/6 scripts (origin @ 94f3073)
bash paper/_engaging/probe.sh              # confirm cuda module, conda env, node3615
# adjust _env.sh FSS_ENV / module names if the probe shows different ones, then:
sbatch paper/_engaging/fig5_roshambo.sbatch
sbatch --export=ALL,TARGET=ADRB2 paper/_engaging/fig6_enrichment.sbatch
squeue --me                                # watch
```
Result JSONs come back to the figure folders; copy them to the laptop (or paste
the contents) and run the local `plot.py` to render the figures.

## Notes
- `--nodelist=node3615`: adjust if the node's SLURM name differs (probe shows it).
- Override the conda env: `sbatch --export=ALL,FSS_ENV=<env> ...`.
- fig6 default target ADRB2 (polar/charged GPCR pocket — a fair place to expect
  ESP to help); change with `TARGET=...`. Start with `--limit-decoys` (in the
  script) then remove for the full screen.
- If ROSHAMBO's build fails, the job still finishes; `run.py` reports the missing
  install and exits cleanly so the FSS-side timing still runs.
