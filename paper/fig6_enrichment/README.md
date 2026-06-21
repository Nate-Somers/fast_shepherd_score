# Figure 6 — virtual-screening enrichment + the ESP ablation

**Claim defended (the decisive one):** the package retrieves actives in a
retrospective virtual screen, and **adding ESP to shape improves retrieval** — the
real utility argument for the ESP differentiator (Fig 4 shows ESP carries
orthogonal information; this shows it *helps*).

## Status: harness complete + verified, awaiting a dataset run
The pipeline (conformers → xTB charges → GPU align/score → ROC-AUC / EF / BEDROC)
is implemented and **smoke-tested end-to-end** (`--smoke`, runs on the curated
drugs — not a benchmark). It needs a real actives/decoys set to produce numbers.
That run is a **large compute job** (thousands of molecules × conformer gen × xTB
single points), best done on a workstation/cluster, not in this session.

## Design
- **Benchmark:** use a **de-biased** set — **DUDE-Z** (matched-property decoys; the
  set ROSHAMBO used, so results are comparable) and/or **LIT-PCBA** (experimental
  actives/inactives, the current low-bias gold standard). Avoid vanilla DUD-E alone
  (analog bias).
- **Per target:** pick a query active; align+score every library molecule by mode;
  rank; label actives=1 / decoys=0; compute **ROC-AUC, EF1%, EF5%, BEDROC(α=20)**.
- **ESP ablation:** run `surf` (shape only) and `esp` (shape+ESP) on the *same*
  library and compare enrichment. `pharm` optional. `lam=0.01` is used for esp
  (Fig 4: the default 0.3 makes ESP nearly inert; justify the chosen `lam` in the
  paper, or sweep it).
- Aggregate EF/BEDROC across targets (mean ± spread); compare to ROSHAMBO / a ROCS
  reference column where available.

## Get the data
DUDE-Z: https://dudez.docking.org/  (per-target `actives.smi` + `decoys.smi`).
LIT-PCBA: https://drugdesign.unistra.fr/LIT-PCBA/ .
Place SMILES files locally, then:
```bash
# first, tractable run: subsample decoys
PYTHONPATH=. python paper/fig6_enrichment/run.py \
    --actives <target>/actives.smi --decoys <target>/decoys.smi --limit-decoys 1000
# full run: drop --limit-decoys (large; budget hours of conformer+xTB time)
```
Outputs `enrichment.json` (per-mode AUC/EF/BEDROC + scores + labels) for `plot.py`.

## Metrics (implemented, standard)
- ROC-AUC (rank-based), EF@1%/5% (enrichment factor), BEDROC α=20 (Truchon & Bayly
  2007). Verified on the smoke run.

## What a good result looks like
EF1% and BEDROC for `esp` ≥ `surf` on electrostatically-driven targets, with the
gap largest where electrostatics discriminate (charged/polar binding sites). Report
honestly per-target — ESP will not help everywhere, and saying where it does is a
stronger result than a single aggregate.
