# Reference implementation map

Paths below are under `shepherd_score/` except tests.

| Concern | Implementation |
|---|---|
| Molecular data and public single-pair methods | `container/_core.py` |
| Result properties | `_ALIGN_KEYS`, `_alignment_property`, `AlignmentResult` in `container/_core.py` |
| Differentiable objective and optimizer | `alignment/_torch.py` |
| Public optimizer exports | `alignment/__init__.py` |
| Gaussian raw overlap and Tanimoto | `score/gaussian_overlap.py` |
| ESP overlap and surface-potential agreement | `score/electrostatic_scoring.py` |
| Typed directional overlap | `score/pharmacophore_scoring.py` |
| Canonical accelerated mode definitions | `accel/_modes.py` |

`score/__init__.py` is empty; existing callers import the score modules
directly. Most optimizers return aligned positions, transform, and score;
pharmacophore optimizers additionally return aligned vectors. Read the called
function before unpacking it.

## Reuse examples

| Mode | Reference optimizer / option |
|---|---|
| `vol` | `optimize_ROCS_overlay` |
| `vol_esp`, `surf_esp` | `optimize_ROCS_esp_overlay`, with different inputs and width handling |
| `vol_and_surf_esp` | `optimize_esp_combo_score_overlay` |
| `vol_mr`, `vol_fukui` | `optimize_vol_lipo_overlay`, substituting the field |
| `vol_pharm` | `optimize_vol_color_overlay`, `directionless=False` |
| `surf_tversky` | `optimize_vol_tversky_overlay`, surface positions |
| `surf_esp_tversky` | `optimize_vol_esp_tversky_overlay`, surface positions and potentials |
| `pharm_tversky` | `optimize_pharm_overlay`, `similarity="tversky"` |
| `vol_avoid` | `optimize_ROCS_overlay`, with avoid cloud and penalty arguments |

## Per-atom feature contract

Follow `get_lipophilicity_contribs`, `get_lipophilicity`, and
`get_lipo_positions` in `_core.py` together:

1. Store the full field in with-H RDKit atom order. Cheap descriptors can be
   eager; expensive descriptors may use a cached lazy property.
2. Compute it from the same molecule/conformer used for its positions.
3. Slice with `_nonH_atoms_idx` for `no_H=True`.
4. Supply matching positions from the with-H conformer indexed by the same
   strict-heavy indices. Do not substitute `atom_pos`, which uses RemoveHs.

The matching `MoleculeProfile` accessors in `screen.py` let the same readers
consume stored data. New fields require explicit store support during acceleration.

## Registration and defaults

`_ALIGN_KEYS` declares per-pair results; `SPECS` declares executable accelerated
modes. These are distinct registrations. Do not register a spec that names
unimplemented kernels or channels just to obtain default values.

Public reference defaults and batched defaults must be checked separately.
For example, `align_with_vol_tversky` currently defaults to 50 repeats and
200 steps, whereas its accelerated spec has 10 seeds and 40 steps. Use explicit
settings in comparisons. Accelerated `num_repeats` overrides seeds only where
`ModeSpec.honors_num_repeats` permits it; do not infer the effective budget
from the public argument alone.
