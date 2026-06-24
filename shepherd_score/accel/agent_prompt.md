# Agent prompt — add a customization to the fast_shepherd_score backend

Paste-ready task brief for a coding agent. Fill in the `<<...>>` placeholders and delete
this line. The full reference is `fast_shepherd_score/docs/ADDING_A_FAST_MODE.md` — read it
before starting.

---

You are working in the **fast_shepherd_score** repo (a GPU/CPU-accelerated fork of
shepherd-score). Code lives under `fast_shepherd_score/shepherd_score/`. Your task is to add
the following customization and make it run on the **fast batched backend** (fused Triton /
numba kernels), not just the slow per-pair path:

> **Customization:** <<DESCRIBE IT — e.g. "a new alignment mode `vol_hbond` that blends atom
> shape Tanimoto with a hydrogen-bond-count term", or "fold a per-atom lipophilicity weight
> into the existing `vol` mode", or "speed up mode X whose kernel still emits grad_R">>
>
> **Do you already have a working PyTorch autograd objective for this** (a `loss.backward()`
> optimizer that produces correct alignments)? <<YES / NO>>. If YES, this is the easy path:
> (a) reshape it to the `objective_<mode>_overlay` contract — it becomes your **gradient
> oracle**; (b) batch it over pairs for a no-kernel GPU speedup and STOP if that's fast enough
> (autograd-batch shortcut below); (c) only if the per-step autograd is the bottleneck, port to
> a fused Triton+numba kernel — **follow the [Recipe](#recipe--turn-your-autograd-objective-into-a-tritonnumba-accel-method)
> section below, which gives the exact kernel/driver code with every optimization labeled.**
> Playbook Scenario C has the same path with more rationale.

## Before you write anything

1. Read `fast_shepherd_score/docs/ADDING_A_FAST_MODE.md` in full — it is the authoritative
   playbook (mental model, decision guide, Scenarios A/B/C, validation gates, pitfalls).
   Follow it; do not improvise an alternate architecture. **If you already have an autograd
   objective, Scenario C is your path.**
2. Read the closest existing template end-to-end and copy its structure:
   - JOINT two-channel combo → `accel/drivers/vol_color.py` + `_align_batch_vol_color` in
     `accel/batch/aligners.py`.
   - Pure shape-only mode → `accel/drivers/surface.py`.
   - Speeding up an existing slow mode (grad_R → in-register dO/dq) → the `pharm` change:
     `accel/kernels/pharm_triton.py:pharm_grad_dq_se3_batch` + `accel/drivers/pharm.py`.
   - SE(3)-invariant per-pair weight folded into a kernel → how ESP rode the shape kernel:
     `accel/kernels/esp_triton.py`.

## Non-negotiable invariants (these break silently if violated)

- **Two backends, two layers.** `backend=` in `container/_batch.py` only chooses
  jax-vs-fast and (for fast) whether to move tensors to CPU first. The Triton-vs-numba
  choice is made **per call by tensor device** in `accel/kernels/dispatch.py`. So: never
  branch on backend inside a driver, and **every new kernel must exist as BOTH a Triton
  kernel (`accel/kernels/*_triton.py`) and a numba twin with an identical signature
  (`accel/kernels/cpu.py`), registered once via `dispatch._make(name, tag)`.**
- **Prefer in-register dO/dq:** the kernel takes the unit quaternion `q`, builds `R(q)` in
  registers, and emits `dO/dq` directly (REF=axis0, FIT=axis1, `dx = ref - rot(fit)`,
  `tl.exp2`, gradient gated behind `NEED_GRAD: tl.constexpr`). The driver applies the
  Tanimoto/Tversky scalar scale straight to `dO/dq` — no host R→q projection. Keep only the
  unit→raw normalization handled by `fused_adam_qt_with_tangent_proj`.
- **Self-overlaps (VAA/VBB) MUST go through the same kernel** as the cross-overlap (identity
  quaternion, zero translation, `NEED_GRAD=False`).
- **Seed from the shape/atom cloud**, never the pharmacophore anchors. Carry separate int32
  `N_real`/`M_real` count tensors for centers vs anchors; pad pharm type slots with
  `Dummy = 8`. Band-pad every size dimension (`accel/batch/_pad.py:_band_key`).
- **Implement the per-pair eager `optimize_<mode>_overlay` in `alignment/_torch.py` FIRST** —
  it is the autograd ground truth your kernels are validated against. *(If you already have a
  working autograd objective, this is done — reshape it to the `objective_<mode>_overlay`
  contract and use it as your oracle.)*

## Recipe — turn your autograd objective into a Triton+numba accel method

This is the exact procedure for the hard part: writing the fused kernel + driver from a working
autograd objective `O(ref, R(q)·fit + t)` → similarity `S = f(O)`. Every optimization the repo
relies on is marked `[opt]` and applied below. You change **two lines** of a copied kernel; the
rest is verbatim.

### Step 0 — factor your objective into (value, force, similarity)
- **value** `O = Σ_{i,j} contribution(ref_i, fit_j_transformed)` — the pairwise reduction; the
  kernel returns this.
- **force** `F_j = ∂O/∂(fit_j_transformed)` (a 3-vector per fit atom) — **the only new math you
  derive.** Get/verify it from autograd: `torch.autograd.grad(O.sum(), fit_transformed)`.
- **similarity** `S = f(O)` (Tanimoto `O/(VAA+VBB−O)`, Tversky, blend) — stays in the **driver**,
  never in the kernel `[opt: chain-rule-in-driver]`.

Key fact `[opt: gradient factorization]`: for any rigid SE(3) transform the rotation gradient is
`dO/dq = Σ_j F_j · ∂(R(q)·b0_j)/∂q` and `dO/dt = Σ_j F_j`. The factor `∂(R(q)·b0)/∂q` (**the dR/dq
tail**) is *identical for every mode* — you copy it verbatim and never re-derive the rotation
gradient. For the Gaussian shape overlap: `contribution = K·exp(−½α r²)`, `K = π^1.5/(2α)^1.5`,
and `F_j = Σ_i (α·g_ij)·(ref_i − fit_jt)`.

### Step 1 — Triton kernel  (`accel/kernels/<family>_triton.py`)
Copy `overlap_score_grad_esp_se3_batch` + `_gauss_overlap_esp_se3_tiled` and edit only the two
`<-- YOUR` lines. Structure (annotated):

```python
from .shape_triton import _OVERLAP_CONFIGS
@triton.autotune(configs=_OVERLAP_CONFIGS, key=['N_pad','M_pad'], cache_results=True)  # [opt: autotune, triton>=3.6]
@triton.jit
def my_score_grad_se3_batch(A_ptr,B_ptr,Q_ptr,T_ptr,Nreal_ptr,Mreal_ptr,
                            BATCH,M_pad,N_pad, half_alpha,k_const,  # + extra scalar params if needed
                            S_ptr,dQ_ptr,dT_ptr,
                            BLOCK: tl.constexpr, NEED_GRAD: tl.constexpr):  # [opt: fused value+grad, DCE when False]
    pid = tl.program_id(0)                                          # [opt: one CTA per pose]
    realN = tl.load(Nreal_ptr+pid); realM = tl.load(Mreal_ptr+pid) # [opt: padding masks via N_real/M_real]
    # offset all *_ptr by pid; load UNIT quaternion qr,qi,qj,qk + tx,ty,tz   [opt: in-register dO/dq -> take q, not R]
    # build rotation r00..r22 from (qr,qi,qj,qk) in registers   (copy verbatim)
    Vacc=0.0; dTx=dTy=dTz=0.0; dQw=dQx=dQy=dQz=0.0; inv_ln2=1.4426950408889634
    for n0 in range(0,N_pad,BLOCK):                 # outer tile = REF  [opt: REF=axis0]
        # load A tile ax,ay,az ; mask_n = offs_n<realN
        for m0 in range(0,M_pad,BLOCK):             # inner tile = FIT  [opt: FIT=axis1]
            # load B tile body-frame bx0,by0,bz0 ; mask_m = offs_m<realM
            bx=r00*bx0+r01*by0+r02*bz0+tx; by=...; bz=...
            dx=ax[:,None]-bx[None,:]; dy=...; dz=...              # [opt: dx = ref - rot(fit)]
            r2=dx*dx+dy*dy+dz*dz
            g = tl.exp2((-half_alpha*r2)*inv_ln2)*k_const        # <-- YOUR value term   [opt: exp2 not exp]
            g = tl.where(mask_n[:,None]&mask_m[None,:], g, 0.0)
            Vacc += tl.sum(g)
            if NEED_GRAD:
                coeff = (2.0*half_alpha)*g                        # <-- YOUR force coeff (= α·g for Gaussian)
                fx=tl.sum(coeff*dx,0); fy=tl.sum(coeff*dy,0); fz=tl.sum(coeff*dz,0)  # F_j per FIT atom
                dTx+=tl.sum(fx); dTy+=tl.sum(fy); dTz+=tl.sum(fz)        # dO/dt = Σ F_j
                # ---- dR/dq tail: VERBATIM from the shape kernel, identical for every mode ----
                dw =fx*(-2*qk*by0+2*qj*bz0)+fy*(2*qk*bx0-2*qi*bz0)+fz*(-2*qj*bx0+2*qi*by0)
                dxq=fx*(2*qj*by0+2*qk*bz0)+fy*(2*qj*bx0-4*qi*by0-2*qr*bz0)+fz*(2*qk*bx0+2*qr*by0-4*qi*bz0)
                dyq=fx*(-4*qj*bx0+2*qi*by0+2*qr*bz0)+fy*(2*qi*bx0+2*qk*bz0)+fz*(-2*qr*bx0+2*qk*by0-4*qj*bz0)
                dzq=fx*(-4*qk*bx0-2*qr*by0+2*qi*bz0)+fy*(2*qr*bx0-4*qk*by0+2*qj*bz0)+fz*(2*qi*bx0+2*qj*by0)
                dQw+=tl.sum(tl.where(mask_m,dw,0.)); dQx+=...; dQy+=...; dQz+=...
    tl.store(S_ptr+pid, Vacc)
    if NEED_GRAD: store dQ_ptr[0..3]=dQw..dQz, dT_ptr[0..2]=dTx..dTz
```
Host wrapper: `grid=(K,)`, flatten with `.contiguous().view(-1)`, `N_real`/`M_real` as int32,
allocate `out_S/out_dQ/out_dT`, launch. Copy `overlap_score_grad_esp_se3_batch`'s wrapper.

### Step 2 — numba twin  (`accel/kernels/cpu.py`), IDENTICAL public name + signature
Copy `_overlap_grad_kernel` + `overlap_score_grad_se3_batch` and edit the same two lines:
```python
@njit(parallel=True, fastmath=False, cache=True)        # [opt: numba twin]
def _my_grad_kernel(A,B,q,t,Nr,Mr,alpha,need_grad):
    K=A.shape[0]; Kc=math.pi**1.5/(2*alpha)**1.5; a2=alpha/2
    V=np.zeros(K); dQ=np.zeros((K,4)); dT=np.zeros((K,3))            # [opt: fp64 accumulation]
    for k in prange(K):                                              # [opt: one pose per prange iter]
        # build r00..r22; for m in range(Mr[k]): for n in range(Nr[k]):
        #   dx=A[k,n,0]-bx; ...; g=Kc*math.exp(-a2*r2)               # [opt: math.exp = only divergence vs exp2]
        #   Vacc+=g; if need_grad: c=alpha*g; fxj+=c*dx; ...
        #   if need_grad: dT+=force; dQ += SAME verbatim dR/dq tail
    return V,dQ,dT
def my_score_grad_se3_batch(A,B,q,t,*,alpha=0.81,N_real=None,M_real=None,NEED_GRAD=True,**_):
    # .detach().cpu().numpy() in -> _my_grad_kernel(...) -> torch.as_tensor(..., device=A.device, dtype=A.dtype) out
```

### Step 3 — register  (`accel/kernels/dispatch.py`)
`my_score_grad_se3_batch = _make("my_score_grad_se3_batch", "<shape|esp|pharm>")`  → device picks
Triton (CUDA) / numba (CPU) per call `[opt: per-call device dispatch]`. The Triton and numba
functions MUST share this exact name.

### Step 4 — driver fine loop  (`accel/drivers/<mode>.py`; copy `surface.py` or `vol_color.py`)
```python
quats,t_seeds = batched_seeds_torch(A,B,N_real,M_real,num_seeds)      # [opt: seeds from shape cloud]
# expand A,B,q,t over P seeds; m_q=v_q=m_t=v_t=zeros_like
VAA = my_score_grad_se3_batch(A,A,q_id,t0,N_real=N,M_real=N,NEED_GRAD=False)[0]   # [opt: self-overlap via SAME kernel]
VBB = my_score_grad_se3_batch(B,B,q_id,t0,N_real=M,M_real=M,NEED_GRAD=False)[0]
for step in range(steps_fine):
    O,dQ,dT = my_score_grad_se3_batch(A,B,q,t,alpha=alpha,N_real=N,M_real=M,NEED_GRAD=True)
    denom = VAA+VBB-O; score = O/denom; scale = (VAA+VBB)/(denom*denom)    # [opt: Tanimoto chain rule IN DRIVER]
    best = _update_best(score,q,t,...)
    if step%5==0: early-stop with (ES_PATIENCE_OVERRIDE or patience)        # [opt: ES override]
    fused_adam_qt_with_tangent_proj(q,t, -dQ*scale.unsqueeze(1), -dT*scale.unsqueeze(1),
                                    m_q,v_q,m_t,v_t, lr)   # [opt: tangent proj + renorm == the unit->raw normalization Jacobian]
# argmax best seed per pair; return (score,q,t)
```
JOINT combo: sum per-channel `g_q=(1-w)*(-scale_a*dQ_a)+w*(-scale_b*dQ_b)` `[opt: joint gradient]`
(both channels emit `dO/dq`, no projection — `vol_color.py:226-232`).

### Step 5 — variants (apply the ones that fit)
- **SE(3)-invariant per-pair weight** (depends only on fixed per-point scalars, e.g. charge):
  multiply `g` by an extra `tl.exp2(-c2*inv_lam*inv_ln2)` factor — the gradient structure is
  UNCHANGED (weight is constant w.r.t. pose), so the same `coeff` + tail still work. This is how
  `esp_triton.py` extends the shape kernel `[opt: SE(3)-invariant weight fold]`.
- **Value-only channel** (the term only scores/selects while another channel steers the pose):
  delete the whole `NEED_GRAD` block, return just `O`, and stream point×atom in registers so you
  never materialize `(K,N,M)`. This is the `esp_combo` `esp_comparison_batch` win (~5–27×)
  `[opt: value-only / no big intermediate]`.
- **Orientation-dependent (directional) coefficient** (e.g. pharmacophore vectors): apply the
  dR/dq tail **twice** — positional force × fit anchor, and weight force × fit vector
  (`pharm_grad_dq_se3_batch`) `[opt: directional dQ]`.

### Step 6 — validate the kernel against your autograd oracle (done-gate 1)
```python
q = unit_quat.double().clone().requires_grad_(True)   # UNIT-norm leaf, float64
R = _rotation_matrix_from_unit_quat(q)                # NOT get_SE3_transform (no in-graph normalize, no fp32)
fit_t = fit.double() @ R.T + t
O = bare_overlap(ref.double(), fit_t)                 # the BARE overlap O, NOT the similarity
(dq_ref,) = torch.autograd.grad(O.sum(), q,  retain_graph=True)   # oracle for kernel dQ
(F_ref,)  = torch.autograd.grad(O.sum(), fit_t)                   # oracle for the kernel force
# assert allclose(numba dQ, dq_ref, atol~1e-12); then assert Triton == numba in fp32
```

## Where to touch (ordered, Scenario A "add a mode" — trim for Scenario B; Scenario C joins at the kernel step after batching the autograd objective)

`score/<family>_scoring.py` (+ `_np`) → `alignment/_torch.py` (+ `__init__.py` export) →
*(optional)* `_torch_analytical.py` + `score/analytical_gradients/_torch.py` lookup tables →
`accel/kernels/<family>_triton.py` (Triton) + `accel/kernels/cpu.py` (numba twin) →
`accel/kernels/dispatch.py` (`_make`) → `accel/drivers/<mode>.py` (driver: seeds → padded
inputs → coarse-to-fine Adam fine loop calling the kernel → early stop → argmax best seed) →
`accel/batch/aligners.py` (`_MODE_SEEDS` entry + `_align_batch_<mode>`) →
`accel/batch/__init__.py` (re-export) → `container/_core.py` (`align_with_<mode>` + bind
staticmethod + `transform_<mode>`/`sim_aligned_<mode>`) → `container/_batch.py`
(`align_with_<mode>` via `_run_fast_or_fallthrough`) → *(optional)* `accel/batch/_dispatch.py`
`_MODE_SPEC` / `accel/cpu_pool.py` / `screen.py` → `tests/test_<mode>.py`.

**Shortcut — minimum viable fast path:** if the customization is a re-blend of existing
channels or an SE(3)-invariant per-pair weight, skip the new kernel: reuse an existing fused
kernel and score the new term in pure torch each fine step (how `esp_combo` works). Only
write a dedicated kernel if profiling shows the per-step torch score+grad is the bottleneck
or the term needs its own coordinate-dependent gradient.

**Shortcut — autograd-batch (no kernel, for an autograd start):** if you already have a working
autograd objective, first just batch it over `(pairs × seeds)` and run the Adam loop on GPU with
`loss.backward()` (no kernel). That alone is usually 1–2 orders of magnitude over the per-pair
loop. Only proceed to a fused kernel when the per-step autograd time/memory is the measured
bottleneck. (Scenario C1.)

## Definition of done — ALL must pass

1. **numba dQ vs torch autograd** of the same objective match to ~1e-17 (CPU tensors). Build the
   autograd reference from a unit-norm leaf `q` via `_rotation_matrix_from_unit_quat` in float64
   (NOT `get_SE3_transform`), and differentiate the bare overlap `O` (not the similarity).
2. **Triton == numba** to fp32 on a GPU box (exp2 vs exp is the only intended divergence).
3. **Self-copy recovers 1.000** for tanimoto AND tversky.
4. **Batched score on distinct pairs == per-pair eager `optimize_<mode>_overlay`.**
5. `MoleculePairBatch.align_with_<mode>(backend='triton')`, `'numba'`, and `'jax'` (or a
   clean `numba_ok=False` refusal) all behave correctly; `MoleculePair.align_with_<mode>`
   works per-pair.
6. New + existing tests pass: `pytest fast_shepherd_score/tests/`.

## Environment

Triton path needs **triton ≥ 3.6** (autotune `cache_results=True`); numba path needs
**numpy ≤ 2.3**. On Windows the working conda env is **GNNenv** (full stack); jax_backend is
`cpu` on the laptop. Run via the env python with `PYTHONPATH` pointed at the repo (do **not**
use `conda run -n GNNenv python` — it mis-parses `--n` flags).

## Report back

Summarize: files touched (in order), which template you copied, the four validation-gate
results with numbers, and any pitfall from the playbook you hit. Do **not** copy the
`vol_color.py` header comment's wording about a "projection tail" into a new mode — that
described an older path; the current code is in-register (see the corrected header).
