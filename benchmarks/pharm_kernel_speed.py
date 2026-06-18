"""Speed A/B: Triton pharm value+grad kernel vs analytical torch (same inputs)."""
import time
import torch

from shepherd_score.score.analytical_gradients._torch import (
    compute_overlap_and_grad_pharm, build_lookup_tables, _rotation_matrix_from_unit_quat)
from shepherd_score.score.pharmacophore_grad_triton import pharm_score_grad_se3_batch


def timeit(fn, reps=30):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    dev = torch.device("cuda")
    alphas, Ks, cats = build_lookup_tables(dev, torch.float32)
    print(f'{"P":>7} {"N":>3} | {"analytic ms":>11} {"kernel ms":>10} {"speedup":>8}')
    for P, N in [(10000, 8), (10000, 16), (20000, 16), (50000, 16)]:
        g = torch.Generator(device=dev).manual_seed(0)
        q = torch.randn(P, 4, device=dev, generator=g); q = q / q.norm(dim=1, keepdim=True)
        R = _rotation_matrix_from_unit_quat(q)
        t = torch.randn(P, 3, device=dev, generator=g)
        ra = torch.randn(P, N, 3, device=dev, generator=g); fa = torch.randn(P, N, 3, device=dev, generator=g)
        rv = torch.randn(P, N, 3, device=dev, generator=g); fv = torch.randn(P, N, 3, device=dev, generator=g)
        rt = torch.randint(0, alphas.shape[0], (P, N), device=dev, generator=g)
        ft = torch.randint(0, alphas.shape[0], (P, N), device=dev, generator=g)

        ta = timeit(lambda: compute_overlap_and_grad_pharm(R, t, rt, ft, ra, fa, rv, fv, extended_points=False))
        tk = timeit(lambda: pharm_score_grad_se3_batch(R, t, rt, ft, ra, fa, rv, fv, alphas, Ks, cats))
        print(f'{P:7d} {N:3d} | {ta*1e3:11.2f} {tk*1e3:10.2f} {ta/tk:7.2f}x', flush=True)
        del ra, fa, rv, fv, R, t
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
