import torch, torch.multiprocessing as mp
def w(q, label):
    try:
        torch.cuda.set_device(0); torch.zeros(1, device="cuda:0").sum().item()
        q.put(f"{label}: OK (device works in fork child)")
    except Exception as e:
        q.put(f"{label}: FAIL: {e}")
if __name__ == "__main__":
    ctx = mp.get_context("fork")
    q = ctx.Queue(); p = ctx.Process(target=w, args=(q, "case1 clean-parent")); p.start(); print(q.get(), flush=True); p.join()
    n = torch.cuda.device_count()  # this calls cuInit
    print(f"  (parent device_count={n}, is_initialized={torch.cuda.is_initialized()})", flush=True)
    q = ctx.Queue(); p = ctx.Process(target=w, args=(q, "case2 after device_count")); p.start(); print(q.get(), flush=True); p.join()
    # forkserver case
    fs = mp.get_context("forkserver"); fs.set_forkserver_preload(["torch"])
    q = fs.Queue(); p = fs.Process(target=w, args=(q, "case3 forkserver")); p.start(); print(q.get(), flush=True); p.join()
