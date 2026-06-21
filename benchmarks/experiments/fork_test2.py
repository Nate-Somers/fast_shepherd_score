import torch, torch.multiprocessing as mp
def w(q, label):
    try:
        import shepherd_score.container as c  # preloaded in forkserver -> no-op
        torch.cuda.set_device(0); torch.zeros(1, device="cuda:0").sum().item()
        q.put(f"{label}: OK")
    except Exception:
        import traceback; q.put(f"{label}: FAIL: {traceback.format_exc().splitlines()[-1]}")
if __name__ == "__main__":
    import shepherd_score.container  # main imports the (triton) stack -> reproduces poison
    fs = mp.get_context("forkserver")
    fs.set_forkserver_preload(["torch", "numpy", "shepherd_score.container"])
    q = fs.Queue(); p = fs.Process(target=w, args=(q, "A forkserver+shepherd-preload")); p.start(); print(q.get(), flush=True); p.join()
    fk = mp.get_context("fork")
    q = fk.Queue(); p = fk.Process(target=w, args=(q, "B fork-after-shepherd-import")); p.start(); print(q.get(), flush=True); p.join()
