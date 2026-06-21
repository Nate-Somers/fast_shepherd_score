import sys, torch, torch.multiprocessing as mp
mod = sys.argv[1] if len(sys.argv) > 1 else "none"
if mod != "none":
    __import__(mod)
def w(q):
    try:
        torch.cuda.set_device(0); torch.zeros(1, device="cuda:0").sum().item(); q.put("OK")
    except Exception as e:
        q.put("FAIL: " + str(e)[:70])
if __name__ == "__main__":
    ctx = mp.get_context("fork"); q = ctx.Queue(); p = ctx.Process(target=w, args=(q,)); p.start()
    print(f"  preimport={mod:55s} -> {q.get()}", flush=True); p.join()
