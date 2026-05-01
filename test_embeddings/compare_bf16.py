"""Compara Crane (bf16) contra Python (también bf16)."""
import os, numpy as np

OUT_DIR = "/home/miguel/Documentos/embs-engine/Crane/test_embeddings"
oracle = np.load(os.path.join(OUT_DIR, "python_oracle_bf16.npz"))

def load_bin(name, shape):
    return np.fromfile(os.path.join(OUT_DIR, name), dtype=np.float32).reshape(shape)

def cosine_per_row(a, b):
    n = min(a.shape[0], b.shape[0])
    sims = []
    for i in range(n):
        ai = a[i].astype(np.float64); bi = b[i].astype(np.float64)
        sims.append((ai @ bi) / (np.linalg.norm(ai) * np.linalg.norm(bi) + 1e-30))
    return sims

def report(name, py, cr):
    print(f"\n--- {name} ---  python:{py.shape}  crane:{cr.shape}")
    if py.shape[0] > cr.shape[0]:
        py = py[-cr.shape[0]:]  # right-align (Python left-pads queries with zeros)
    sims = cosine_per_row(py, cr)
    diff = py - cr
    l2 = np.linalg.norm(diff, axis=1)
    print(f"  per-row cosine: min={min(sims):.7f} mean={np.mean(sims):.7f} max={max(sims):.7f}")
    print(f"  per-row L2 diff: mean={np.mean(l2):.4e} max={np.max(l2):.4e}")
    print(f"  max abs diff: {np.max(np.abs(diff)):.4e}")
    print(f"  bit-exact rows: {sum(1 for d in l2 if d == 0.0)} / {len(l2)}")

report("query0", oracle["query0"], load_bin("crane_query0.bin", (-1, 2560)))
report("query1", oracle["query1"], load_bin("crane_query1.bin", (-1, 2560)))
report("image0", oracle["image0"], load_bin("crane_image0.bin", (-1, 2560)))
report("image1", oracle["image1"], load_bin("crane_image1.bin", (-1, 2560)))

py_s = oracle["scores"]
cr_s = np.fromfile(os.path.join(OUT_DIR, "crane_scores.bin"), dtype=np.float32).reshape(2, 2)
print(f"\n--- scores ---")
print(f"  python bf16:\n{py_s}")
print(f"  crane  bf16:\n{cr_s}")
print(f"  abs diff: {np.abs(py_s - cr_s)}")
print(f"  bit-exact: {(py_s == cr_s).all()}")
