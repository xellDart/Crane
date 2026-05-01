"""Compara salida Crane (.bin f32) vs oráculo Python (.npz)."""
import os, sys
import numpy as np

OUT_DIR = "/home/miguel/Documentos/embs-engine/Crane/test_embeddings"

oracle = np.load(os.path.join(OUT_DIR, "python_oracle.npz"))

def load_bin(name, shape):
    path = os.path.join(OUT_DIR, name)
    arr = np.fromfile(path, dtype=np.float32)
    return arr.reshape(shape)

def cosine_sim(a, b):
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    return float((a_flat @ b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-30))

def cosine_per_row(a, b):
    """Per-row cosine: a (N,D) vs b (N,D)."""
    n = min(a.shape[0], b.shape[0])
    sims = []
    for i in range(n):
        ai = a[i].astype(np.float64); bi = b[i].astype(np.float64)
        s = (ai @ bi) / (np.linalg.norm(ai) * np.linalg.norm(bi) + 1e-30)
        sims.append(s)
    return sims

def report(name, py, cr):
    print(f"\n--- {name} ---")
    print(f"  python: {py.shape}    crane: {cr.shape}")
    if py.shape != cr.shape:
        # Try aligning right (Python may have left-padding zeros)
        n_py = py.shape[0]; n_cr = cr.shape[0]
        if n_py > n_cr:
            # Python padded — compare last N rows
            py_aligned = py[-n_cr:]
            print(f"  shape mismatch — comparing last {n_cr} rows of python vs crane")
            print(f"    flat cosine = {cosine_sim(py_aligned, cr):.6f}")
            sims = cosine_per_row(py_aligned, cr)
            print(f"    per-row cosine: min={min(sims):.6f} mean={np.mean(sims):.6f} max={max(sims):.6f}")
            # Check if Python's leading rows are zero (pads)
            leading = py[:n_py - n_cr]
            print(f"    leading {leading.shape[0]} python rows: max abs = {np.max(np.abs(leading)):.6e}")
        else:
            cr_aligned = cr[-n_py:]
            print(f"  shape mismatch — comparing last {n_py} rows of crane vs python")
            print(f"    flat cosine = {cosine_sim(cr_aligned, py):.6f}")
    else:
        print(f"  flat cosine = {cosine_sim(py, cr):.6f}")
        sims = cosine_per_row(py, cr)
        print(f"  per-row cosine: min={min(sims):.6f} mean={np.mean(sims):.6f} max={max(sims):.6f}")
        diff = py - cr
        print(f"  per-row L2 diff: mean={np.mean(np.linalg.norm(diff,axis=1)):.6e} max={np.max(np.linalg.norm(diff,axis=1)):.6e}")

def shape(name):
    arr = oracle[name]
    return arr.shape

report("query0", oracle["query0"], load_bin("crane_query0.bin", (-1, 2560)))
report("query1", oracle["query1"], load_bin("crane_query1.bin", (-1, 2560)))
report("image0", oracle["image0"], load_bin("crane_image0.bin", (-1, 2560)))
report("image1", oracle["image1"], load_bin("crane_image1.bin", (-1, 2560)))

py_scores = oracle["scores"]
cr_scores = np.fromfile(os.path.join(OUT_DIR, "crane_scores.bin"), dtype=np.float32).reshape(2, 2)
print(f"\n--- scores ---")
print(f"  python:\n{py_scores}")
print(f"  crane:\n{cr_scores}")
print(f"  abs diff: {np.abs(py_scores - cr_scores)}")
print(f"  rel diff: {np.abs(py_scores - cr_scores) / np.abs(py_scores)}")

# Ranking check
py_rank = np.argsort(-py_scores, axis=1)
cr_rank = np.argsort(-cr_scores, axis=1)
print(f"  python ranking: {py_rank.tolist()}")
print(f"  crane  ranking: {cr_rank.tolist()}")
print(f"  RANKINGS MATCH: {(py_rank == cr_rank).all()}")
