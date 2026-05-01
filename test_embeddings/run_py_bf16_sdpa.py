"""bf16 + SDPA estándar (sin flash-attn) — para aislar si flash-attn es el causante."""
import os, sys, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/home/miguel/Documentos/nebuia-embs/colqwen3-4b/scripts")
from ops_colqwen3_embedder import OpsColQwen3Embedder

OUT_DIR = "/home/miguel/Documentos/embs-engine/Crane/test_embeddings"
img_a = Image.open(os.path.join(OUT_DIR, "images/00_white_32.png")).convert("RGB")
img_b = Image.open(os.path.join(OUT_DIR, "images/01_black_16.png")).convert("RGB")
queries = ["Is attention really all you need?", "What is the amount of bananas farmed in Salvador?"]

embedder = OpsColQwen3Embedder(
    model_name="/home/miguel/Documentos/nebuia-embs/colqwen3-4b",
    dims=2560,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",   # ← sin flash-attn
)

q_embs = embedder.encode_queries(queries)
i_embs = embedder.encode_images([img_a, img_b])
scores = embedder.compute_scores(q_embs, i_embs)
print("Scores (Python bf16 + SDPA, no flash-attn):")
print(scores)
np.savez(os.path.join(OUT_DIR, "python_oracle_bf16_sdpa.npz"),
    query0=q_embs[0].to(torch.float32).cpu().numpy(),
    image0=i_embs[0].to(torch.float32).cpu().numpy(),
    image1=i_embs[1].to(torch.float32).cpu().numpy(),
    scores=scores.to(torch.float32).cpu().numpy())
