"""Mismo oráculo pero con dtype=bfloat16 para comparar contra Crane (también bf16)."""
import os, sys, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/miguel/Documentos/nebuia-embs/colqwen3-4b/scripts")
from ops_colqwen3_embedder import OpsColQwen3Embedder

OUT_DIR = "/home/miguel/Documentos/embs-engine/Crane/test_embeddings"
img_a = Image.open(os.path.join(OUT_DIR, "images/00_white_32.png")).convert("RGB")
img_b = Image.open(os.path.join(OUT_DIR, "images/01_black_16.png")).convert("RGB")

queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
]

embedder = OpsColQwen3Embedder(
    model_name="/home/miguel/Documentos/nebuia-embs/colqwen3-4b",
    dims=2560,
    dtype=torch.bfloat16,                       # ← BF16 igual que Crane
    attn_implementation="flash_attention_2",
)

q_embs = embedder.encode_queries(queries)
i_embs = embedder.encode_images([img_a, img_b])

print("Query[0] shape:", tuple(q_embs[0].shape))
print("Image[0] shape:", tuple(i_embs[0].shape))

scores = embedder.compute_scores(q_embs, i_embs)
print("Scores (bf16 Python):")
print(scores)

np.savez(
    os.path.join(OUT_DIR, "python_oracle_bf16.npz"),
    query0=q_embs[0].to(torch.float32).cpu().numpy(),
    query1=q_embs[1].to(torch.float32).cpu().numpy(),
    image0=i_embs[0].to(torch.float32).cpu().numpy(),
    image1=i_embs[1].to(torch.float32).cpu().numpy(),
    scores=scores.to(torch.float32).cpu().numpy(),
)
