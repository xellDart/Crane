"""Captura salida del modelo Python como oráculo.

Genera 2 imágenes sintéticas (32x32 blanca, 16x16 negra) y 2 queries.
Guarda embeddings + matriz de scores para comparar contra Crane.
"""
import os
import sys
import json
import numpy as np
import torch
from PIL import Image

# Hacer determinístico
torch.use_deterministic_algorithms(False)  # flash-attn no es det por defecto, pero está OK
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Cargar el embedder de la misma carpeta del modelo
sys.path.insert(0, "/home/miguel/Documentos/nebuia-embs/colqwen3-4b/scripts")
from ops_colqwen3_embedder import OpsColQwen3Embedder

OUT_DIR = "/home/miguel/Documentos/embs-engine/Crane/test_embeddings"
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# 2 imágenes sintéticas exactamente como en el ejemplo del usuario
img_a = Image.new("RGB", (32, 32), color="white")
img_b = Image.new("RGB", (16, 16), color="black")

# Guardar como PNG (lossless)
img_a.save(os.path.join(IMG_DIR, "00_white_32.png"))
img_b.save(os.path.join(IMG_DIR, "01_black_16.png"))

queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
]

# Cargar modelo (local, fp16, flash-attn 2 — igual al ejemplo del usuario)
embedder = OpsColQwen3Embedder(
    model_name="/home/miguel/Documentos/nebuia-embs/colqwen3-4b",
    dims=2560,
    dtype=torch.float16,
    attn_implementation="flash_attention_2",
)

# Encoding
q_embs = embedder.encode_queries(queries)
i_embs = embedder.encode_images([img_a, img_b])

print("Query[0] shape:", tuple(q_embs[0].shape))
print("Query[1] shape:", tuple(q_embs[1].shape))
print("Image[0] shape:", tuple(i_embs[0].shape))
print("Image[1] shape:", tuple(i_embs[1].shape))

# Score
scores = embedder.compute_scores(q_embs, i_embs)
print("Scores:")
print(scores)

# Guardar todo
np.savez(
    os.path.join(OUT_DIR, "python_oracle.npz"),
    query0=q_embs[0].to(torch.float32).cpu().numpy(),
    query1=q_embs[1].to(torch.float32).cpu().numpy(),
    image0=i_embs[0].to(torch.float32).cpu().numpy(),
    image1=i_embs[1].to(torch.float32).cpu().numpy(),
    scores=scores.to(torch.float32).cpu().numpy(),
)

# Guardar metadatos
meta = {
    "queries": queries,
    "image_paths": ["00_white_32.png", "01_black_16.png"],
    "shapes": {
        "query0": list(q_embs[0].shape),
        "query1": list(q_embs[1].shape),
        "image0": list(i_embs[0].shape),
        "image1": list(i_embs[1].shape),
    },
    "scores": scores.cpu().tolist(),
}
with open(os.path.join(OUT_DIR, "python_oracle_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved to:", os.path.join(OUT_DIR, "python_oracle.npz"))
