# ColQwen3 Embeddings

Multi-vector document embeddings using [Ops-Colqwen3-4B](https://huggingface.co/OpenSearch-AI/Ops-Colqwen3-4B), implemented in Rust on the Crane inference engine.

Produces ColBERT-style per-token embeddings for images and text queries, scored with MaxSim late interaction.

## Architecture

```
Image  -->  Qwen3-VL Vision Encoder (24 layers)
                      |
                      v
         Qwen3-VL Text Decoder (36 layers)  <--  Visual prompt
                      |
                      v
              custom_text_proj (Linear 2560 -> 2560)
                      |
                      v
                 L2 Normalize
                      |
                      v
            Per-token embeddings (seq_len, 2560)
```

Query encoding follows the same decoder path but without the vision encoder.

## Quick start

### Download model

```bash
# BF16 (8.5 GB)
huggingface-cli download OpenSearch-AI/Ops-Colqwen3-4B --local-dir ./checkpoints/colqwen3
```

### Build

```bash
cargo build --release --features cuda -p crane-examples --bin embedding_simple
```

### Run

```bash
./target/release/embedding_simple \
    ./checkpoints/colqwen3 \
    ./images \
    "tabla de accionistas" \
    --top-k 4 \
    --bf16
```

Output:

```
Found 21 images in ./images
Loading ColQwen3 model from: ./checkpoints/colqwen3 (bf16=true)
Model loaded in 1.66s

Encoding 21 images...
  [21/21] encoded
Images encoded in 5.31s (0.25s/image)

Encoding query: "tabla de accionistas"
Query encoded in 0.08s

Computing MaxSim scores...
Scoring done in 0.0046s

============================================================
Top 4 results for: "tabla de accionistas"
============================================================
1. documento_index_3.jpg (score: 11.3750)
2. documento_index_4.jpg (score: 11.3750)
3. documento_index_7.jpg (score: 11.3125)
4. documento_index_2.jpg (score: 10.6875)
```

## API

### `ColQwen3Emb`

```rust
use crane_core::models::colqwen3_emb::ColQwen3Emb;

// Load model
let mut model = ColQwen3Emb::from_local("./checkpoints/colqwen3", false, true)?;

// Encode images -> Vec<Tensor>, each (seq_len, 2560)
let image_embs = model.encode_images(&["doc1.jpg", "doc2.jpg"])?;

// Encode queries -> Vec<Tensor>, each (seq_len, 2560)
let query_embs = model.encode_queries(&["shareholder table"])?;

// MaxSim scoring -> Tensor (num_queries, num_images)
let scores = ColQwen3Emb::score(&query_embs, &image_embs, 128)?;
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `path` | Directory with `config.json`, `tokenizer.json`, `preprocessor_config.json`, `*.safetensors` |
| `cpu` | Force CPU inference (slow, mainly for testing) |
| `bf16` | Use BF16 on CUDA (recommended: ~50% less VRAM, same quality) |

## Validation against Python

Tested against `transformers` + `AutoModel.from_pretrained("OpenSearch-AI/Ops-Colqwen3-4B")` on 21 document images:

| Rank | Image | Python (BF16) | Rust (BF16) |
|------|-------|--------------|-------------|
| 1 | documento_index_3.jpg | 11.4375 | 11.3750 |
| 2 | documento_index_4.jpg | 11.3750 | 11.3750 |
| 3 | documento_index_7.jpg | 11.3125 | 11.3125 |
| 4 | documento_index_2.jpg | 10.6875 | 10.6875 |
| 5 | documento_index_1.jpg | 10.5625 | 10.5625 |

- Top-5 ranking identical
- 9/21 scores exactly equal
- Max delta 0.375 in mid-range (BF16 accumulation order differences between PyTorch and Candle)

## Performance (RTX 4090)

| Operation | Time |
|-----------|------|
| Model load | 1.7s |
| Image encode | 0.25s/image |
| Query encode | 0.08s |
| MaxSim (21 images) | 0.005s |

## Files changed

```
crane-core/src/models/colqwen3_emb/
  mod.rs           -- module registration
  model.rs         -- ColQwen3Emb: config, model, encode_images, encode_queries, score

crane-core/src/models/qwen3_vl/model.rs
  -- Made VisionModel, TextDecoder, MRoPE pub for reuse
  -- Added TextDecoder::forward_hidden() for embedding (no lm_head)
  -- Made lm_head optional (saves 1.5 GB for embedding-only models)

crane-core/src/models/mod.rs
  -- Added pub mod colqwen3_emb

example/src/embedding_simple.rs
  -- CLI binary: load model, encode images, encode query, score, top-k

example/Cargo.toml
  -- Registered embedding_simple binary
```
