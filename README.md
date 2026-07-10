# Crane — ColQwen3 embeddings fork

A trimmed fork of [Crane](https://github.com/lucasjinreal/Crane) reduced to a single job: **fast ColQwen3 multi-vector embeddings in Rust**. It is the in-process inference backend for [nebuia-embs](https://github.com/xellDart/nebuia-embs) — a `crane-core` **library crate**, not a server.

The general-purpose inference engine (chat, TTS, OCR, multi-model serving, OpenAI-compatible HTTP API, `crane-ctl` daemon) has been **removed**. What remains is the forward-only ColQwen3 path: a Qwen3-VL vision+text backbone that turns document page images and text queries into late-interaction (ColBERT-style) multi-vector embeddings, plus a MaxSim scorer.

## What it is

- **Library crate `crane-core`** — `ColQwen3Emb`: load weights, encode images/queries, score with MaxSim.
- **Two example binaries** (`example/`):
  - `embedding_simple` — rank a folder of page images against a text query.
  - `embedding_oracle_test` — dump embeddings/scores as raw f32 to bit-compare against the Python reference.
- **Forward-only.** No autoregressive decode, no KV cache, no sampling, no `lm_head`, no continuous batching — embeddings only. That is what keeps it lean.

## Build

Cargo workspace of two crates: `crane-core` (library) + `example` (binaries). The default build is CPU / F32.

```bash
git clone -b feat/colqwen3-embeddings https://github.com/xellDart/Crane.git && cd Crane

# Auto-detect CUDA + GPU compute capability, enabling flash-attn on Ampere+ (SM_80+)
bash build.sh            # flags: --cpu, --no-flash, --no-clean

# Or manual
cargo build --release --features flash-attn   # CUDA + Flash Attention (SM_80+)
cargo build --release --features cuda          # CUDA, no flash-attn
cargo build --release                          # CPU / F32
```

| Feature | Effect |
|---------|--------|
| `cuda` | CUDA backend + compiles the fused CUDA kernels (PTX built at compile time) |
| `flash-attn` | implies `cuda`; adds Flash Attention v2 (requires SM_80+) |
| `cudnn` / `mkl` / `accelerate` | optional backend accelerators |

BF16 is used only on CUDA (`bf16 && cuda`); CPU builds run F32.

## Library usage

```rust
use crane_core::models::colqwen3_emb::ColQwen3Emb;

// Load ColQwen3 weights from a HF-style directory (config.json,
// preprocessor_config.json, tokenizer.json, *.safetensors — mmap'd).
let mut model = ColQwen3Emb::from_local("./ops-colqwen3-4b", /*cpu=*/false, /*bf16=*/true)?;
model.set_dims(1280); // Matryoshka truncation (<= projection dims, default 2560)

// Encode page images -> one (num_tokens, dims) tensor per page
let pages = model.encode_images(&["p1.jpg", "p2.jpg"])?;
// or straight from memory, no disk I/O:
let pages = model.encode_images_from_bytes(&[jpeg_bytes])?;

// Encode a text query (adds the "Query: " prefix + token augmentation)
let q = model.encode_queries(&["tabla de accionistas"])?;

// Score. Either one-shot...
let scores = ColQwen3Emb::score(&q, &pages, /*batch_size=*/16)?; // (n_queries, n_pages) f32
// ...or cache the stacked passage tensor on-GPU and re-score cheaply:
let stacked = ColQwen3Emb::stack_passages(&pages)?;             // (n_pages, dims, max_sp)
let scores  = ColQwen3Emb::score_stacked(&q, &stacked, 16)?;
```

Public API — all in `crane-core/src/models/colqwen3_emb/model.rs`:

| Function | Purpose |
|----------|---------|
| `from_local(path, cpu, bf16)` | load config + tokenizer + safetensors, build vision / decoder / projection / M-RoPE |
| `set_dims(dims)` | Matryoshka dimension truncation |
| `encode_images(&[paths])` / `encode_images_from_bytes(&[&[u8]])` | image → multi-vector embedding |
| `encode_queries(&[&str])` | text → multi-vector embedding |
| `stack_passages(&[Tensor])` | pad + stack passages into a cacheable GPU tensor |
| `score(qs, ps, batch)` | MaxSim, one-shot (= `stack_passages` + `score_stacked`) |
| `score_stacked(qs, ps_t, batch)` | MaxSim against a pre-stacked / cached tensor |

`candle` is re-exported as `crane_core::models::candle_core`.

## Optimizations

The encode stage is compute-bound on a single 4090 (dominated by the vision forward pass), so the work targets **latency, eliminating host round-trips, and removing stalls** — not batching. Every change is validated bit-identical against the Python oracle.

**Attention & kernels**
- **Flash Attention v2** (`flash-attn`, SM_80+) — non-causal for the vision encoder, causal for the decoder prefill; masks are handled internally, none are materialized.
- **Flash-native `(B,S,H,D)` decoder** — Q/K/V stay in `(batch, seq, heads, dim)` end-to-end; RoPE is applied with `candle_nn::rotary_emb::rope_thd` in that same layout, so there are no transpose round-trips on the hot path.
- **Fused CUDA kernels** (BF16, compiled to PTX at build time): residual **add + RMSNorm** in a single pass, **SiLU(gate)·up** SwiGLU, plus **fused QKV** and **fused gate+up** projections (one concatenated matmul each).
- **GQA + per-head QK-norm** — grouped-query attention handled natively by flash-attn (no head expansion).

**Vision-encode pipeline**
- **Parallel CPU preprocessing** — `smart_resize` + CatmullRom resample run on the rayon pool while the GPU stays fed.
- **Pipelined / streamed encode** — decoded images flow through a bounded `sync_channel(8)`; the GPU consumes each image as it becomes ready instead of one blocking upload, and results are re-ordered back to input order.
- **Per-image vision forwards** — patches from different images never mix, keeping output bit-stable.
- **On-device patchify + normalize** — raw u8 is uploaded once, then `(x/255 − mean)/std` and the patch reshape happen on the GPU using pre-uploaded mean/std tensors.

**No host round-trips**
- **M-RoPE positions on the host** — `(t, h, w)` positions are computed on the CPU and `inv_freq` is kept host-side, so there is no device→host download per forward.
- **Vision mask on the host** — deepstack vision features are scattered into the text stream using host positions directly; the mask tensor is never uploaded.
- **GPU-resident passages** — `stack_passages` builds the padded tensor once; callers cache it and re-score via `score_stacked` with no re-upload. MaxSim chunks passages by `batch_size` to bound VRAM.

**Precision**
- **BF16** end-to-end on CUDA (F32 on CPU); final scores are cast to F32.
- **Matryoshka dims** — `set_dims` truncates the projection (default 2560; e.g. 1280) to trade storage for a small accuracy delta.

## Profiling

Set `CRANE_PROFILE=1` to get per-image, per-stage timing — **patchify / vision forward / prep (tokenize+merge+rope) / decoder+proj / decode-wait** — with percentages. It inserts `device.synchronize()` between stages while on, and is a complete no-op when off (`CRANE_PROFILE` is the only runtime env var the library reads).

## License

Fork of [lucasjinreal/Crane](https://github.com/lucasjinreal/Crane). See [LICENSE](LICENSE).
