# Crane — ColQwen multi-vector embeddings (Rust)

A trimmed fork of [Crane](https://github.com/lucasjinreal/Crane) reduced to a single job:
**fast ColQwen document-retrieval embeddings in Rust**. It is the in-process inference
backend for [nebuia-embs](https://github.com/xellDart/nebuia-embs) — a `crane-core`
**library crate**, not a server.

The general-purpose inference engine (chat, TTS, OCR, multi-model serving, OpenAI HTTP
API, `crane-ctl` daemon) has been **removed**. What remains is the forward-only retrieval
path: a Qwen3-VL / Qwen3.5-VL vision+text backbone that turns document page images and
text queries into late-interaction (ColBERT/ColPali-style) multi-vector embeddings, plus
a MaxSim scorer.

## Supported models

All three are ColQwen-family multi-vector retrievers; `ColEmbedder::from_local` picks the
right one from `config.json`'s `model_type`. They share the MaxSim scoring path.

| Model | `model_type` | Backbone | Head | Notes |
|-------|-------------|----------|------|-------|
| **ops** — ColQwen3-4B | `ops_colqwen3` | Qwen3-VL (all full-attention) | `custom_text_proj` → L2 | the original path; 98% GEMM-bound; fastest |
| **Vultron** — ColQwen3.5-8B | `qwen3_5` | Qwen3.5-VL **hybrid** (24 GatedDeltaNet + 8 full-attn) | `custom_text_proj` → 320-d → L2 | ColPali-style; #1 in-domain on our legal docs |
| **Argus** — ColQwen3.5-9B | `argus_colqwen35` | Qwen3.5-VL **hybrid** (shared with Vultron) | MoE retrieval head | same backbone + Mixture-of-Experts head |

The two Qwen3.5-VL models add a **Gated Delta Net** (linear-attention) hybrid backbone;
ops has none (pure full-attention), so the GDN-specific optimizations below apply only to
Vultron/Argus. Each model lives in its own isolated module — `colqwen3_emb/`,
`colqwen3_5/`, `argus_colqwen35/` — sharing the `qwen3_vl` / `qwen3_5_vl` backbones.

## Build

Cargo workspace of two crates: `crane-core` (library) + `example` (binaries). Default is
CPU / F32.

```bash
git clone -b feat/colqwen3-embeddings https://github.com/xellDart/Crane.git && cd Crane

# Auto-detect CUDA + GPU compute capability, enabling flash-attn on Ampere+ (SM_80+)
bash build.sh            # flags: --cpu, --no-flash, --no-clean

# Or manual
cargo build --release --features flash-attn   # CUDA + Flash Attention (SM_80+)  ← prod build
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

Use the unified `ColEmbedder` — it dispatches to the right model by `config.json`:

```rust
use crane_core::models::col_embedder::ColEmbedder;

// Loads ops / Vultron / Argus depending on the checkpoint's model_type.
let mut model = ColEmbedder::from_local("./vultron-colqwen35-8b", /*cpu=*/false, /*bf16=*/true)?;

let pages = model.encode_images(&["p1.jpg", "p2.jpg"])?;   // (num_tokens, dims) per page
let q     = model.encode_queries(&["tabla de accionistas"])?;
let scores = ColEmbedder::score(&q, &pages, /*batch_size=*/128)?; // (n_queries, n_pages) f32
```

The ops model also has a direct API (`ColQwen3Emb` in `colqwen3_emb/model.rs`) with
`set_dims(dims)` for Matryoshka truncation, `stack_passages` / `score_stacked` for a
cached on-GPU passage tensor, and `encode_images_from_bytes` for disk-free encoding.
`candle` is re-exported as `crane_core::models::candle_core`.

**Example binaries** (`example/`): `embedding_simple`, `embedding_oracle_test`,
`argus_oracle_test`, `argus_eval`, `col_eval` (generic eval over any model → results
JSON), `gdn_parity` (recurrent-vs-chunked scan parity), `fp8_test`.

## Optimizations

Encode is compute-bound on a single 4090, so the work targets **latency, host round-trips,
and stalls** — not batching. Every change is validated against the Python oracle (bit-exact
where possible) and, for the lossy ones (bf16 scan, FP8), against **retrieval ranking on
real documents**, not just cosine.

### Full-attention path (all models)
- **Flash Attention v2** (`flash-attn`, SM_80+) — non-causal vision, causal decoder prefill; masks handled internally.
- **Flash-native `(B,S,H,D)` decoder** — Q/K/V stay in `(batch, seq, heads, dim)`; partial RoPE via `rope_thd` in-layout, no transpose round-trips.
- **Fused CUDA kernels** (BF16→PTX at build): residual **add+RMSNorm**, **SiLU(gate)·up** SwiGLU, fused QKV + fused gate+up projections.
- **Vision pipeline** — parallel CPU `smart_resize`/CatmullRom on rayon; pipelined encode through a bounded `sync_channel(8)`; on-device patchify + normalize; per-image forwards keep output bit-stable.
- **No host round-trips** — M-RoPE positions + `inv_freq` and the vision scatter mask stay host-side; GPU-resident cached passages (`stack_passages` / `score_stacked`).

### Hybrid Gated-Delta-Net path (Vultron / Argus only)
The 24 linear-attention layers were the bottleneck; a naive recurrent scan ran ~3s/page.
- **Chunked delta-rule scan** — port of `torch_chunk_gated_delta_rule` (chunk 64): batched matmuls over chunks instead of a per-token loop. **~2.85× (≈3s → ≈1s/page)**, parity cos 0.9995 vs recurrent.
- **3 custom CUDA kernels** (`kernels/fused_ops.cu`): `causal_conv1d_silu` (depthwise causal conv+SiLU, **444→8ms**), `chunk_delta_invert` (shared-mem forward substitution, **201→2ms**), `fused_rmsnorm_gated` (**31→2ms**).
- **scan_prep glue fusion** — the decay-mask/attn0 elementwise chain (7 candle launches over `(G,C,C)`) collapsed into two grid-stride kernels (`gdn_decay_mask`, `gdn_neg_lower_mul`), bit-identical.
- **bf16 scan** (default on) — the cross-chunk GEMMs + state recurrence run in bf16 tensor-core instead of F32 (~**−36ms/page**); decay math + intra-chunk inverse stay F32. Retrieval-validated (top-1 identical 96%, all changes score ties). `CRANE_GDN_BF16=0` forces the exact F32 path.
- **Cached async-alloc pool** — the CUDA memory-pool release threshold is raised at load so freed per-layer temporaries stay cached instead of being returned to the OS (a synchronizing op); this removes stalls in alloc-churning paths (notably FP8).

### FP8 W8A8 (opt-in, all models with an MLP)
- **`CRANE_FP8=1`** runs the MLP `gate_up` / `down` projections in FP8 E4M3 via cuBLASLt (custom op through cudarc's raw API — candle has the dtype but no FP8 matmul). Weights quantized once at load (bf16 copies dropped); activations dynamically per-token quantized on-device.
- **~2× on the isolated MLP GEMMs**, ~9% real end-to-end encode speedup. Off by default: E4M3's 3-bit mantissa caps embedding parity at cos ~0.991 (not fixable with finer scaling), a small quality cost on a quality-first retriever. Tunables: `CRANE_FP8_WS` (workspace MB), `CRANE_FP8_FASTACC=0` (full-precision accumulation).

## Environment flags

Runtime knobs read by the library (all optional; defaults in **bold**):

| Flag | Effect |
|------|--------|
| `CRANE_PROFILE=1` | per-stage timing (`VULTRON_PROFILE` / `GDN_PROF` lines). Inserts `synchronize()` between stages — **distorts real throughput, use for relative section analysis only**. |
| `CRANE_GDN_BF16` | bf16 scan matmuls for the hybrid models (**on**; `=0` forces exact F32). |
| `CRANE_GDN_RECURRENT=1` | sequential recurrent scan instead of chunked (parity A/B). |
| `CRANE_GDN_FUSED_SCAN=1` | single-launch fused recurrence kernel (F32-only, reference; slower at this shape). |
| `CRANE_FP8=1` | FP8 W8A8 MLP (**off**). |
| `CRANE_FP8_WS=<MB>` | cuBLASLt FP8 workspace (**32**). |
| `CRANE_FP8_FASTACC=0` | disable FP8 fast accumulation (**on**). |
| `CRANE_VISUAL_TOKENS=<n>` | max visual tokens for the hybrid models (**1792**, the author's recommendation). |
| `CRANE_ARGUS_NO_MOE=1` | run Argus without its MoE head (debug/ablation). |

## Profiling

`CRANE_PROFILE=1` prints per-image, per-stage timing (patchify / vision forward / prep /
decoder+proj / decode-wait, plus `GDN_PROF` per-section GDN timings for the hybrid models).
It inserts `device.synchronize()` between stages while on — great for *relative* section
attribution, but the forced syncs inflate absolute numbers, so measure real throughput
from an actual encode run (e.g. `col_eval`), not from the profiler.

## License

Fork of [lucasjinreal/Crane](https://github.com/lucasjinreal/Crane). See [LICENSE](LICENSE).
