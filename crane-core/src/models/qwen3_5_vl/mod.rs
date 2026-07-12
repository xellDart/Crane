//! Qwen3.5-VL hybrid backbone (Argus-Colqwen3.5-9B).
//!
//! Vision tower (Qwen3.5 ViT + learned interpolated pos-embed, no deepstack) plus
//! a 32-layer hybrid text decoder that interleaves Gated-DeltaNet linear-attention
//! layers with gated full-attention layers (partial RoPE, QK-norm, GQA). Forward-only
//! prefill: no KV cache, no autoregressive decode, no recurrent-state save/restore.
//!
//! Public API mirrors the (trimmed) `qwen3_vl` backbone so a retrieval head can drive
//! it the same way. See `model.rs` for the port and the parity notes.

mod model;

pub use model::{
    gdn_prof_report, MRoPE, Qwen35VLConfig, Qwen35VLTextConfig, Qwen35VLVisionConfig,
    RopeParameters, TextConfig, TextDecoder, VisionConfig, VisionModel,
};
