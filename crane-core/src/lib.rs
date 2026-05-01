//! # crane-core (embeddings)
//!
//! Inference core for ColQwen3 multi-vector embeddings on top of Candle.
//!
//! Modules:
//! - [`fused_ops`] — fused CUDA kernels (silu-mul, add+rmsnorm, attention)
//! - [`models`]    — Qwen3-VL backbone and ColQwen3 embedder
//! - [`utils`]     — image preprocessing helpers (smart_resize)
//!
//! Feature flags: `cuda`, `flash-attn`, `cudnn`, `mkl`, `accelerate`.

pub mod fused_ops;
pub mod models;
pub mod utils;
