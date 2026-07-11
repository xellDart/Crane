//! Argus-Colqwen3.5-9B: region-aware, query-conditioned Mixture-of-Experts
//! visual document retriever built on the Qwen3.5-VL hybrid backbone.
//!
//! Reference: DataScience-UIBK/Argus-Colqwen3.5-9b-v0 (modeling_argus.py).
//! This module is fully isolated from `colqwen3_emb` (the ops model); the two
//! are selected at load time by `config.json` `model_type`.

pub mod model;

pub use model::{ArgusColqwen35Emb, ArgusConfig};
