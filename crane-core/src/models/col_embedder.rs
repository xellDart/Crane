//! Load-time dispatch between the two ColBERT-style embedders by the model's
//! `config.json` `model_type`:
//!   - `ops_colqwen3` (or anything unknown/absent) → `ColQwen3Emb` (unchanged)
//!   - `argus_colqwen35`                            → `ArgusColqwen35Emb`
//!
//! Both variants expose an identical public surface, so callers hold a single
//! `ColEmbedder` and never branch. The ops path delegates verbatim — no
//! behavioral change for existing deployments.

use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::Deserialize;
use std::path::Path;

use super::argus_colqwen35::ArgusColqwen35Emb;
use super::colqwen3_emb::ColQwen3Emb;

#[derive(Deserialize)]
struct ModelTypeProbe {
    #[serde(default)]
    model_type: String,
}

pub enum ColEmbedder {
    Ops(ColQwen3Emb),
    Argus(ArgusColqwen35Emb),
}

impl ColEmbedder {
    /// Read `config.json` `model_type` and construct the matching embedder.
    pub fn from_local(path: impl AsRef<Path>, cpu: bool, bf16: bool) -> Result<Self> {
        let base = path.as_ref();
        let probe: ModelTypeProbe =
            serde_json::from_str(&std::fs::read_to_string(base.join("config.json"))?)?;
        match probe.model_type.as_str() {
            "argus_colqwen35" => Ok(Self::Argus(ArgusColqwen35Emb::from_local(base, cpu, bf16)?)),
            // ops_colqwen3 and any legacy/unknown value keep the original path.
            _ => Ok(Self::Ops(ColQwen3Emb::from_local(base, cpu, bf16)?)),
        }
    }

    pub fn model_kind(&self) -> &'static str {
        match self {
            Self::Ops(_) => "ops_colqwen3",
            Self::Argus(_) => "argus_colqwen35",
        }
    }

    pub fn device(&self) -> &Device {
        match self {
            Self::Ops(m) => &m.device,
            Self::Argus(m) => &m.device,
        }
    }

    pub fn set_dims(&mut self, dims: usize) {
        match self {
            Self::Ops(m) => m.set_dims(dims),
            Self::Argus(m) => m.set_dims(dims),
        }
    }

    pub fn encode_images<P: AsRef<Path> + Sync>(&mut self, image_paths: &[P]) -> Result<Vec<Tensor>> {
        match self {
            Self::Ops(m) => m.encode_images(image_paths),
            Self::Argus(m) => m.encode_images(image_paths),
        }
    }

    pub fn encode_images_from_bytes(&mut self, images: &[&[u8]]) -> Result<Vec<Tensor>> {
        match self {
            Self::Ops(m) => m.encode_images_from_bytes(images),
            Self::Argus(m) => m.encode_images_from_bytes(images),
        }
    }

    pub fn encode_queries(&mut self, queries: &[&str]) -> Result<Vec<Tensor>> {
        match self {
            Self::Ops(m) => m.encode_queries(queries),
            Self::Argus(m) => m.encode_queries(queries),
        }
    }

    // Scoring is stateless and identical across both models (plain MaxSim over
    // L2-normalized multi-vectors); delegate to one implementation.
    pub fn stack_passages(ps: &[Tensor]) -> Result<Tensor> {
        ColQwen3Emb::stack_passages(ps)
    }

    pub fn score(qs: &[Tensor], ps: &[Tensor], batch_size: usize) -> Result<Tensor> {
        ColQwen3Emb::score(qs, ps, batch_size)
    }

    pub fn score_stacked(qs: &[Tensor], ps_t: &Tensor, batch_size: usize) -> Result<Tensor> {
        ColQwen3Emb::score_stacked(qs, ps_t, batch_size)
    }
}
