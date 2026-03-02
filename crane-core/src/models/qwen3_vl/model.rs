//! Qwen3-VL: Vision-Language model combining a custom ViT encoder with a Qwen3 text decoder.
//!
//! Architecture:
//!   - Vision: Conv3D patch embed + 24-layer ViT + spatial merge + DeepStack
//!   - Text: 28-layer Qwen3 decoder with M-RoPE (3D rotary pos), QK-norm, GQA
//!   - Uses the existing Qwen3VLProcessor from processor.rs for image preprocessing

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Module, Shape, Tensor, D};
use candle_nn::{self, linear, linear_no_bias, Activation, Embedding, LayerNorm, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

use super::config::PreprocessorConfig;

fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<RmsNorm> {
    let w = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?;
    Ok(RmsNorm::new(w, eps))
}

/// Qwen3.5 uses `(1 + weight) * x_normalized` (weights trained from zero-init).
/// This helper loads the weight and pre-adds 1 so standard RmsNorm gives the right result.
fn rms_norm_qwen35(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<RmsNorm> {
    let w = vb.get_with_hints(size, "weight", candle_nn::Init::Const(0.))?;
    let w = w.affine(1.0, 1.0)?; // stored_weight + 1
    Ok(RmsNorm::new(w, eps))
}

/// Load a raw layernorm weight tensor; for Qwen3.5 add 1 to match the (1+w) parameterization.
fn ln_weight_for_cfg(size: usize, cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Tensor> {
    let w = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?;
    if cfg.is_hybrid() { w.affine(1.0, 1.0) } else { Ok(w) }
}

// ── Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3VLConfig {
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub hidden_act: String,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    // Direct field (qwen3-vl) or nested in rope_parameters (qwen3.5)
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    // Hybrid architecture fields (qwen3.5 — absent in qwen3-vl)
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "default_linear_num_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_num_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_linear_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,
    // Gated output for full-attention layers (qwen3.5)
    #[serde(default)]
    pub attn_output_gate: bool,
}

impl TextConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().map(|p| p.rope_theta))
            .unwrap_or(1_000_000.0)
    }

    /// Number of head dimensions that receive rotary position embeddings.
    /// For Qwen3.5: head_dim * partial_rotary_factor (e.g. 256 * 0.25 = 64).
    /// For Qwen3-VL: head_dim (full rotation).
    pub fn rope_dim(&self) -> usize {
        let factor = self.rope_parameters.as_ref()
            .map(|p| p.partial_rotary_factor)
            .unwrap_or(1.0);
        ((self.head_dim as f64 * factor).round() as usize).max(2)
    }

    /// mrope_section — from rope_parameters (Qwen3.5) or rope_scaling (Qwen3-VL).
    pub fn rope_mrope_section(&self) -> Vec<usize> {
        if let Some(ref p) = self.rope_parameters {
            if !p.mrope_section.is_empty() {
                return p.mrope_section.clone();
            }
        }
        if let Some(ref s) = self.rope_scaling {
            if !s.mrope_section.is_empty() {
                return s.mrope_section.clone();
            }
        }
        vec![24, 20, 20] // default for Qwen3-VL-2B
    }

    /// Whether this config uses the hybrid GDN/full-attention architecture.
    pub fn is_hybrid(&self) -> bool {
        !self.layer_types.is_empty()
    }

    pub fn layer_type(&self, i: usize) -> &str {
        if self.layer_types.is_empty() {
            "full_attention"
        } else {
            &self.layer_types[i]
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

fn default_partial_rotary_factor() -> f64 { 1.0 }

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub mrope_interleaved: bool,
}

fn default_head_dim() -> usize { 128 }
fn default_linear_num_heads() -> usize { 16 }
fn default_linear_head_dim() -> usize { 128 }
fn default_conv_kernel_dim() -> usize { 4 }

// ── Vision: Patch Embedding ──────────────────────────────────────────

struct VisionPatchEmbed {
    proj: Linear,
}

impl VisionPatchEmbed {
    fn new(vb: VarBuilder, cfg: &VisionConfig) -> candle_core::Result<Self> {
        // Conv3D weight [out, in_c, t, h, w] → reshape to linear [out, in_c*t*h*w]
        let w = vb.pp("proj").get_with_hints(
            (cfg.hidden_size, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size),
            "weight",
            candle_nn::Init::Const(0.),
        )?;
        let b = vb.pp("proj").get_with_hints(cfg.hidden_size, "bias", candle_nn::Init::Const(0.))?;
        let in_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let w_2d = w.reshape((cfg.hidden_size, in_dim))?;
        let proj = Linear::new(w_2d, Some(b));
        Ok(Self { proj })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: (num_patches, in_dim) → (num_patches, hidden_size)
        self.proj.forward(x)
    }
}

// ── Vision: Rotary Embedding (2D spatial) ────────────────────────────
//
// Matches HF's Qwen3VLVisionRotaryEmbedding(head_dim // 2):
//   - dim = head_dim // 2 = 32 → inv_freq has 16 elements
//   - Computes freq_table: (max_hw, 16) from outer(arange, inv_freq)
//   - 2D positions: [h_freq(16), w_freq(16)] → 32-dim → doubled to 64-dim

struct VisionRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl VisionRotaryEmbedding {
    fn new(half_head_dim: usize, _device: &Device) -> candle_core::Result<Self> {
        // HF: dim = head_dim // 2; inv_freq = 1.0 / (10000 ** (arange(0, dim, 2) / dim))
        let theta = 10000f64;
        let inv_freq: Vec<f32> = (0..half_head_dim / 2)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / half_head_dim as f64) as f32)
            .collect();
        Ok(Self { inv_freq })
    }

    fn forward(&self, grid_thw: &Tensor, merge_size: usize, device: &Device) -> candle_core::Result<(Tensor, Tensor)> {
        // Matches HF's rot_pos_emb: computes 2D position embeddings in merge-block order
        let grid_thw_vec = grid_thw.to_vec2::<u32>()?;
        let n_freq = self.inv_freq.len(); // 16

        // Compute freq_table: outer(arange(max_hw), inv_freq) → (max_hw, n_freq)
        let max_hw = grid_thw_vec.iter().flat_map(|g| [g[1], g[2]]).max().unwrap_or(1) as usize;
        let mut freq_table = vec![vec![0f32; n_freq]; max_hw];
        for pos in 0..max_hw {
            for j in 0..n_freq {
                freq_table[pos][j] = pos as f32 * self.inv_freq[j];
            }
        }

        // Build position embeddings in merge-block order
        let mut all_freqs: Vec<f32> = Vec::new();

        for grid in &grid_thw_vec {
            let t = grid[0] as usize;
            let h = grid[1] as usize;
            let w = grid[2] as usize;
            let merged_h = h / merge_size;
            let merged_w = w / merge_size;

            // Iterate in merge-block order: (block_row, block_col, intra_row, intra_col)
            for _frame in 0..t {
                for br in 0..merged_h {
                    for bc in 0..merged_w {
                        for ir in 0..merge_size {
                            for ic in 0..merge_size {
                                let row = br * merge_size + ir;
                                let col = bc * merge_size + ic;
                                // Each position gets [h_freqs(16), w_freqs(16)] = 32 values
                                all_freqs.extend_from_slice(&freq_table[row]);
                                all_freqs.extend_from_slice(&freq_table[col]);
                            }
                        }
                    }
                }
            }
        }

        let total_patches = all_freqs.len() / (2 * n_freq);
        let half_rope_dim = 2 * n_freq; // 32
        let freqs = Tensor::from_vec(all_freqs, (total_patches, half_rope_dim), device)?;

        // cos/sin of half-dim freqs (apply_vision_rope uses split-half formula directly)
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok((cos, sin))
    }
}

// ── Vision: Attention ────────────────────────────────────────────────

struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(vb: VarBuilder, cfg: &VisionConfig) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let qkv = linear(h, 3 * h, vb.pp("qkv"))?;
        let proj = linear(h, h, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_heads,
            head_dim: h / cfg.num_heads,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (seq_len, _) = x.dims2()?;
        let qkv = self.qkv.forward(x)?; // (seq, 3*hidden)
        let qkv = qkv.reshape((seq_len, 3, self.num_heads, self.head_dim))?;
        let q = qkv.i((.., 0, .., ..))?.contiguous()?; // (seq, heads, dim)
        let k = qkv.i((.., 1, .., ..))?.contiguous()?;
        let v = qkv.i((.., 2, .., ..))?.contiguous()?;

        // Apply rotary embeddings
        let q = self.apply_vision_rope(&q, cos, sin)?;
        let k = self.apply_vision_rope(&k, cos, sin)?;

        // Scaled dot-product attention (Flash Attention dispatch for 3D vision tensors)
        // q, k, v are (seq, heads, dim) — flash_attn_3d handles the layout
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let out = crate::fused_ops::attention::scaled_dot_product_attention_3d(&q, &k, &v)?;

        // Reshape: (seq, heads, dim) -> (seq, hidden)
        let out = out.contiguous()?;
        let out = out.reshape((seq_len, self.num_heads * self.head_dim))?;
        self.proj.forward(&out)
    }

    fn apply_vision_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        // x: (seq, heads, dim), cos/sin: (seq, half_dim) — already half-dim, no doubling needed.
        // Split-half formula (avoids cat for rotated tensor):
        //   out_lo = x_lo * cos - x_hi * sin
        //   out_hi = x_lo * sin + x_hi * cos
        let (_seq, _heads, dim) = x.dims3()?;
        let half = dim / 2;

        let cos = cos.unsqueeze(1)?; // (seq, 1, half)
        let sin = sin.unsqueeze(1)?;

        let x1 = x.narrow(2, 0, half)?;
        let x2 = x.narrow(2, half, half)?;

        let out_lo = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out_hi = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[&out_lo, &out_hi], 2)
    }
}

// ── Vision: MLP ──────────────────────────────────────────────────────

struct VisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMLP {
    fn new(vb: VarBuilder, cfg: &VisionConfig) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("linear_fc1"))?,
            fc2: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("linear_fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?; // gelu_pytorch_tanh
        self.fc2.forward(&x)
    }
}

// ── Vision: Block ────────────────────────────────────────────────────

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMLP,
}

impl VisionBlock {
    fn new(vb: VarBuilder, cfg: &VisionConfig) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            norm1: candle_nn::layer_norm(h, 1e-6, vb.pp("norm1"))?,
            norm2: candle_nn::layer_norm(h, 1e-6, vb.pp("norm2"))?,
            attn: VisionAttention::new(vb.pp("attn"), cfg)?,
            mlp: VisionMLP::new(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self.attn.forward(&x, cos, sin)?;
        let x = (x + residual)?;
        let residual = &x;
        let h = self.norm2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }
}

// ── Vision: Patch Merger ─────────────────────────────────────────────

struct VisionPatchMerger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    merged_dim: usize,
    post_shuffle_norm: bool,
}

impl VisionPatchMerger {
    fn new(
        vb: VarBuilder,
        cfg: &VisionConfig,
        post_shuffle_norm: bool,
    ) -> candle_core::Result<Self> {
        let merged_dim = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let norm_dim = if post_shuffle_norm { merged_dim } else { cfg.hidden_size };
        Ok(Self {
            norm: candle_nn::layer_norm(norm_dim, 1e-6, vb.pp("norm"))?,
            fc1: linear(merged_dim, merged_dim, vb.pp("linear_fc1"))?,
            fc2: linear(merged_dim, cfg.out_hidden_size, vb.pp("linear_fc2"))?,
            merged_dim,
            post_shuffle_norm,
        })
    }

    fn forward(&self, x: &Tensor, _grid_thw: &Tensor) -> candle_core::Result<Tensor> {
        // Patches are already in merge-block order (br, bc, ir, ic) from preprocessing.
        // Just group every merge^2 consecutive patches into one vector.
        // Matches HF: x.view(-1, self.hidden_size)
        let merged_dim = self.merged_dim;

        let merged = if !self.post_shuffle_norm {
            // Main merger: norm on individual patches, then reshape to merged
            let normed = self.norm.forward(x)?;
            normed.reshape(((), merged_dim))?
        } else {
            // DeepStack merger: reshape to merged, then norm
            let reshaped = x.reshape(((), merged_dim))?;
            self.norm.forward(&reshaped)?
        };

        let x = self.fc1.forward(&merged)?.gelu()?;
        self.fc2.forward(&x)
    }
}

// ── Vision Model ─────────────────────────────────────────────────────

pub struct VisionModel {
    patch_embed: VisionPatchEmbed,
    pos_embed: Embedding,
    num_grid_per_side: usize,
    spatial_merge_size: usize,
    rotary_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    merger: VisionPatchMerger,
    deepstack_mergers: Vec<VisionPatchMerger>,
    deepstack_indexes: Vec<usize>,
}

impl VisionModel {
    /// Create a new VisionModel. `vb` must already be prefixed (e.g. `vb.pp("model.visual")` or `vb.pp("visual")`).
    pub fn new(vb: VarBuilder, cfg: &VisionConfig) -> candle_core::Result<Self> {
        let patch_embed = VisionPatchEmbed::new(vb.pp("patch_embed"), cfg)?;
        let pos_embed = candle_nn::embedding(
            cfg.num_position_embeddings, cfg.hidden_size, vb.pp("pos_embed"),
        )?;
        let head_dim = cfg.hidden_size / cfg.num_heads; // 64
        // HF: VisionRotaryEmbedding(head_dim // 2) → dim=32, inv_freq has 16 elements
        let rotary_emb = VisionRotaryEmbedding::new(head_dim / 2, vb.device())?;

        // num_position_embeddings = grid_side^2 (e.g., 2304 = 48*48)
        let num_grid_per_side = (cfg.num_position_embeddings as f64).sqrt() as usize;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(VisionBlock::new(vb.pp(&format!("blocks.{}", i)), cfg)?);
        }

        let merger = VisionPatchMerger::new(vb.pp("merger"), cfg, false)?;

        let mut deepstack_mergers = Vec::new();
        for i in 0..cfg.deepstack_visual_indexes.len() {
            deepstack_mergers.push(VisionPatchMerger::new(
                vb.pp(&format!("deepstack_merger_list.{}", i)),
                cfg,
                true,
            )?);
        }

        Ok(Self {
            patch_embed,
            pos_embed,
            num_grid_per_side,
            spatial_merge_size: cfg.spatial_merge_size,
            rotary_emb,
            blocks,
            merger,
            deepstack_mergers,
            deepstack_indexes: cfg.deepstack_visual_indexes.clone(),
        })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
    ) -> candle_core::Result<(Tensor, Vec<Tensor>)> {
        let device = pixel_values.device();

        // pixel_values: (total_patches, in_dim) from processor
        let mut hidden = self.patch_embed.forward(pixel_values)?;

        // Add positional embeddings with bilinear interpolation
        let pos_embeds = self.fast_pos_embed_interpolate(grid_thw)?;
        hidden = (hidden + pos_embeds)?;

        // Compute 2D rotary embeddings for vision attention
        let (cos, sin) = self.rotary_emb.forward(grid_thw, self.spatial_merge_size, device)?;
        // Cast vision RoPE cos/sin to match hidden state dtype (needed for BF16)
        let cos = cos.to_dtype(hidden.dtype())?;
        let sin = sin.to_dtype(hidden.dtype())?;

        // Forward through blocks, extracting DeepStack features
        let mut deepstack_features = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, &cos, &sin)?;

            if let Some(ds_idx) = self.deepstack_indexes.iter().position(|&x| x == i) {
                let feat = self.deepstack_mergers[ds_idx].forward(&hidden, grid_thw)?;
                deepstack_features.push(feat);
            }
        }

        // Main merger
        let merged = self.merger.forward(&hidden, grid_thw)?;
        Ok((merged, deepstack_features))
    }

    /// Bilinear interpolation of position embeddings for variable resolution.
    /// Matches HF's Qwen3VLVisionModel.fast_pos_embed_interpolate
    fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> candle_core::Result<Tensor> {
        let grid_thw_vec = grid_thw.to_vec2::<u32>()?;
        let device = grid_thw.device();
        let n = self.num_grid_per_side; // 48
        let merge = self.spatial_merge_size;

        // pos_embed table: (n*n, hidden)
        let pos_table = self.pos_embed.embeddings();

        let mut all_pos_embeds = Vec::new();

        for grid in &grid_thw_vec {
            let t = grid[0] as usize;
            let h = grid[1] as usize;
            let w = grid[2] as usize;

            // Compute interpolation indices and weights
            // h_idxs = linspace(0, n-1, h), w_idxs = linspace(0, n-1, w)
            let h_idxs: Vec<f32> = (0..h).map(|i| i as f32 * (n - 1) as f32 / (h.max(1) - 1).max(1) as f32).collect();
            let w_idxs: Vec<f32> = (0..w).map(|i| i as f32 * (n - 1) as f32 / (w.max(1) - 1).max(1) as f32).collect();

            // For each (row, col) position, compute 4-point bilinear interpolation
            let mut idx_00 = Vec::new();
            let mut idx_01 = Vec::new();
            let mut idx_10 = Vec::new();
            let mut idx_11 = Vec::new();
            let mut w_00 = Vec::new();
            let mut w_01 = Vec::new();
            let mut w_10 = Vec::new();
            let mut w_11 = Vec::new();

            for &hi in &h_idxs {
                let h_floor = hi as usize;
                let h_ceil = (h_floor + 1).min(n - 1);
                let dh = hi - h_floor as f32;

                for &wi in &w_idxs {
                    let w_floor = wi as usize;
                    let w_ceil = (w_floor + 1).min(n - 1);
                    let dw = wi - w_floor as f32;

                    idx_00.push((h_floor * n + w_floor) as u32);
                    idx_01.push((h_floor * n + w_ceil) as u32);
                    idx_10.push((h_ceil * n + w_floor) as u32);
                    idx_11.push((h_ceil * n + w_ceil) as u32);

                    w_00.push((1.0 - dh) * (1.0 - dw));
                    w_01.push((1.0 - dh) * dw);
                    w_10.push(dh * (1.0 - dw));
                    w_11.push(dh * dw);
                }
            }

            // Lookup + weighted sum
            let e00 = pos_table.index_select(&Tensor::new(idx_00.as_slice(), device)?, 0)?;
            let e01 = pos_table.index_select(&Tensor::new(idx_01.as_slice(), device)?, 0)?;
            let e10 = pos_table.index_select(&Tensor::new(idx_10.as_slice(), device)?, 0)?;
            let e11 = pos_table.index_select(&Tensor::new(idx_11.as_slice(), device)?, 0)?;

            // Cast interpolation weights to match pos_table dtype (needed for BF16)
            let pos_dtype = pos_table.dtype();
            let w00_t = Tensor::new(w_00.as_slice(), device)?.unsqueeze(1)?.to_dtype(pos_dtype)?;
            let w01_t = Tensor::new(w_01.as_slice(), device)?.unsqueeze(1)?.to_dtype(pos_dtype)?;
            let w10_t = Tensor::new(w_10.as_slice(), device)?.unsqueeze(1)?.to_dtype(pos_dtype)?;
            let w11_t = Tensor::new(w_11.as_slice(), device)?.unsqueeze(1)?.to_dtype(pos_dtype)?;

            let pos_embed = (e00.broadcast_mul(&w00_t)?
                + e01.broadcast_mul(&w01_t)?
                + e10.broadcast_mul(&w10_t)?
                + e11.broadcast_mul(&w11_t)?)?; // (h*w, hidden)

            // Repeat for temporal frames and permute to merge-block order
            for _frame in 0..t {
                // Permute to merge-block order: (h, w) → (h/m, w/m, m, m)
                let merged_h = h / merge;
                let merged_w = w / merge;
                let hidden_size = pos_embed.dim(1)?;
                let pe = pos_embed
                    .reshape((merged_h, merge, merged_w, merge, hidden_size))?
                    .permute((0, 2, 1, 3, 4))? // (merged_h, merged_w, merge, merge, hidden)
                    .reshape((merged_h * merged_w * merge * merge, hidden_size))?
                    .contiguous()?;
                all_pos_embeds.push(pe);
            }
        }

        Tensor::cat(&all_pos_embeds, 0)
    }
}

// ── M-RoPE (Multimodal Rotary Position Embeddings) ───────────────────

pub struct MRoPE {
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
    dtype: DType,
}

impl MRoPE {
    pub fn new(cfg: &TextConfig, device: &Device, dtype: DType) -> candle_core::Result<Self> {
        let rope_dim = cfg.rope_dim(); // partial_rotary_factor applied (64 for qwen3.5)
        let theta    = cfg.rope_theta();
        let half_dim = rope_dim / 2;  // 32 for qwen3.5, 64 for qwen3-vl
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?;

        let mrope_section = cfg.rope_mrope_section(); // [11,11,10] for qwen3.5

        Ok(Self { inv_freq, mrope_section, dtype })
    }

    pub fn forward(&self, position_ids: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        // position_ids: (3, seq_len) — [temporal, height, width]
        let device = self.inv_freq.device();
        let half_dim = self.inv_freq.dims()[0]; // 64
        let seq_len = position_ids.dim(1)?;
        let sections = &self.mrope_section; // [24, 20, 20] for Qwen3-VL-2B
        let inv_freq_vec = self.inv_freq.to_vec1::<f32>()?;

        // Get position values for each dimension
        let t_pos = position_ids.i(0)?.to_vec1::<i64>()?; // temporal
        let h_pos = position_ids.i(1)?.to_vec1::<i64>()?; // height
        let w_pos = position_ids.i(2)?.to_vec1::<i64>()?; // width

        let min_section = *sections.iter().min().unwrap(); // 10 for [11,11,10]; 20 for [24,20,20]

        // Build interleaved frequency table matching HF apply_multimodal_rotary_pos_emb.
        // Layout: for each of the min_section rounds, one T freq, one H freq, one W freq
        // (interleaved [T,H,W,T,H,W,...]).  After that, any excess per section in order T, H, W.
        let mut output = vec![vec![0f32; half_dim]; seq_len];
        let interleaved_end = min_section * 3;

        // positions per dimension [T, H, W]
        let dim_names = [&t_pos, &h_pos, &w_pos];

        for s in 0..seq_len {
            let tp = t_pos[s] as f32;
            let hp = h_pos[s] as f32;
            let wp = w_pos[s] as f32;
            let dim_pos = [tp, hp, wp];

            // Interleaved region: [T, H, W, T, H, W, ...]
            for i in 0..min_section {
                let base = i * 3;
                output[s][base]     = dim_pos[0] * inv_freq_vec[base];
                output[s][base + 1] = dim_pos[1] * inv_freq_vec[base + 1];
                output[s][base + 2] = dim_pos[2] * inv_freq_vec[base + 2];
            }

            // Remaining (extra) frequencies per dimension in order T, H, W
            let mut f = interleaved_end;
            for (d, &sec_len) in sections.iter().enumerate() {
                for _ in 0..(sec_len - min_section) {
                    output[s][f] = dim_pos[d] * inv_freq_vec[f];
                    f += 1;
                }
            }
            // f should now equal half_dim
        }
        let _ = dim_names; // suppress unused-variable warning

        let output = Tensor::new(output, device)?; // (seq_len, half_dim)
        let cos = output.cos()?.to_dtype(self.dtype)?;
        let sin = output.sin()?.to_dtype(self.dtype)?;
        Ok((cos, sin))
    }
}

// ── Text: Helpers for GDN ────────────────────────────────────────────

/// L2-normalize `x` along its last dimension (per-head unit norm).
fn l2_normalize(x: &Tensor) -> candle_core::Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.clamp(1e-12_f64, f64::MAX)?.sqrt()?;
    x.broadcast_div(&norm)
}

/// Element-wise softplus: log(1 + exp(x)).
fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    // softplus(x) = log(1 + exp(x))
    // Numerically stable: max(x,0) + log(1 + exp(-|x|))
    //   = max(x,0) + log(exp(-|x|) + 1)
    let pos   = x.clamp(0.0_f64, f64::MAX)?;
    let abs_x = x.abs()?;
    // log(exp(-|x|) + 1): since exp(-|x|) in (0,1], this is safe
    let inner     = abs_x.neg()?.exp()?.add(&Tensor::ones_like(&abs_x)?)?;
    let log_inner = inner.log()?;
    pos.add(&log_inner)
}

// ── Text: Gated Delta Network (linear attention) ─────────────────────

/// RMSNorm with a SiLU gate applied to the output: `rms_norm(x) * silu(gate)`.
struct RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor, gate: &Tensor) -> candle_core::Result<Tensor> {
        // x, gate: (..., size)
        let x = x.to_dtype(DType::F32)?;
        let gate = gate.to_dtype(DType::F32)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let x_scaled = x_norm.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        // SiLU gate: x * sigmoid(x)
        let g = candle_nn::Activation::Silu.forward(&gate)?;
        x_scaled.mul(&g)?.to_dtype(self.weight.dtype())
    }
}

/// Gated Delta Network — the linear-attention layer in Qwen3.5.
///
/// Weights (per layer):
///   linear_attn.in_proj_qkv  (key_dim + key_dim + value_dim, hidden)
///   linear_attn.in_proj_z    (value_dim, hidden)
///   linear_attn.in_proj_b    (num_heads, hidden)
///   linear_attn.in_proj_a    (num_heads, hidden)
///   linear_attn.conv1d.weight (conv_dim, 1, kernel_size)
///   linear_attn.A_log        (num_heads,)
///   linear_attn.dt_bias      (num_heads,)
///   linear_attn.norm.weight  (head_v_dim,)
///   linear_attn.out_proj     (hidden, value_dim)
struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z:   Linear,
    in_proj_b:   Linear,
    in_proj_a:   Linear,
    conv1d_w:    Tensor,   // (conv_dim, kernel_size) — squeezed from (conv_dim, 1, ks)
    a_log:       Tensor,   // (num_heads,)
    dt_bias:     Tensor,   // (num_heads,)
    norm:        RmsNormGated,
    out_proj:    Linear,
    // dims
    num_heads:  usize,
    key_dim:    usize,
    value_dim:  usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_ks:    usize,
    conv_dim:   usize,
    // decode state
    conv_state:      Option<Tensor>, // (B, conv_dim, conv_ks - 1)
    recurrent_state: Option<Tensor>, // (B, num_heads, head_k_dim, head_v_dim)
}

impl GatedDeltaNet {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h    = cfg.hidden_size;
        let nkh  = cfg.linear_num_key_heads;
        let nvh  = cfg.linear_num_value_heads;
        let hkd  = cfg.linear_key_head_dim;
        let hvd  = cfg.linear_value_head_dim;
        let ks   = cfg.linear_conv_kernel_dim;
        let kd   = nkh * hkd;   // key_dim  = 2048
        let vd   = nvh * hvd;   // value_dim = 2048
        let conv_dim = kd + kd + vd;  // 6144

        let in_proj_qkv = linear_no_bias(h, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z   = linear_no_bias(h, vd,       vb.pp("in_proj_z"))?;
        // in_proj_b / in_proj_a output num_heads scalars (not per-head-dim vectors)
        let in_proj_b = linear_no_bias(h, nvh, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(h, nvh, vb.pp("in_proj_a"))?;

        let conv1d_raw = vb.get((conv_dim, 1, ks), "conv1d.weight")?;
        let conv1d_w   = conv1d_raw.reshape((conv_dim, ks))?;

        let a_log   = vb.get(nvh, "A_log")?;
        let dt_bias = vb.get(nvh, "dt_bias")?;

        let norm     = RmsNormGated::new(hvd, cfg.rms_norm_eps, vb.pp("norm"))?;
        let out_proj = linear_no_bias(vd, h, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj_qkv, in_proj_z, in_proj_b, in_proj_a,
            conv1d_w, a_log, dt_bias, norm, out_proj,
            num_heads: nvh, key_dim: kd, value_dim: vd,
            head_k_dim: hkd, head_v_dim: hvd,
            conv_ks: ks, conv_dim,
            conv_state: None, recurrent_state: None,
        })
    }

    /// Apply depthwise causal conv1d to `x: (B, T, C)` → `(B, T, C)` with SiLU.
    /// Uses `self.conv_state` as left-padding during prefill/decode.
    fn apply_conv1d(&self, x: &Tensor, is_decode: bool) -> candle_core::Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let ks = self.conv_ks;
        // Transpose to (B, C, T)
        let x_t = x.transpose(1, 2)?;

        // Pad left with zeros (prefill always) or with stored conv_state (decode only).
        // During prefill conv_state may already be set (we save input before calling this),
        // but we must NOT use it as padding — prefill always uses causal zero-padding.
        let pad = if is_decode {
            match &self.conv_state {
                Some(s) => s.clone(),
                None => Tensor::zeros((b, c, ks - 1), x_t.dtype(), x_t.device())?,
            }
        } else {
            Tensor::zeros((b, c, ks - 1), x_t.dtype(), x_t.device())?
        };

        let x_padded = Tensor::cat(&[&pad, &x_t], 2)?; // (B, C, T + ks - 1)

        // Vectorised depthwise conv: for each pos, element-wise mul with kernel, sum over ks.
        // Window shape for all T positions: stack as (B, C, T, ks), then reduce.
        let windows: candle_core::Result<Vec<Tensor>> = (0..t)
            .map(|i| x_padded.narrow(2, i, ks))
            .collect();
        let windows = windows?;
        let stacked = Tensor::stack(&windows, 2)?; // (B, C, T, ks)
        let w = self.conv1d_w.to_dtype(stacked.dtype())?;
        // w: (C, ks) → broadcast to (1, C, 1, ks)
        let w = w.unsqueeze(0)?.unsqueeze(2)?;
        let out = stacked.broadcast_mul(&w)?.sum(D::Minus1)?; // (B, C, T)

        // SiLU
        let out = candle_nn::Activation::Silu.forward(&out)?;
        out.transpose(1, 2) // (B, T, C)
    }

    /// Update the conv sliding-window state from the conv INPUT `x_input: (B, C, T)`.
    ///
    /// Prefill: stores the last `conv_ks - 1` input tokens (zero-padded if T < ks-1).
    /// Decode:  shifts the existing state left by 1 and appends the new token (x_input is (B,C,1)),
    ///          so the window always covers the last `ks-1` input tokens seen so far.
    fn update_conv_state(&mut self, x_input: &Tensor, t: usize, is_decode: bool) -> candle_core::Result<()> {
        let need = self.conv_ks - 1;
        let state = if is_decode {
            // Decode (t=1): shift existing state [old[1:], new_tok]
            match &self.conv_state {
                Some(s) => {
                    if need <= 1 {
                        x_input.clone()
                    } else {
                        // s: (B, C, need); drop oldest token, append new
                        let tail = s.narrow(2, 1, need - 1)?;
                        Tensor::cat(&[&tail, x_input], 2)?
                    }
                }
                None => {
                    // First decode without prior state (shouldn't happen in normal flow,
                    // but handle gracefully by zero-padding on the left)
                    if need <= 1 {
                        x_input.clone()
                    } else {
                        let (b, c, _) = x_input.dims3()?;
                        let pad = Tensor::zeros((b, c, need - 1), x_input.dtype(), x_input.device())?;
                        Tensor::cat(&[&pad, x_input], 2)?
                    }
                }
            }
        } else {
            // Prefill: store last `need` input tokens (zero-pad if T < need)
            if t >= need {
                x_input.narrow(2, t - need, need)?
            } else {
                let (b, c, _) = x_input.dims3()?;
                let pad = Tensor::zeros((b, c, need - t), x_input.dtype(), x_input.device())?;
                Tensor::cat(&[&pad, x_input], 2)?
            }
        };
        self.conv_state = Some(state);
        Ok(())
    }

    /// Single-step recurrent GDN update.  All inputs are (B, num_heads, head_dim).
    /// Returns `(output (B, num_heads, head_v_dim), new_state)`.
    fn recurrent_step(
        &self,
        state: &Tensor,          // (B, H, Hk, Hv)
        q: &Tensor,              // (B, H, Hk)  — L2-normalised & scaled
        k: &Tensor,              // (B, H, Hk)  — L2-normalised
        v: &Tensor,              // (B, H, Hv)
        g_exp: &Tensor,          // (B, H)  — exp(decay), >= 0
        beta: &Tensor,           // (B, H)  — sigmoid(b), in [0,1]
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let g4d    = g_exp.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // (B,H,1,1)
        let beta2d = beta.unsqueeze(D::Minus1)?;                         // (B,H,1)

        // Decay state first (matches HF torch_recurrent_gated_delta_rule)
        let state = state.broadcast_mul(&g4d)?; // S = S * g

        // kv_pred = S · k  (prediction uses DECAYED state)
        let kv_pred = state.broadcast_mul(&k.unsqueeze(D::Minus1)?)?.sum(D::Minus2)?;

        // delta = (v - kv_pred) * beta  (B,H,Hv)
        let delta = v.sub(&kv_pred)?.broadcast_mul(&beta2d)?;

        // outer(k, delta): (B,H,Hk,1) × (B,H,1,Hv) → (B,H,Hk,Hv)
        let outer = k.unsqueeze(D::Minus1)?.broadcast_mul(&delta.unsqueeze(D::Minus2)?)?;
        let new_state = state.add(&outer)?;

        // out = S_new · q:  (B,H,Hk,Hv) × q(B,H,Hk,1) → sum dim 2 → (B,H,Hv)
        let out = new_state.broadcast_mul(&q.unsqueeze(D::Minus1)?)?.sum(D::Minus2)?;

        Ok((out, new_state))
    }

    fn forward(&mut self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let is_decode = self.recurrent_state.is_some() && t == 1;

        // ── 1. Projections ────────────────────────────────────────────
        let qkv = self.in_proj_qkv.forward(xs)?;  // (B, T, conv_dim)
        let z   = self.in_proj_z.forward(xs)?;     // (B, T, value_dim)
        let b_proj = self.in_proj_b.forward(xs)?;  // (B, T, num_heads)
        let a_proj = self.in_proj_a.forward(xs)?;  // (B, T, num_heads)

        // ── 2. Causal conv1d + SiLU ───────────────────────────────────
        // Transpose to (B, C, T) before updating/using the conv state.
        let qkv_input_t = qkv.transpose(1, 2)?.contiguous()?; // (B, C, T) — pre-conv input

        // Apply conv FIRST using the OLD conv state as left-padding (decode),
        // or zero-padding (prefill).  Must come before state update.
        let qkv = self.apply_conv1d(&qkv, is_decode)?;  // (B, T, conv_dim)

        // Now update the conv state with the new input tokens.
        self.update_conv_state(&qkv_input_t, t, is_decode)?;

        // ── 3. Split Q, K, V ─────────────────────────────────────────
        let q = qkv.narrow(2, 0,                       self.key_dim)?; // (B,T,key_dim)
        let k = qkv.narrow(2, self.key_dim,            self.key_dim)?;
        let v = qkv.narrow(2, self.key_dim * 2,        self.value_dim)?;

        let q = q.reshape((b, t, self.num_heads, self.head_k_dim))?;
        let k = k.reshape((b, t, self.num_heads, self.head_k_dim))?;
        let v = v.reshape((b, t, self.num_heads, self.head_v_dim))?;

        // ── 4. L2-normalise Q, K; scale Q ────────────────────────────
        let q = l2_normalize(&q)?;
        let k = l2_normalize(&k)?;
        let scale = 1.0 / (self.head_k_dim as f64).sqrt();
        let q = (q * scale)?;

        // ── 5. Decay & beta ───────────────────────────────────────────
        // beta = sigmoid(b_proj)  (B, T, H)
        let beta = candle_nn::ops::sigmoid(&b_proj)?;
        // g = -A_log.exp() * softplus(a_proj + dt_bias)
        let a_log_f = self.a_log.to_dtype(DType::F32)?;
        let dt_f    = self.dt_bias.to_dtype(DType::F32)?;
        let a_proj_f = a_proj.to_dtype(DType::F32)?;
        let g = {
            let sp = softplus(&a_proj_f.broadcast_add(&dt_f)?)?; // (B,T,H)
            let decay = a_log_f.exp()?;                           // (H,)
            sp.broadcast_mul(&decay)?.neg()?                      // g in (-inf, 0]
        };
        // exp(g) = actual decay factor in [0, 1]
        let g_exp = g.exp()?.to_dtype(xs.dtype())?;
        let beta  = beta.to_dtype(xs.dtype())?;

        // Transpose heads: (B, T, H, dim) → per-step we'll slice T
        let q = q.transpose(1, 2)?; // (B, H, T, Hk)
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let g_exp = g_exp.transpose(1, 2)?;   // (B, H, T)
        let beta  = beta.transpose(1, 2)?;

        // ── 6. Recurrent computation ──────────────────────────────────
        let mut state = match &self.recurrent_state {
            Some(s) => s.clone(),
            None => Tensor::zeros(
                (b, self.num_heads, self.head_k_dim, self.head_v_dim),
                xs.dtype(), xs.device(),
            )?,
        };

        let mut step_outputs = Vec::with_capacity(t);
        for pos in 0..t {
            let q_t    = q.narrow(2, pos, 1)?.squeeze(2)?;      // (B, H, Hk)
            let k_t    = k.narrow(2, pos, 1)?.squeeze(2)?;
            let v_t    = v.narrow(2, pos, 1)?.squeeze(2)?;
            let g_t    = g_exp.narrow(2, pos, 1)?.squeeze(2)?;  // (B, H)
            let beta_t = beta.narrow(2, pos, 1)?.squeeze(2)?;

            let (out_t, new_state) = self.recurrent_step(&state, &q_t, &k_t, &v_t, &g_t, &beta_t)?;
            state = new_state;
            step_outputs.push(out_t.unsqueeze(1)?); // (B, 1, H, Hv)
        }
        self.recurrent_state = Some(state);

        // Concat → (B, T, H, Hv) → (B, T, value_dim)
        let core_out = Tensor::cat(&step_outputs, 1)?;          // (B, T, H, Hv)
        let core_out = core_out.reshape((b, t, self.value_dim))?;

        // ── 7. Gated RMSNorm ─────────────────────────────────────────
        // Reshape to (B*T*H, Hv) for per-head norm, z to same
        let bth = b * t * self.num_heads;
        let co_flat = core_out.reshape((bth, self.head_v_dim))?;
        let z_flat  = z.reshape((b * t, self.num_heads, self.head_v_dim))?
                       .reshape((bth, self.head_v_dim))?;

        let normed = self.norm.forward(&co_flat, &z_flat)?; // (B*T*H, Hv)

        let normed = normed.reshape((b, t, self.value_dim))?;

        // ── 8. Output projection ──────────────────────────────────────
        self.out_proj.forward(&normed)
    }

    fn clear_state(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

// ── Text: Attention (Qwen3 with QK-norm, GQA) ───────────────────────

struct TextAttention {
    qkv_proj: Option<Linear>,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    q_dim: usize,
    kv_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    /// Number of head dimensions that receive RoPE (partial_rotary_factor * head_dim).
    /// For Qwen3.5 this is 64 (25% of 256); for Qwen3-VL it equals head_dim.
    rope_dim: usize,
    /// When true, q_proj output is doubled: first half = query, second half = gate.
    /// After attention: attn_out = attn_out * sigmoid(gate) before o_proj.
    attn_output_gate: bool,
    kv_cache: Option<(Tensor, Tensor)>,
    cache_seq_len: usize,
}

impl TextAttention {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h   = cfg.hidden_size;
        let nh  = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd  = cfg.head_dim;
        let q_dim  = nh * hd;
        let kv_dim = nkv * hd;
        let gate   = cfg.attn_output_gate;

        // q_proj output: doubled when using gated output (query + gate both nh*hd)
        let q_proj_out = if gate { q_dim * 2 } else { q_dim };
        let q_proj = linear_no_bias(h, q_proj_out, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(h, kv_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(h, kv_dim, vb.pp("v_proj"))?;

        // Fused QKV: merge weights into single matmul.
        // For gated mode, only fuse the query-half of q_proj (first q_dim rows) with k, v.
        let qkv_proj = if gate {
            // Can't fuse when q has extra gate rows — skip fusion
            None
        } else {
            let qkv_w = Tensor::cat(&[q_proj.weight(), k_proj.weight(), v_proj.weight()], 0)?;
            Some(Linear::new(qkv_w, None))
        };

        Ok(Self {
            qkv_proj,
            q_proj,
            k_proj,
            v_proj,
            o_proj: linear_no_bias(q_dim, h, vb.pp("o_proj"))?,
            q_norm: if cfg.is_hybrid() { rms_norm_qwen35(hd, cfg.rms_norm_eps, vb.pp("q_norm"))? } else { rms_norm(hd, cfg.rms_norm_eps, vb.pp("q_norm"))? },
            k_norm: if cfg.is_hybrid() { rms_norm_qwen35(hd, cfg.rms_norm_eps, vb.pp("k_norm"))? } else { rms_norm(hd, cfg.rms_norm_eps, vb.pp("k_norm"))? },
            q_dim,
            kv_dim,
            num_heads: nh,
            num_kv_heads: nkv,
            num_kv_groups: nh / nkv,
            head_dim: hd,
            rope_dim: cfg.rope_dim(),
            attn_output_gate: gate,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    /// `batch_kv_info`: for batch decode, `Some((kv_lens, original_max_kv))` enables
    /// per-sequence attention that produces results identical to sequential decode.
    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;

        // When attn_output_gate: q_proj outputs num_heads * head_dim * 2.
        // HF interleaved layout: each head h has [q_h(head_dim), gate_h(head_dim)].
        // Must reshape to (B, T, num_heads, head_dim*2) then split on the last dim.
        let (q, gate_signal, k, v) = if self.attn_output_gate {
            let q_full = self.q_proj.forward(xs)?;  // (B, T, num_heads * head_dim * 2)
            let q_full_r = q_full.reshape((b, seq_len, self.num_heads, self.head_dim * 2))?;
            let q_half = q_full_r.narrow(D::Minus1, 0, self.head_dim)?
                .contiguous()?.reshape((b, seq_len, self.q_dim))?;
            let gate = q_full_r.narrow(D::Minus1, self.head_dim, self.head_dim)?
                .contiguous()?.reshape((b, seq_len, self.q_dim))?;
            (q_half, Some(gate), self.k_proj.forward(xs)?, self.v_proj.forward(xs)?)
        } else if let Some(ref qkv) = self.qkv_proj {
            let qkv_out = qkv.forward(xs)?;
            let q = qkv_out.narrow(D::Minus1, 0, self.q_dim)?;
            let k = qkv_out.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
            let v = qkv_out.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;
            (q, None, k, v)
        } else {
            (self.q_proj.forward(xs)?, None, self.k_proj.forward(xs)?, self.v_proj.forward(xs)?)
        };

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;

        // QK-norm (per head)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to (b, heads, seq, dim)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply M-RoPE (partial if rope_dim < head_dim)
        let (q, k) = if self.rope_dim < self.head_dim {
            // Qwen3.5: only rotate first rope_dim dimensions; leave the rest unchanged
            let rd = self.rope_dim;
            let pass = self.head_dim - rd;
            let q_r  = candle_nn::rotary_emb::rope(&q.narrow(D::Minus1, 0, rd)?.contiguous()?, cos, sin)?;
            let q_p  = q.narrow(D::Minus1, rd, pass)?.contiguous()?;
            let k_r  = candle_nn::rotary_emb::rope(&k.narrow(D::Minus1, 0, rd)?.contiguous()?, cos, sin)?;
            let k_p  = k.narrow(D::Minus1, rd, pass)?.contiguous()?;
            (Tensor::cat(&[&q_r, &q_p], D::Minus1)?,
             Tensor::cat(&[&k_r, &k_p], D::Minus1)?)
        } else {
            (candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?,
             candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?)
        };
        let v = v.contiguous()?;

        // Pre-allocated KV cache with slice_set: O(1) per decode step
        let (k, v) = match &self.kv_cache {
            None => {
                let (b_k, h_k, s_k, d_k) = k.dims4()?;
                let room = 256;
                let buf_k = Tensor::zeros((b_k, h_k, s_k + room, d_k), k.dtype(), k.device())?;
                let buf_v = Tensor::zeros((b_k, h_k, s_k + room, d_k), v.dtype(), v.device())?;
                buf_k.slice_set(&k, 2, 0)?;
                buf_v.slice_set(&v, 2, 0)?;
                self.kv_cache = Some((buf_k, buf_v));
                self.cache_seq_len = s_k;
                (k, v)
            }
            Some((buf_k, buf_v)) => {
                let new_total = self.cache_seq_len + seq_len;
                let buf_len = buf_k.dim(2)?;
                if new_total <= buf_len {
                    buf_k.slice_set(&k, 2, self.cache_seq_len)?;
                    buf_v.slice_set(&v, 2, self.cache_seq_len)?;
                } else {
                    // Rare: need to grow buffer
                    let (b_k, h_k, _, d_k) = buf_k.dims4()?;
                    let new_buf_k = Tensor::zeros((b_k, h_k, new_total + 256, d_k), buf_k.dtype(), buf_k.device())?;
                    let new_buf_v = Tensor::zeros((b_k, h_k, new_total + 256, d_k), buf_v.dtype(), buf_v.device())?;
                    new_buf_k.slice_set(&buf_k.narrow(2, 0, self.cache_seq_len)?, 2, 0)?;
                    new_buf_v.slice_set(&buf_v.narrow(2, 0, self.cache_seq_len)?, 2, 0)?;
                    new_buf_k.slice_set(&k, 2, self.cache_seq_len)?;
                    new_buf_v.slice_set(&v, 2, self.cache_seq_len)?;
                    self.kv_cache = Some((new_buf_k, new_buf_v));
                }
                let (buf_k, buf_v) = self.kv_cache.as_ref().unwrap();
                let k_view = buf_k.narrow(2, 0, new_total)?;
                let v_view = buf_v.narrow(2, 0, new_total)?;
                self.cache_seq_len = new_total;
                (k_view, v_view)
            }
        };

        // GQA-grouped SDPA for decode (seq_len=1): avoid expanding KV heads
        if self.num_kv_groups > 1 && seq_len == 1 {
            let scale = 1.0 / (self.head_dim as f64).sqrt();

            // Batch decode: when KV lengths differ, use per-sequence attention
            // over only real data (no padding). When all equal, fall through
            // to the fast batched GQA path below.
            if let Some((kv_lens, original_max_kv)) = batch_kv_info {
                if b > 1 && kv_lens.iter().any(|&l| l != original_max_kv) {
                    let rounds_done = self.cache_seq_len - original_max_kv;
                    let mut outputs = Vec::with_capacity(b);
                    for i in 0..b {
                        let q_i = q.narrow(0, i, 1)?; // [1, nh, 1, hd]
                        let offset = original_max_kv - kv_lens[i];
                        let real_len = kv_lens[i] + rounds_done;
                        let k_i = k.narrow(0, i, 1)?.narrow(2, offset, real_len)?.contiguous()?;
                        let v_i = v.narrow(0, i, 1)?.narrow(2, offset, real_len)?.contiguous()?;

                        let q_g = (q_i.reshape((1, self.num_kv_heads, self.num_kv_groups, self.head_dim))? * scale)?;
                        let k_t = k_i.transpose(2, 3)?.contiguous()?;
                        let attn = q_g.contiguous()?.matmul(&k_t)?;
                        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
                        let out = attn.matmul(&v_i)?;
                        outputs.push(out);
                    }
                    let out = Tensor::cat(&outputs, 0)?; // [N, nh_kv, groups, hd]
                    let out = out
                        .reshape((b, self.num_heads, self.head_dim))?
                        .reshape((b, 1, self.num_heads * self.head_dim))?;
                    return self.proj_with_gate(out, gate_signal.as_ref());
                }
            }

            // Fast batched GQA: b=1, or batch with equal kv_lens (no padding)
            let q_g = (q.reshape((b, self.num_kv_heads, self.num_kv_groups, self.head_dim))? * scale)?;
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let attn = q_g.contiguous()?.matmul(&k_t)?;
            let attn = match mask {
                Some(m) => attn.broadcast_add(m)?,
                None => attn,
            };
            let attn = candle_nn::ops::softmax_last_dim(&attn)?;
            let out = attn.matmul(&v.contiguous()?)?;
            let out = out
                .reshape((b, self.num_heads, self.head_dim))?
                .reshape((b, 1, self.num_heads * self.head_dim))?;
            return self.proj_with_gate(out, gate_signal.as_ref());
        }

        // Standard path (prefill or num_kv_groups==1)
        // Flash Attention dispatch (handles GQA natively, no KV head expansion needed)
        let attn_output = crate::fused_ops::attention::scaled_dot_product_attention(
            &q.contiguous()?,
            &k,
            &v,
            mask,
            seq_len > 1, // causal for prefill
        )?;
        let out = attn_output.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        self.proj_with_gate(out, gate_signal.as_ref())
    }

    /// Apply optional sigmoid gate then o_proj.
    /// `gate`: flat `(B, T, q_dim)` from the doubled q_proj (qwen3.5 gated attention).
    fn proj_with_gate(&self, out: Tensor, gate: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let out = if let Some(g) = gate {
            out.mul(&candle_nn::ops::sigmoid(g)?)?
        } else {
            out
        };
        self.o_proj.forward(&out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        self.cache_seq_len = 0;
    }
}

// ── Text: MLP (SwiGLU) ──────────────────────────────────────────────

struct TextMLP {
    gate_up_proj: Option<Linear>,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl TextMLP {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = linear_no_bias(h, i, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(h, i, vb.pp("up_proj"))?;

        // Fused gate+up: merge into single matmul
        let gate_up_proj = {
            let gu_w = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
            Some(Linear::new(gu_w, None))
        };

        Ok(Self {
            gate_up_proj,
            gate_proj,
            up_proj,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
            intermediate_size: i,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if let Some(ref gate_up) = self.gate_up_proj {
            let gu = gate_up.forward(x)?;
            #[cfg(feature = "cuda")]
            {
                if gu.device().is_cuda() {
                    let activated = crate::fused_ops::fused_silu_mul(
                        &gu.contiguous()?,
                        self.intermediate_size,
                    )?;
                    return self.down_proj.forward(&activated);
                }
            }
            // CPU fallback
            let gate = gu.narrow(D::Minus1, 0, self.intermediate_size)?;
            let up = gu.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
            let gate = candle_nn::Activation::Silu.forward(&gate)?;
            self.down_proj.forward(&(gate * up)?)
        } else {
            let gate = self.gate_proj.forward(x)?.apply(&Activation::Silu)?;
            let up = self.up_proj.forward(x)?;
            self.down_proj.forward(&(gate * up)?)
        }
    }
}

// ── Text: Decoder Layer ──────────────────────────────────────────────

struct TextDecoderLayer {
    self_attn: TextAttention,
    mlp: TextMLP,
    input_ln: RmsNorm,
    post_attn_ln_weight: Tensor,
    rms_norm_eps: f64,
}

impl TextDecoderLayer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let post_attn_ln_weight = ln_weight_for_cfg(
            cfg.hidden_size, cfg, vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn: TextAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: TextMLP::new(cfg, vb.pp("mlp"))?,
            input_ln: if cfg.is_hybrid() {
                rms_norm_qwen35(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?
            } else {
                rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?
            },
            post_attn_ln_weight,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, mask, batch_kv_info)?;

        // Fused: new_residual = residual + attn_out; normalized = rmsnorm(new_residual)
        let (new_residual, h) = crate::fused_ops::fused_add_rmsnorm(
            residual,
            &xs,
            &self.post_attn_ln_weight,
            self.rms_norm_eps,
        )?;

        let h = self.mlp.forward(&h)?;
        &new_residual + h
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Text: Linear (GDN) Decoder Layer ─────────────────────────────────

struct LinearDecoderLayer {
    linear_attn: GatedDeltaNet,
    mlp:         TextMLP,
    input_ln:    RmsNorm,
    post_attn_ln_weight: Tensor,
    rms_norm_eps: f64,
}

impl LinearDecoderLayer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let post_attn_ln_weight = ln_weight_for_cfg(
            cfg.hidden_size, cfg, vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            linear_attn: GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?,
            mlp:         TextMLP::new(cfg, vb.pp("mlp"))?,
            input_ln:    rms_norm_qwen35(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attn_ln_weight,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&mut self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = self.linear_attn.forward(&xs)?;

        let (new_residual, h) = crate::fused_ops::fused_add_rmsnorm(
            residual,
            &xs,
            &self.post_attn_ln_weight,
            self.rms_norm_eps,
        )?;
        let h = self.mlp.forward(&h)?;
        &new_residual + h
    }

    fn clear_state(&mut self) {
        self.linear_attn.clear_state();
    }
}

// ── Text: Hybrid Decoder Layer ────────────────────────────────────────

enum HybridDecoderLayer {
    Full(TextDecoderLayer),
    Linear(LinearDecoderLayer),
}

impl HybridDecoderLayer {
    fn new(cfg: &TextConfig, layer_idx: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        match cfg.layer_type(layer_idx) {
            "linear_attention" => Ok(Self::Linear(LinearDecoderLayer::new(cfg, vb)?)),
            _                  => Ok(Self::Full(TextDecoderLayer::new(cfg, vb)?)),
        }
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Full(l)   => l.forward(xs, cos, sin, mask, batch_kv_info),
            Self::Linear(l) => l.forward(xs),
        }
    }

    fn clear_cache(&mut self) {
        match self {
            Self::Full(l)   => l.clear_kv_cache(),
            Self::Linear(l) => l.clear_state(),
        }
    }

    fn kv_cache_bytes(&self) -> u64 {
        match self {
            Self::Full(l) => l.self_attn.kv_cache.as_ref().map(|(k, v)| {
                (k.elem_count() + v.elem_count()) as u64 * k.dtype().size_in_bytes() as u64
            }).unwrap_or(0),
            Self::Linear(_) => 0,
        }
    }

    fn get_kv_cache(&self) -> Option<(Tensor, Tensor)> {
        match self {
            Self::Full(l) => l.self_attn.kv_cache.as_ref().map(|(k, v)| {
                let len = l.self_attn.cache_seq_len;
                if len > 0 && len < k.dim(2).unwrap_or(0) {
                    (
                        k.narrow(2, 0, len).and_then(|t| t.contiguous()).unwrap_or_else(|_| k.clone()),
                        v.narrow(2, 0, len).and_then(|t| t.contiguous()).unwrap_or_else(|_| v.clone()),
                    )
                } else {
                    (k.clone(), v.clone())
                }
            }),
            Self::Linear(_) => None,
        }
    }

    fn set_kv_cache(&mut self, cache: Option<(Tensor, Tensor)>) {
        if let Self::Full(l) = self {
            match cache {
                Some((k, v)) => {
                    let seq_len = k.dim(2).unwrap_or(0);
                    let room = 256;
                    let (b, h, _s, d) = k.dims4().unwrap();
                    let buf_k = Tensor::zeros((b, h, seq_len + room, d), k.dtype(), k.device()).unwrap();
                    let buf_v = Tensor::zeros((b, h, seq_len + room, d), v.dtype(), v.device()).unwrap();
                    buf_k.slice_set(&k, 2, 0).unwrap();
                    buf_v.slice_set(&v, 2, 0).unwrap();
                    l.self_attn.kv_cache = Some((buf_k, buf_v));
                    l.self_attn.cache_seq_len = seq_len;
                }
                None => { l.self_attn.kv_cache = None; l.self_attn.cache_seq_len = 0; }
            }
        }
    }

    fn set_batched_kv(
        &mut self,
        kv: Option<(Tensor, Tensor)>,
        seq_len: usize,
        extra_room: usize,
    ) -> candle_core::Result<()> {
        if let Self::Full(l) = self {
            match kv {
                Some((k, v)) => {
                    if extra_room > 0 {
                        let (b, h, s, d) = k.dims4()?;
                        let buf_k = Tensor::zeros((b, h, s + extra_room, d), k.dtype(), k.device())?;
                        let buf_v = Tensor::zeros((b, h, s + extra_room, d), v.dtype(), v.device())?;
                        buf_k.slice_set(&k, 2, 0)?;
                        buf_v.slice_set(&v, 2, 0)?;
                        l.self_attn.kv_cache = Some((buf_k, buf_v));
                    } else {
                        l.self_attn.kv_cache = Some((k, v));
                    }
                    l.self_attn.cache_seq_len = seq_len;
                }
                None => { l.self_attn.kv_cache = None; l.self_attn.cache_seq_len = 0; }
            }
        }
        Ok(())
    }

    fn extract_batch_kv_for_seqs(
        &mut self,
        n_seqs: usize,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Option<(Tensor, Tensor)>>> {
        match self {
            Self::Full(l) => {
                let mut results = Vec::with_capacity(n_seqs);
                if let Some((ref full_k, ref full_v)) = l.self_attn.kv_cache {
                    for i in 0..n_seqs {
                        let row_k = full_k.narrow(0, i, 1)?;
                        let row_v = full_v.narrow(0, i, 1)?;
                        let total = kv_lens[i] + rounds_done;
                        let offset = original_max_kv - kv_lens[i];
                        results.push(Some((
                            row_k.narrow(2, offset, total)?.contiguous()?,
                            row_v.narrow(2, offset, total)?.contiguous()?,
                        )));
                    }
                } else {
                    for _ in 0..n_seqs { results.push(None); }
                }
                l.self_attn.kv_cache = None;
                l.self_attn.cache_seq_len = 0;
                Ok(results)
            }
            Self::Linear(l) => {
                l.clear_state();
                Ok((0..n_seqs).map(|_| None).collect())
            }
        }
    }
}

// ── Helper: scatter vision features into full sequence ───────────────

fn scatter_vision_features(
    vis_feat: &Tensor,
    mask_vec: &[f32],
    seq_len: usize,
    hidden_size: usize,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    // vis_feat: (vis_tokens, hidden), mask_vec: bool-like per position
    // Returns (1, seq_len, hidden) with vis_feat placed at mask positions.
    // Optimized: detect contiguous ranges and batch-assign (typically 1-2 ranges for 1-2 images).
    let num_vis = vis_feat.dim(0)?;
    let positions: Vec<usize> = mask_vec.iter()
        .enumerate()
        .filter(|(_, &m)| m > 0.5)
        .map(|(i, _)| i)
        .take(num_vis)
        .collect();

    if positions.is_empty() {
        return Tensor::zeros((1, seq_len, hidden_size), dtype, device);
    }

    let mut result = Tensor::zeros((1, seq_len, hidden_size), dtype, device)?;
    let mut vis_offset = 0;
    let mut range_start = positions[0];
    let mut range_len = 1;

    for i in 1..positions.len() {
        if positions[i] == positions[i - 1] + 1 {
            range_len += 1;
        } else {
            // Flush current contiguous range as single block
            let block = vis_feat.narrow(0, vis_offset, range_len)?.unsqueeze(0)?;
            result = result.slice_assign(
                &[0..1, range_start..range_start + range_len, 0..hidden_size],
                &block,
            )?;
            vis_offset += range_len;
            range_start = positions[i];
            range_len = 1;
        }
    }
    // Flush last range
    let block = vis_feat.narrow(0, vis_offset, range_len)?.unsqueeze(0)?;
    result = result.slice_assign(
        &[0..1, range_start..range_start + range_len, 0..hidden_size],
        &block,
    )?;
    Ok(result)
}

// ── Text: Decoder ────────────────────────────────────────────────────

pub struct TextDecoder {
    embed_tokens: Embedding,
    layers: Vec<HybridDecoderLayer>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl TextDecoder {
    /// Create a new TextDecoder. Set `load_lm_head` to false for embedding-only mode (saves ~1.5GB VRAM).
    pub fn new(cfg: &TextConfig, vb: VarBuilder, load_lm_head: bool) -> candle_core::Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(HybridDecoderLayer::new(cfg, i, vb.pp(&format!("layers.{}", i)))?);
        }
        let norm = if cfg.is_hybrid() {
            rms_norm_qwen35(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?
        } else {
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?
        };

        // Keep lm_head in F32 for BF16 models: cuBLAS computes BF16 matmul with F32
        // accumulation but casts output back to BF16, which can flip argmax at the EOS
        // boundary. F32 lm_head preserves full accumulation precision → correct token selection.
        let lm_head = if load_lm_head {
            let is_bf16 = vb.dtype() == DType::BF16;
            Some(if cfg.tie_word_embeddings {
                let w = embed_tokens.embeddings().clone();
                Linear::new(if is_bf16 { w.to_dtype(DType::F32)? } else { w }, None)
            } else {
                let raw = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
                if is_bf16 {
                    Linear::new(raw.weight().to_dtype(DType::F32)?, None)
                } else {
                    raw
                }
            })
        } else {
            None
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    pub fn embed(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        self.embed_tokens.forward(ids)
    }

    fn causal_mask(&self, tgt: usize, offset: usize) -> candle_core::Result<Tensor> {
        let mask: Vec<f32> = (0..tgt)
            .flat_map(|i| (0..tgt).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt, tgt), &self.device)?;
        let mask = if offset > 0 {
            let z = Tensor::zeros((tgt, offset), self.dtype, &self.device)?;
            Tensor::cat(&[&z, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((1, 1, tgt, tgt + offset))?.to_dtype(self.dtype)
    }

    fn forward_embeds(
        &mut self,
        xs: Tensor,
        cos: &Tensor,
        sin: &Tensor,
        offset: usize,
        deepstack_features: Option<&[Tensor]>,
        vision_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (_b, seq_len, _) = xs.dims3()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(self.causal_mask(seq_len, offset)?)
        };

        // Pre-compute scattered deepstack features to avoid borrow issues
        let scattered_ds = if let (Some(ds), Some(vm)) = (deepstack_features, vision_mask) {
            let mask_vec = vm.to_vec1::<f32>()?;
            let mut scattered = Vec::new();
            for feat in ds.iter() {
                let padded = scatter_vision_features(
                    feat, &mask_vec, seq_len, self.hidden_size, self.dtype, &self.device,
                )?;
                scattered.push(padded);
            }
            Some(scattered)
        } else {
            None
        };

        let mut h = xs;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward(&h, cos, sin, mask.as_ref(), None)?;

            // DeepStack: inject vision features at early layers
            if let Some(ref scattered) = scattered_ds {
                if i < scattered.len() {
                    h = (h + &scattered[i])?;
                }
            }

        }

        let h = self.norm.forward(&h)?;
        let h = h.narrow(1, seq_len - 1, 1)?;
        // Cast to F32 for lm_head when BF16 (lm_head weights stored in F32 for precision)
        let lm_head = self.lm_head.as_ref().expect("lm_head required for generation");
        let h = if self.dtype == DType::BF16 { h.to_dtype(DType::F32)? } else { h };
        let logits = h.apply(lm_head)?;

        Ok(logits)
    }

    /// Forward pass that returns ALL hidden states after norm (no lm_head, no token narrowing).
    /// Used by embedding models (e.g. ColQwen3) that need multi-vector representations.
    pub fn forward_hidden(
        &mut self,
        xs: Tensor,
        cos: &Tensor,
        sin: &Tensor,
        deepstack_features: Option<&[Tensor]>,
        vision_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (_b, seq_len, _) = xs.dims3()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(self.causal_mask(seq_len, 0)?)
        };

        let scattered_ds = if let (Some(ds), Some(vm)) = (deepstack_features, vision_mask) {
            let mask_vec = vm.to_vec1::<f32>()?;
            let mut scattered = Vec::new();
            for feat in ds.iter() {
                let padded = scatter_vision_features(
                    feat, &mask_vec, seq_len, self.hidden_size, self.dtype, &self.device,
                )?;
                scattered.push(padded);
            }
            Some(scattered)
        } else {
            None
        };

        let mut h = xs;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward(&h, cos, sin, mask.as_ref(), None)?;
            if let Some(ref scattered) = scattered_ds {
                if i < scattered.len() {
                    h = (h + &scattered[i])?;
                }
            }
        }

        self.norm.forward(&h)
    }

    fn forward_ids(
        &mut self,
        ids: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        offset: usize,
    ) -> candle_core::Result<Tensor> {
        let embeds = self.embed(ids)?;
        self.forward_embeds(embeds, cos, sin, offset, None, None)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total bytes held by the model's KV caches (no GPU copies).
    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.layers.iter().map(|l| l.kv_cache_bytes()).sum()
    }

    /// Extract per-layer KV caches, narrowed to the valid `cache_seq_len`.
    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers.iter().map(|l| l.get_kv_cache()).collect()
    }

    /// Restore per-layer KV caches into the model.
    pub fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        for (layer, cache) in self.layers.iter_mut().zip(caches) {
            layer.set_kv_cache(cache);
        }
    }

    // ── Batched Decode ──────────────────────────────────────────────────

    /// Pad per-sequence KV caches to the same length and load into model layers.
    /// Returns `(kv_lens, max_kv_len)`.
    pub fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        let kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;
        let device = self.embed_tokens.embeddings().device();

        let kv_lens: Vec<usize> = seq_kv_caches
            .iter()
            .map(|caches| {
                // Use the first non-None cache to determine KV length.
                // Hybrid models (e.g. Qwen3.5) have GDN layers (no KV cache) before
                // full-attention layers, so caches[0] is None — must skip those.
                caches
                    .iter()
                    .find_map(|c| c.as_ref().map(|(k, _)| k.dim(2).unwrap_or(0)))
                    .unwrap_or(0)
            })
            .collect();
        let max_kv_len = kv_lens.iter().copied().max().unwrap_or(0);

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_caches: Vec<&Option<(Tensor, Tensor)>> =
                seq_kv_caches.iter().map(|seq| &seq[layer_idx]).collect();

            let batched_kv = crate::models::qwen3::modeling::pad_and_stack_kv_caches(
                &layer_caches,
                max_kv_len,
                kv_heads,
                head_dim,
                device,
                self.dtype,
            )?;

            layer.set_batched_kv(batched_kv, max_kv_len, extra_room)?;
        }

        Ok((kv_lens, max_kv_len))
    }

    /// Run one batched decode step with pre-computed M-RoPE cos/sin.
    pub fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        let mut h = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        for (_li, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward(&h, cos, sin, attention_mask, batch_kv_info)?;
        }

        let h = self.norm.forward(&h)?;
        let h = h.narrow(1, 0, 1)?;
        let lm_head = self.lm_head.as_ref().expect("lm_head required for generation");
        let h = if self.dtype == DType::BF16 { h.to_dtype(DType::F32)? } else { h };
        h.apply(lm_head)
    }

    /// Extract per-sequence KV caches from batched state.
    pub fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        let n_seqs = kv_lens.len();
        let num_layers = self.layers.len();
        let mut result: Vec<Vec<Option<(Tensor, Tensor)>>> = (0..n_seqs)
            .map(|_| Vec::with_capacity(num_layers))
            .collect();

        for layer in self.layers.iter_mut() {
            let per_seq = layer.extract_batch_kv_for_seqs(
                n_seqs, kv_lens, original_max_kv, rounds_done,
            )?;
            for (i, kv) in per_seq.into_iter().enumerate() {
                result[i].push(kv);
            }
        }

        Ok(result)
    }

    // ── GDN (linear attention) state management ──────────────────────

    /// Extract per-sequence GDN recurrent+conv states from the batched model.
    /// Returns `[n_seqs][n_layers]` where linear layers have `Some((recurrent, conv))`.
    /// Clears the GDN state from all linear layers after extraction.
    pub fn extract_batch_gdn_states(
        &mut self,
        n_seqs: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        let n_layers = self.layers.len();
        let mut result: Vec<Vec<Option<(Tensor, Tensor)>>> =
            (0..n_seqs).map(|_| vec![None; n_layers]).collect();

        for (li, layer) in self.layers.iter_mut().enumerate() {
            if let HybridDecoderLayer::Linear(l) = layer {
                let r = l.linear_attn.recurrent_state.take();
                let c = l.linear_attn.conv_state.take();
                if let (Some(r), Some(c)) = (r, c) {
                    for i in 0..n_seqs {
                        let r_i = r.narrow(0, i, 1)?.contiguous()?;
                        let c_i = c.narrow(0, i, 1)?.contiguous()?;
                        result[i][li] = Some((r_i, c_i));
                    }
                }
            }
        }
        Ok(result)
    }

    /// Extract single-sequence (B=1) GDN states from the model.
    /// Used after prefill to save states before the model is cleared.
    pub fn extract_single_gdn_states(&mut self) -> Vec<Option<(Tensor, Tensor)>> {
        let n_layers = self.layers.len();
        let mut result: Vec<Option<(Tensor, Tensor)>> = vec![None; n_layers];
        for (li, layer) in self.layers.iter().enumerate() {
            if let HybridDecoderLayer::Linear(l) = layer {
                let r = l.linear_attn.recurrent_state.as_ref();
                let c = l.linear_attn.conv_state.as_ref();
                if let (Some(r), Some(c)) = (r, c) {
                    result[li] = Some((r.clone(), c.clone()));
                }
            }
        }
        result
    }

    /// Restore per-sequence GDN states into the model as a batch tensor.
    /// `states[i][li]` = Some((recurrent_i, conv_i)) for sequence i, linear layer li.
    /// None entries are filled with zeros.
    pub fn restore_batch_gdn_states(
        &mut self,
        states: &[Vec<Option<(Tensor, Tensor)>>],
    ) -> candle_core::Result<()> {
        if states.is_empty() {
            return Ok(());
        }
        let n_seqs = states.len();
        let device = self.embed_tokens.embeddings().device().clone();

        for (li, layer) in self.layers.iter_mut().enumerate() {
            if let HybridDecoderLayer::Linear(l) = layer {
                let gdn = &l.linear_attn;
                let h = gdn.num_heads;
                let hk = gdn.head_k_dim;
                let hv = gdn.head_v_dim;
                let c_dim = gdn.conv_dim;
                let ks = gdn.conv_ks;

                // Collect per-sequence recurrent and conv states
                let mut recurrent_rows = Vec::with_capacity(n_seqs);
                let mut conv_rows = Vec::with_capacity(n_seqs);

                for i in 0..n_seqs {
                    let entry = if li < states[i].len() { states[i][li].as_ref() } else { None };
                    match entry {
                        Some((r, c)) => {
                            recurrent_rows.push(r.clone());
                            conv_rows.push(c.clone());
                        }
                        None => {
                            recurrent_rows.push(
                                Tensor::zeros((1, h, hk, hv), self.dtype, &device)?
                            );
                            conv_rows.push(
                                Tensor::zeros((1, c_dim, ks - 1), self.dtype, &device)?
                            );
                        }
                    }
                }

                l.linear_attn.recurrent_state = Some(Tensor::cat(&recurrent_rows, 0)?);
                l.linear_attn.conv_state = Some(Tensor::cat(&conv_rows, 0)?);
            }
        }
        Ok(())
    }
}

// ── Qwen3VL: Main Model ─────────────────────────────────────────────

pub struct Qwen3VL {
    vision: VisionModel,
    decoder: TextDecoder,
    mrope: MRoPE,
    tokenizer: Tokenizer,
    config: Qwen3VLConfig,
    preproc_cfg: PreprocessorConfig,
    pub device: Device,
    dtype: DType,
    /// Cached normalization tensors (created once, reused for all images)
    img_mean: Tensor,
    img_std: Tensor,
}

pub struct Qwen3VLResult {
    pub text: String,
    pub tokens_generated: usize,
    pub duration_secs: f32,
}

impl Qwen3VL {
    pub fn from_local(path: impl AsRef<Path>, cpu: bool, bf16: bool) -> Result<Self> {
        let device = if cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
        let dtype = if bf16 && device.is_cuda() { DType::BF16 } else { DType::F32 };

        let base = path.as_ref();
        let config: Qwen3VLConfig =
            serde_json::from_str(&std::fs::read_to_string(base.join("config.json"))?)?;
        let preproc_cfg: PreprocessorConfig =
            serde_json::from_str(&std::fs::read_to_string(base.join("preprocessor_config.json"))?)?;
        let tokenizer = Tokenizer::from_file(base.join("tokenizer.json")).map_err(E::msg)?;

        let safetensors: Vec<std::path::PathBuf> = std::fs::read_dir(base)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        if safetensors.is_empty() {
            anyhow::bail!("No safetensors files found in {}", base.display());
        }
        let refs: Vec<&std::path::Path> = safetensors.iter().map(|p| p.as_path()).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&refs, dtype, &device)? };

        println!("Loading vision encoder...");
        let vision = VisionModel::new(vb.pp("model.visual"), &config.vision_config)?;

        println!("Loading text decoder...");
        let decoder = TextDecoder::new(&config.text_config, vb.pp("model.language_model"), true)?;

        println!("Initializing M-RoPE...");
        let mrope = MRoPE::new(&config.text_config, &device, dtype)?;

        // Pre-compute normalization tensors on device (reused for all images)
        let img_mean = Tensor::new(
            &[preproc_cfg.image_mean[0] as f32, preproc_cfg.image_mean[1] as f32, preproc_cfg.image_mean[2] as f32],
            &device,
        )?.reshape((3, 1, 1))?;
        let img_std = Tensor::new(
            &[preproc_cfg.image_std[0] as f32, preproc_cfg.image_std[1] as f32, preproc_cfg.image_std[2] as f32],
            &device,
        )?.reshape((3, 1, 1))?;

        println!("Model loaded!");

        Ok(Self { vision, decoder, mrope, tokenizer, config, preproc_cfg, device, dtype, img_mean, img_std })
    }

    fn compute_mrope_positions(
        &self,
        input_ids: &[u32],
        grid_thw: &Tensor,
    ) -> Result<(Tensor, i64)> {
        // Matches HF's Qwen3VLModel.get_rope_index:
        // - Text tokens: all 3 dims get the same sequential position
        // - Image tokens: t/h/w positions offset by text_pos before the image block
        // - After image: next text_pos = max across all dims + 1
        let grid_thw_vec = grid_thw.to_vec2::<u32>()?;
        let merge = self.config.vision_config.spatial_merge_size as i64;
        let image_token = self.config.image_token_id;

        let seq_len = input_ids.len();
        let mut t_pos = vec![0i64; seq_len];
        let mut h_pos = vec![0i64; seq_len];
        let mut w_pos = vec![0i64; seq_len];

        let mut text_pos: i64 = 0;
        let mut img_idx = 0;
        let mut i = 0;

        while i < seq_len {
            if input_ids[i] == image_token && img_idx < grid_thw_vec.len() {
                let grid = &grid_thw_vec[img_idx];
                let t = grid[0] as i64;
                let h = (grid[1] as i64) / merge;
                let w = (grid[2] as i64) / merge;
                let n_tokens = (t * h * w) as usize;

                let mut idx = 0;
                for frame in 0..t {
                    for row in 0..h {
                        for col in 0..w {
                            if i + idx < seq_len {
                                t_pos[i + idx] = text_pos + frame;
                                h_pos[i + idx] = text_pos + row;
                                w_pos[i + idx] = text_pos + col;
                            }
                            idx += 1;
                        }
                    }
                }

                // Advance text_pos past the max position used by the image block
                let max_dim = std::cmp::max(t, std::cmp::max(h, w));
                text_pos += max_dim;
                i += n_tokens;
                img_idx += 1;
            } else {
                t_pos[i] = text_pos;
                h_pos[i] = text_pos;
                w_pos[i] = text_pos;
                text_pos += 1;
                i += 1;
            }
        }

        let t_tensor = Tensor::new(t_pos, &self.device)?;
        let h_tensor = Tensor::new(h_pos, &self.device)?;
        let w_tensor = Tensor::new(w_pos, &self.device)?;
        let position_ids = Tensor::stack(&[t_tensor, h_tensor, w_tensor], 0)?; // (3, seq_len)
        // Return (position_ids, next_text_pos) — next_text_pos is needed for autoregressive generation
        Ok((position_ids, text_pos))
    }

    /// Build a chat prompt with the correct number of image pad tokens per image.
    /// `user_text` is the raw user message (no chat template wrapping).
    /// Returns the full prompt with <|vision_start|><|image_pad|>...<|vision_end|> per image.
    pub fn build_prompt(&self, user_text: &str, grid_thw: &Tensor) -> Result<String> {
        let grid_thw_vec = grid_thw.to_vec2::<u32>()?;
        let merge = self.config.vision_config.spatial_merge_size as u32;

        let mut image_blocks = String::new();
        for grid in &grid_thw_vec {
            let t = grid[0];
            let h = grid[1] / merge;
            let w = grid[2] / merge;
            let n_tokens = (t * h * w) as usize;
            image_blocks.push_str("<|vision_start|>");
            for _ in 0..n_tokens {
                image_blocks.push_str("<|image_pad|>");
            }
            image_blocks.push_str("<|vision_end|>");
        }

        // Qwen3.5 is a thinking model: the chat template appends
        // "<think>\n\n</think>\n\n" after the assistant prefix to suppress
        // the chain-of-thought and jump straight to the answer.
        let think_prefix = if self.config.text_config.is_hybrid() {
            "<think>\n\n</think>\n\n"
        } else {
            ""
        };
        Ok(format!(
            "<|im_start|>user\n{}{}<|im_end|>\n<|im_start|>assistant\n{}",
            image_blocks, user_text, think_prefix
        ))
    }

    /// GPU-accelerated greedy sampling
    fn greedy_sample(logits: &Tensor) -> anyhow::Result<u32> {
        // gpu_argmax only supports BF16; for F32 models use standard argmax
        #[cfg(feature = "cuda")]
        {
            if logits.device().is_cuda() && logits.dtype() == DType::BF16 {
                return Ok(crate::fused_ops::gpu_argmax(logits)?);
            }
        }
        Ok(logits.flatten_all()?.argmax(D::Minus1)?.to_dtype(DType::U32)?.to_scalar::<u32>()?)
    }

    pub fn recognize_stream<P: AsRef<Path>, F>(
        &mut self,
        image_paths: &[P],
        user_text: &str,
        max_new_tokens: usize,
        mut callback: F,
    ) -> Result<Qwen3VLResult>
    where
        F: FnMut(&str),
    {
        let start = Instant::now();

        // 1. Preprocess images
        let (pixel_values, grid_thw) = self.preprocess_images(image_paths)?;

        // 2. Run vision encoder
        let (image_embeds, deepstack_features) = self.vision.forward(&pixel_values, &grid_thw)?;

        // 3. Build prompt with correct number of <|image_pad|> tokens
        let prompt = self.build_prompt(user_text, &grid_thw)?;

        // 4. Tokenize
        let input_ids = self.tokenizer.encode(prompt.as_str(), false)
            .map_err(|e| E::msg(format!("Tokenizer: {}", e)))?
            .get_ids().to_vec();

        // 5. Compute M-RoPE position IDs
        let (position_ids, next_gen_pos) = self.compute_mrope_positions(&input_ids, &grid_thw)?;
        let (cos, sin) = self.mrope.forward(&position_ids)?;

        // 6. Build input embeddings: replace <|image_pad|> tokens with vision embeds
        let (input_embeds, vision_mask) = self.merge_embeddings(&input_ids, &image_embeds)?;

        // 7. First forward pass
        self.decoder.clear_kv_cache();
        let ds_ref: Vec<Tensor> = deepstack_features;
        let logits = self.decoder.forward_embeds(
            input_embeds, &cos, &sin, 0,
            Some(&ds_ref), Some(&vision_mask),
        )?;
        let mut next_token = Self::greedy_sample(&logits)?;

        let eos_id = self.tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);

        let mut generated = Vec::new();
        let prefill_len = input_ids.len();
        let mut gen_pos_val = next_gen_pos;
        let mut text = String::new();

        // 8. Autoregressive generation
        for step in 0..max_new_tokens {
            generated.push(next_token);
            if next_token == eos_id { break; }
            if let Ok(s) = self.tokenizer.decode(&[next_token], false) {
                callback(&s);
                text.push_str(&s);
            }

            let gen_pos = Tensor::new(&[gen_pos_val], &self.device)?;
            let gen_pos_3d = Tensor::stack(&[gen_pos.clone(), gen_pos.clone(), gen_pos.clone()], 0)?;
            let (cos_step, sin_step) = self.mrope.forward(&gen_pos_3d)?;

            let kv_offset = prefill_len + step;
            let token_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.decoder.forward_ids(&token_tensor, &cos_step, &sin_step, kv_offset)?;
            next_token = Self::greedy_sample(&logits)?;
            gen_pos_val += 1;
        }

        Ok(Qwen3VLResult {
            text,
            tokens_generated: generated.len(),
            duration_secs: start.elapsed().as_secs_f32(),
        })
    }

    fn merge_embeddings(
        &self,
        input_ids: &[u32],
        image_embeds: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        // Optimized: batch contiguous vision tokens into single narrow() calls
        // instead of per-token i(idx) + unsqueeze. For 500+ vision tokens per image,
        // this reduces tensor ops from O(N) to O(num_images).
        let image_token = self.config.image_token_id;
        let num_vis_tokens = image_embeds.dim(0)?;
        let mut parts: Vec<Tensor> = Vec::new();
        let mut mask_vals: Vec<f32> = Vec::new();
        let mut vis_idx = 0;
        let mut current_text: Vec<u32> = Vec::new();
        let mut vis_run_start: Option<usize> = None;
        let mut vis_run_len = 0usize;

        let flush_text = |text: &mut Vec<u32>, parts: &mut Vec<Tensor>, mask: &mut Vec<f32>, device: &Device, decoder: &TextDecoder| -> candle_core::Result<()> {
            if !text.is_empty() {
                let ids = Tensor::new(text.as_slice(), device)?;
                let emb = decoder.embed(&ids)?;
                mask.extend(std::iter::repeat(0.0f32).take(text.len()));
                parts.push(emb);
                text.clear();
            }
            Ok(())
        };

        for &id in input_ids {
            if id == image_token && vis_idx < num_vis_tokens {
                // Flush any pending text first
                flush_text(&mut current_text, &mut parts, &mut mask_vals, &self.device, &self.decoder)?;
                // Track contiguous vision token run
                if vis_run_start.is_none() {
                    vis_run_start = Some(vis_idx);
                    vis_run_len = 0;
                }
                vis_run_len += 1;
                vis_idx += 1;
            } else {
                // Flush any pending vision token run
                if let Some(start) = vis_run_start.take() {
                    let block = image_embeds.narrow(0, start, vis_run_len)?;
                    parts.push(block);
                    mask_vals.extend(std::iter::repeat(1.0f32).take(vis_run_len));
                    vis_run_len = 0;
                }
                current_text.push(id);
            }
        }
        // Flush remaining vision run
        if let Some(start) = vis_run_start.take() {
            let block = image_embeds.narrow(0, start, vis_run_len)?;
            parts.push(block);
            mask_vals.extend(std::iter::repeat(1.0f32).take(vis_run_len));
        }
        // Flush remaining text
        flush_text(&mut current_text, &mut parts, &mut mask_vals, &self.device, &self.decoder)?;

        let combined = Tensor::cat(&parts, 0)?.unsqueeze(0)?;
        let mask = Tensor::new(mask_vals, &self.device)?;
        Ok((combined, mask))
    }

    fn preprocess_images<P: AsRef<Path>>(
        &self,
        image_paths: &[P],
    ) -> Result<(Tensor, Tensor)> {
        let merge_size = self.preproc_cfg.merge_size;
        let patch_size = self.preproc_cfg.patch_size;
        let temporal_patch_size = self.preproc_cfg.temporal_patch_size;
        let factor = patch_size * merge_size;
        let min_pixels = self.preproc_cfg.size.shortest_edge;
        let max_pixels = self.preproc_cfg.size.longest_edge;

        let mut all_pixels = Vec::new();
        let mut all_grid_thw = Vec::new();

        for path in image_paths {
            let img = image::open(path.as_ref())?;
            let img = img.to_rgb8();
            let (w, h) = (img.width(), img.height());

            // Smart resize (match Crane's processor logic)
            let (rh, rw) = crate::utils::image_utils::smart_resize(
                h as usize, w as usize, factor, min_pixels, max_pixels,
            )?;
            let img = image::imageops::resize(&img, rw as u32, rh as u32, image::imageops::FilterType::CatmullRom);

            // GPU-accelerated normalization with cached mean/std tensors
            let raw: Vec<u8> = img.into_raw();
            let raw_tensor = Tensor::from_vec(raw, (rh, rw, 3), &Device::Cpu)?
                .permute((2, 0, 1))?
                .to_device(&self.device)?;
            let raw_f32 = (raw_tensor.to_dtype(DType::F32)? * (1.0 / 255.0))?;
            let tensor = raw_f32.broadcast_sub(&self.img_mean)?.broadcast_div(&self.img_std)?
                .unsqueeze(0)?.to_dtype(self.dtype)?;

            // Duplicate for temporal dim (images → 2 frames)
            let tensor = Tensor::cat(&[&tensor, &tensor], 0)?; // (2, 3, rh, rw)

            // Patchify (same as processor.rs process_vision_tensor)
            let t_dim = 2usize;
            let grid_t = t_dim / temporal_patch_size;
            let grid_h = rh / patch_size;
            let grid_w = rw / patch_size;

            let tensor = tensor.reshape(Shape::from(vec![
                grid_t, temporal_patch_size,
                3,
                grid_h / merge_size, merge_size, patch_size,
                grid_w / merge_size, merge_size, patch_size,
            ]))?;
            let tensor = tensor.permute(vec![0, 3, 6, 4, 7, 2, 1, 5, 8])?;
            let tensor = tensor.reshape((
                grid_t * grid_h * grid_w,
                3 * temporal_patch_size * patch_size * patch_size,
            ))?.contiguous()?;

            all_pixels.push(tensor);
            all_grid_thw.push(Tensor::from_vec(
                vec![grid_t as u32, grid_h as u32, grid_w as u32],
                (1, 3),
                &self.device,
            )?);
        }

        let pixel_values = Tensor::cat(&all_pixels, 0)?;
        let grid_thw = Tensor::cat(&all_grid_thw, 0)?;
        Ok((pixel_values, grid_thw))
    }

    // ── Public accessors for engine integration ──────────────────────

    /// Mutable reference to the text decoder.
    pub fn decoder_mut(&mut self) -> &mut TextDecoder {
        &mut self.decoder
    }

    /// Reference to the text decoder.
    pub fn decoder(&self) -> &TextDecoder {
        &self.decoder
    }

    /// Reference to the M-RoPE module.
    pub fn mrope(&self) -> &MRoPE {
        &self.mrope
    }

    /// Reference to the tokenizer.
    pub fn tokenizer_ref(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Reference to the model config.
    pub fn config(&self) -> &Qwen3VLConfig {
        &self.config
    }

    /// Model data type.
    pub fn model_dtype(&self) -> DType {
        self.dtype
    }

    // ── Engine helper: single decode step ────────────────────────────

    /// Decode one token given its ID, the current generation position,
    /// and the KV cache offset. Returns logits for the next token.
    pub fn decode_step(
        &mut self,
        token_id: u32,
        gen_pos: i64,
        kv_offset: usize,
    ) -> Result<Tensor> {
        let gen_pos_t = Tensor::new(&[gen_pos], &self.device)?;
        let gen_pos_3d = Tensor::stack(&[gen_pos_t.clone(), gen_pos_t.clone(), gen_pos_t], 0)?;
        let (cos, sin) = self.mrope.forward(&gen_pos_3d)?;
        let token_tensor = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
        let logits = self.decoder.forward_ids(&token_tensor, &cos, &sin, kv_offset)?;
        Ok(logits)
    }

    // ── Engine helper: full prefill from image bytes ─────────────────

    /// Preprocess images from raw bytes (no temp files), run vision encoder,
    /// build prompt, compute M-RoPE, merge embeddings, prefill.
    ///
    /// Returns `(logits, input_ids, next_gen_pos, prefill_len)`.
    pub fn prefill_from_bytes(
        &mut self,
        images: &[Vec<u8>],
        user_text: &str,
    ) -> Result<(Tensor, Vec<u32>, i64, usize)> {
        // 1. Preprocess images from bytes
        let (pixel_values, grid_thw) = self.preprocess_images_from_bytes(images)?;

        // 2. Vision encoder
        let (image_embeds, deepstack_features) = self.vision.forward(&pixel_values, &grid_thw)?;

        // 3. Build prompt
        let prompt = self.build_prompt(user_text, &grid_thw)?;

        // 4. Tokenize
        let input_ids = self.tokenizer.encode(prompt.as_str(), false)
            .map_err(|e| E::msg(format!("Tokenizer: {}", e)))?
            .get_ids().to_vec();

        // 5. Compute M-RoPE
        let (position_ids, next_gen_pos) = self.compute_mrope_positions(&input_ids, &grid_thw)?;
        let (cos, sin) = self.mrope.forward(&position_ids)?;

        // 6. Merge embeddings
        let (input_embeds, vision_mask) = self.merge_embeddings(&input_ids, &image_embeds)?;

        // 7. Forward (prefill)
        self.decoder.clear_kv_cache();
        let ds_ref: Vec<Tensor> = deepstack_features;
        let logits = self.decoder.forward_embeds(
            input_embeds, &cos, &sin, 0,
            Some(&ds_ref), Some(&vision_mask),
        )?;

        let prefill_len = input_ids.len();
        Ok((logits, input_ids, next_gen_pos, prefill_len))
    }

    /// Preprocess images from raw bytes (no temp files needed).
    fn preprocess_images_from_bytes(
        &self,
        images: &[Vec<u8>],
    ) -> Result<(Tensor, Tensor)> {
        let merge_size = self.preproc_cfg.merge_size;
        let patch_size = self.preproc_cfg.patch_size;
        let temporal_patch_size = self.preproc_cfg.temporal_patch_size;
        let factor = patch_size * merge_size;
        let min_pixels = self.preproc_cfg.size.shortest_edge;
        let max_pixels = self.preproc_cfg.size.longest_edge;

        let mut all_pixels = Vec::new();
        let mut all_grid_thw = Vec::new();

        for raw_bytes in images {
            let img = image::load_from_memory(raw_bytes)
                .map_err(|e| E::msg(format!("Failed to decode image from bytes: {e}")))?;
            let img = img.to_rgb8();
            let (w, h) = (img.width(), img.height());

            let (rh, rw) = crate::utils::image_utils::smart_resize(
                h as usize, w as usize, factor, min_pixels, max_pixels,
            )?;
            let img = image::imageops::resize(&img, rw as u32, rh as u32, image::imageops::FilterType::CatmullRom);

            // GPU-accelerated normalization
            let raw: Vec<u8> = img.into_raw();
            let raw_tensor = Tensor::from_vec(raw, (rh, rw, 3), &Device::Cpu)?
                .permute((2, 0, 1))?
                .to_device(&self.device)?;
            let raw_f32 = (raw_tensor.to_dtype(DType::F32)? * (1.0 / 255.0))?;
            let tensor = raw_f32.broadcast_sub(&self.img_mean)?.broadcast_div(&self.img_std)?
                .unsqueeze(0)?.to_dtype(self.dtype)?;

            let tensor = Tensor::cat(&[&tensor, &tensor], 0)?;

            let t_dim = 2usize;
            let grid_t = t_dim / temporal_patch_size;
            let grid_h = rh / patch_size;
            let grid_w = rw / patch_size;

            let tensor = tensor.reshape(Shape::from(vec![
                grid_t, temporal_patch_size,
                3,
                grid_h / merge_size, merge_size, patch_size,
                grid_w / merge_size, merge_size, patch_size,
            ]))?;
            let tensor = tensor.permute(vec![0, 3, 6, 4, 7, 2, 1, 5, 8])?;
            let tensor = tensor.reshape((
                grid_t * grid_h * grid_w,
                3 * temporal_patch_size * patch_size * patch_size,
            ))?.contiguous()?;

            all_pixels.push(tensor);
            all_grid_thw.push(Tensor::from_vec(
                vec![grid_t as u32, grid_h as u32, grid_w as u32],
                (1, 3),
                &self.device,
            )?);
        }

        let pixel_values = Tensor::cat(&all_pixels, 0)?;
        let grid_thw = Tensor::cat(&all_grid_thw, 0)?;
        Ok((pixel_values, grid_thw))
    }
}
