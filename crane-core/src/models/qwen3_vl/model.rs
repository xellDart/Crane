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
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub mrope_interleaved: bool,
}

fn default_head_dim() -> usize { 128 }

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
        let dim = cfg.head_dim;
        let theta = cfg.rope_theta;
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?;

        let mrope_section = cfg.rope_scaling.as_ref()
            .map(|s| s.mrope_section.clone())
            .unwrap_or_else(|| vec![24, 20, 20]);

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

        let min_section = *sections.iter().min().unwrap(); // 20

        // Build interleaved frequency table matching HF's apply_interleaved_mrope:
        //
        // HF computes: freqs[d, s, f] = inv_freq[f] * position[d, s]
        // Then interleaves: output[s, f] picks from T/H/W based on f's position:
        //   f % 3 == 0 → T (temporal)  for f < min_section*3
        //   f % 3 == 1 → H (height)   for f < min_section*3
        //   f % 3 == 2 → W (width)    for f < min_section*3
        //   f >= min_section*3 → T (remaining temporal frequencies)
        //
        // Each output[s, f] = inv_freq[f] * position_of_assigned_dim[s]
        let mut output = vec![vec![0f32; half_dim]; seq_len];
        let interleaved_end = min_section * 3; // 60

        for s in 0..seq_len {
            let tp = t_pos[s] as f32;
            let hp = h_pos[s] as f32;
            let wp = w_pos[s] as f32;

            // Interleaved region: [T,H,W,T,H,W,...] each using its own inv_freq[f]
            for i in 0..min_section {
                let base = i * 3;
                output[s][base]     = tp * inv_freq_vec[base];     // T at inv_freq[base]
                output[s][base + 1] = hp * inv_freq_vec[base + 1]; // H at inv_freq[base+1]
                output[s][base + 2] = wp * inv_freq_vec[base + 2]; // W at inv_freq[base+2]
            }

            // Remaining temporal frequencies (indices 60..63)
            for f in interleaved_end..half_dim {
                output[s][f] = tp * inv_freq_vec[f];
            }
        }

        let output = Tensor::new(output, device)?; // (seq_len, half_dim)
        let cos = output.cos()?.to_dtype(self.dtype)?;
        let sin = output.sin()?.to_dtype(self.dtype)?;
        Ok((cos, sin))
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
    kv_cache: Option<(Tensor, Tensor)>,
    cache_seq_len: usize,
}

impl TextAttention {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;

        let q_proj = linear_no_bias(h, q_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(h, kv_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(h, kv_dim, vb.pp("v_proj"))?;

        // Fused QKV: merge weights into single matmul
        let qkv_proj = {
            let qkv_w = Tensor::cat(&[q_proj.weight(), k_proj.weight(), v_proj.weight()], 0)?;
            Some(Linear::new(qkv_w, None))
        };

        Ok(Self {
            qkv_proj,
            q_proj,
            k_proj,
            v_proj,
            o_proj: linear_no_bias(q_dim, h, vb.pp("o_proj"))?,
            q_norm: rms_norm(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            q_dim,
            kv_dim,
            num_heads: nh,
            num_kv_heads: nkv,
            num_kv_groups: nh / nkv,
            head_dim: hd,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;

        // Fused QKV projection: 1 matmul instead of 3
        let (q, k, v) = if let Some(ref qkv) = self.qkv_proj {
            let qkv_out = qkv.forward(xs)?;
            let q = qkv_out.narrow(D::Minus1, 0, self.q_dim)?;
            let k = qkv_out.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
            let v = qkv_out.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;
            (q, k, v)
        } else {
            (self.q_proj.forward(xs)?, self.k_proj.forward(xs)?, self.v_proj.forward(xs)?)
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

        // Apply M-RoPE
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
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
            return self.o_proj.forward(&out);
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
        let post_attn_ln_weight = vb
            .pp("post_attention_layernorm")
            .get_with_hints(cfg.hidden_size, "weight", candle_nn::Init::Const(1.))?;
        Ok(Self {
            self_attn: TextAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: TextMLP::new(cfg, vb.pp("mlp"))?,
            input_ln: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
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
    ) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, mask)?;

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
    layers: Vec<TextDecoderLayer>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
}

impl TextDecoder {
    /// Create a new TextDecoder. Set `load_lm_head` to false for embedding-only mode (saves ~1.5GB VRAM).
    pub fn new(cfg: &TextConfig, vb: VarBuilder, load_lm_head: bool) -> candle_core::Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(TextDecoderLayer::new(cfg, vb.pp(&format!("layers.{}", i)))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

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
            h = layer.forward(&h, cos, sin, mask.as_ref())?;

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
        h.apply(lm_head)
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
            h = layer.forward(&h, cos, sin, mask.as_ref())?;
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
            layer.clear_kv_cache();
        }
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

        Ok(format!(
            "<|im_start|>user\n{}{}<|im_end|>\n<|im_start|>assistant\n",
            image_blocks, user_text
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
}
