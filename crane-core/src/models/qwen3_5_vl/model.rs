//! Qwen3.5-VL hybrid backbone — forward-only prefill port.
//!
//! Ported from the pre-trim `qwen3_vl.rs` (git 7adee9a) and reconciled against the
//! HuggingFace `transformers.models.qwen3_5` reference (v4.57) for the Argus-
//! Colqwen3.5-9B checkpoint. Weight prefixes: `visual.*`, `language_model.*`.
//!
//! Parity notes (things a reviewer must double-check against HF):
//!   * All TEXT RMSNorms use the `(1 + weight)` parameterization (Qwen3_5RMSNorm),
//!     i.e. weights are trained from zero-init. Applies to input_layernorm,
//!     post_attention_layernorm, q_norm, k_norm and the final `norm`. The GDN
//!     internal `norm` (Qwen3_5RMSNormGated) uses the weight AS-IS (no +1).
//!   * Gated Delta Net (linear attention): K/V head counts differ
//!     (num_k_heads=16, num_v_heads=32), so Q and K are `repeat_interleave`d by
//!     num_v_heads/num_k_heads = 2 before the delta rule — matching HF. The
//!     pre-trim code assumed equal head counts and OMITTED this; it is required here.
//!   * The delta rule is computed as a sequential recurrent scan in F32 (matching
//!     `torch_recurrent_gated_delta_rule`). HF's default *prefill* path uses the
//!     algebraically-identical chunked kernel (`chunk_gated_delta_rule`); results
//!     match up to F32 op-ordering, not bit-for-bit against the fused FLA kernel.
//!   * Gated full attention: q_proj emits 2*(num_heads*head_dim); reshaped to
//!     (.., num_heads, 2*head_dim) and chunked → (query, gate). q_norm/k_norm are
//!     per-head RMSNorm. Partial RoPE rotates only the first `rope_dim` (=64) dims.
//!     `attn_out = attn_out * sigmoid(gate)` before o_proj.
//!   * Attention uses the CURRENT crate convention: `rope_thd` on (B,S,H,D) +
//!     `scaled_dot_product_attention_bshd`, not the pre-trim `rope` + (B,H,S,D).
//!   * Vision pos-embed uses `fast_pos_embed_interpolate` (bilinear interpolation
//!     of the learned `visual.pos_embed.weight` table), copied verbatim from the
//!     current `qwen3_vl` tower. `deepstack_visual_indexes` is empty for Argus.

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{
    self, linear, linear_no_bias, Activation, Embedding, LayerNorm, Linear, RmsNorm, VarBuilder,
};
use serde::Deserialize;

// ── Norm helpers ─────────────────────────────────────────────────────

/// RMSNorm with the Qwen3.5 `(1 + weight)` parameterization (weights zero-init).
fn rms_norm_1plusw(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<RmsNorm> {
    let w = vb.get_with_hints(size, "weight", candle_nn::Init::Const(0.))?;
    let w = w.affine(1.0, 1.0)?; // stored_weight + 1
    Ok(RmsNorm::new(w, eps))
}

/// Raw `(1 + weight)` layernorm tensor for use with `fused_add_rmsnorm`.
fn ln_weight_1plusw(size: usize, vb: VarBuilder) -> candle_core::Result<Tensor> {
    let w = vb.get_with_hints(size, "weight", candle_nn::Init::Const(0.))?;
    w.affine(1.0, 1.0)
}

/// L2-normalize `x` along the last dim: `x * rsqrt(sum(x^2) + eps)` (HF `l2norm`, eps 1e-6).
fn l2_normalize(x: &Tensor) -> candle_core::Result<Tensor> {
    let denom = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
    x.broadcast_div(&denom)
}

/// Numerically-stable softplus: `max(x,0) + log(1 + exp(-|x|))`.
fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    let pos = x.clamp(0.0_f64, f64::MAX)?;
    let abs_x = x.abs()?;
    let inner = abs_x.neg()?.exp()?.add(&Tensor::ones_like(&abs_x)?)?;
    pos.add(&inner.log()?)
}

// ── Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35VLConfig {
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
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    /// Argus nests rope params under `rope_parameters` (theta 1e7, mrope [11,11,10]).
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    /// Top-level `rope_theta` fallback (absent in Argus text_config).
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    // Hybrid architecture fields.
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
    /// Full-attention layers use a sigmoid output gate on the query projection.
    #[serde(default)]
    pub attn_output_gate: bool,
}

impl TextConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().map(|p| p.rope_theta))
            .unwrap_or(1_000_000.0)
    }

    fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.partial_rotary_factor)
            .unwrap_or(1.0)
    }

    /// Number of head dims that receive RoPE: `head_dim * partial_rotary_factor` (=64 for Argus).
    pub fn rope_dim(&self) -> usize {
        ((self.head_dim as f64 * self.partial_rotary_factor()).round() as usize).max(2)
    }

    /// mrope_section — from `rope_parameters` ([11,11,10] for Argus).
    pub fn rope_mrope_section(&self) -> Vec<usize> {
        if let Some(ref p) = self.rope_parameters {
            if !p.mrope_section.is_empty() {
                return p.mrope_section.clone();
            }
        }
        vec![24, 20, 20]
    }

    /// Whether this config uses the hybrid GDN / full-attention architecture.
    pub fn is_hybrid(&self) -> bool {
        !self.layer_types.is_empty()
    }

    /// Layer kind at index `i`: `"linear_attention"` or `"full_attention"`.
    pub fn layer_type(&self, i: usize) -> &str {
        if self.layer_types.is_empty() {
            "full_attention"
        } else {
            &self.layer_types[i]
        }
    }

    /// True if layer `i` is a full (gated) attention layer.
    pub fn is_full_attention(&self, i: usize) -> bool {
        self.layer_type(i) == "full_attention"
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_interleaved: bool,
}

fn default_partial_rotary_factor() -> f64 {
    1.0
}
fn default_head_dim() -> usize {
    128
}
fn default_max_pos() -> usize {
    32768
}
fn default_linear_num_heads() -> usize {
    16
}
fn default_linear_head_dim() -> usize {
    128
}
fn default_conv_kernel_dim() -> usize {
    4
}

/// Re-export aliases so callers can name the types unambiguously.
pub type Qwen35VLTextConfig = TextConfig;
pub type Qwen35VLVisionConfig = VisionConfig;

// ── Vision: Patch Embedding ──────────────────────────────────────────

struct VisionPatchEmbed {
    proj: Linear,
}

impl VisionPatchEmbed {
    fn new(vb: VarBuilder, cfg: &VisionConfig) -> candle_core::Result<Self> {
        let w = vb.pp("proj").get_with_hints(
            (
                cfg.hidden_size,
                cfg.in_channels,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ),
            "weight",
            candle_nn::Init::Const(0.),
        )?;
        let b = vb
            .pp("proj")
            .get_with_hints(cfg.hidden_size, "bias", candle_nn::Init::Const(0.))?;
        let in_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let w_2d = w.reshape((cfg.hidden_size, in_dim))?;
        Ok(Self {
            proj: Linear::new(w_2d, Some(b)),
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.proj.forward(x)
    }
}

// ── Vision: Rotary Embedding (2D spatial) ────────────────────────────

struct VisionRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl VisionRotaryEmbedding {
    fn new(half_head_dim: usize, _device: &Device) -> candle_core::Result<Self> {
        let theta = 10000f64;
        let inv_freq: Vec<f32> = (0..half_head_dim / 2)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / half_head_dim as f64) as f32)
            .collect();
        Ok(Self { inv_freq })
    }

    fn forward(
        &self,
        grid_thw: &Tensor,
        merge_size: usize,
        device: &Device,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let grid_thw_vec = grid_thw.to_vec2::<u32>()?;
        let n_freq = self.inv_freq.len();

        let max_hw = grid_thw_vec
            .iter()
            .flat_map(|g| [g[1], g[2]])
            .max()
            .unwrap_or(1) as usize;
        let mut freq_table = vec![vec![0f32; n_freq]; max_hw];
        for pos in 0..max_hw {
            for j in 0..n_freq {
                freq_table[pos][j] = pos as f32 * self.inv_freq[j];
            }
        }

        let mut all_freqs: Vec<f32> = Vec::new();
        for grid in &grid_thw_vec {
            let t = grid[0] as usize;
            let h = grid[1] as usize;
            let w = grid[2] as usize;
            let merged_h = h / merge_size;
            let merged_w = w / merge_size;

            for _frame in 0..t {
                for br in 0..merged_h {
                    for bc in 0..merged_w {
                        for ir in 0..merge_size {
                            for ic in 0..merge_size {
                                let row = br * merge_size + ir;
                                let col = bc * merge_size + ic;
                                all_freqs.extend_from_slice(&freq_table[row]);
                                all_freqs.extend_from_slice(&freq_table[col]);
                            }
                        }
                    }
                }
            }
        }

        let total_patches = all_freqs.len() / (2 * n_freq);
        let half_rope_dim = 2 * n_freq;
        let freqs = Tensor::from_vec(all_freqs, (total_patches, half_rope_dim), device)?;
        Ok((freqs.cos()?, freqs.sin()?))
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
        Ok(Self {
            qkv: linear(h, 3 * h, vb.pp("qkv"))?,
            proj: linear(h, h, vb.pp("proj"))?,
            num_heads: cfg.num_heads,
            head_dim: h / cfg.num_heads,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (seq_len, _) = x.dims2()?;
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((seq_len, 3, self.num_heads, self.head_dim))?;
        let q = qkv.i((.., 0, .., ..))?.contiguous()?;
        let k = qkv.i((.., 1, .., ..))?.contiguous()?;
        let v = qkv.i((.., 2, .., ..))?.contiguous()?;

        let q = self.apply_vision_rope(&q, cos, sin)?;
        let k = self.apply_vision_rope(&k, cos, sin)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let out = crate::fused_ops::attention::scaled_dot_product_attention_3d(&q, &k, &v)?;

        let out = out
            .contiguous()?
            .reshape((seq_len, self.num_heads * self.head_dim))?;
        self.proj.forward(&out)
    }

    fn apply_vision_rope(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (_seq, _heads, dim) = x.dims3()?;
        let half = dim / 2;
        let cos = cos.unsqueeze(1)?;
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
        // Empirically the reference vision MLP matches the exact erf GELU
        // (`.gelu()` tanh-approx measurably worsened parity), despite the
        // "gelu_pytorch_tanh" label. Matches the current qwen3_vl tower.
        let x = self.fc1.forward(x)?.gelu_erf()?;
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
        let h = self.mlp.forward(&self.norm2.forward(&x)?)?;
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
        let norm_dim = if post_shuffle_norm {
            merged_dim
        } else {
            cfg.hidden_size
        };
        Ok(Self {
            norm: candle_nn::layer_norm(norm_dim, 1e-6, vb.pp("norm"))?,
            fc1: linear(merged_dim, merged_dim, vb.pp("linear_fc1"))?,
            fc2: linear(merged_dim, cfg.out_hidden_size, vb.pp("linear_fc2"))?,
            merged_dim,
            post_shuffle_norm,
        })
    }

    fn forward(&self, x: &Tensor, _grid_thw: &Tensor) -> candle_core::Result<Tensor> {
        let merged_dim = self.merged_dim;
        let merged = if !self.post_shuffle_norm {
            self.norm.forward(x)?.reshape(((), merged_dim))?
        } else {
            self.norm.forward(&x.reshape(((), merged_dim))?)?
        };
        self.fc2.forward(&self.fc1.forward(&merged)?.gelu()?)
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
    /// `vb` must already be prefixed to the vision tower (e.g. `vb.pp("visual")`).
    pub fn new(vb: VarBuilder, cfg: &VisionConfig) -> candle_core::Result<Self> {
        let patch_embed = VisionPatchEmbed::new(vb.pp("patch_embed"), cfg)?;
        let pos_embed = candle_nn::embedding(
            cfg.num_position_embeddings,
            cfg.hidden_size,
            vb.pp("pos_embed"),
        )?;
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rotary_emb = VisionRotaryEmbedding::new(head_dim / 2, vb.device())?;
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

    /// Returns `(merged image embeddings (num_merged_tokens, out_hidden_size), deepstack_features)`.
    /// `deepstack_features` is empty for Argus (no deepstack indexes).
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
    ) -> candle_core::Result<(Tensor, Vec<Tensor>)> {
        let device = pixel_values.device();
        let mut hidden = self.patch_embed.forward(pixel_values)?;

        let pos_embeds = self.fast_pos_embed_interpolate(grid_thw)?;
        hidden = (hidden + pos_embeds)?;

        let (cos, sin) = self
            .rotary_emb
            .forward(grid_thw, self.spatial_merge_size, device)?;
        let cos = cos.to_dtype(hidden.dtype())?;
        let sin = sin.to_dtype(hidden.dtype())?;

        let mut deepstack_features = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, &cos, &sin)?;
            if let Some(ds_idx) = self.deepstack_indexes.iter().position(|&x| x == i) {
                let feat = self.deepstack_mergers[ds_idx].forward(&hidden, grid_thw)?;
                deepstack_features.push(feat);
            }
        }

        let merged = self.merger.forward(&hidden, grid_thw)?;
        Ok((merged, deepstack_features))
    }

    /// Bilinear interpolation of the learned position table for variable resolution.
    /// Matches HF `Qwen3VLVisionModel.fast_pos_embed_interpolate`.
    fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> candle_core::Result<Tensor> {
        let grid_thw_vec = grid_thw.to_vec2::<u32>()?;
        let device = grid_thw.device();
        let n = self.num_grid_per_side;
        let merge = self.spatial_merge_size;
        let pos_table = self.pos_embed.embeddings();

        let mut all_pos_embeds = Vec::new();
        for grid in &grid_thw_vec {
            let t = grid[0] as usize;
            let h = grid[1] as usize;
            let w = grid[2] as usize;

            let h_idxs: Vec<f32> = (0..h)
                .map(|i| i as f32 * (n - 1) as f32 / (h.max(1) - 1).max(1) as f32)
                .collect();
            let w_idxs: Vec<f32> = (0..w)
                .map(|i| i as f32 * (n - 1) as f32 / (w.max(1) - 1).max(1) as f32)
                .collect();

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

            let e00 = pos_table.index_select(&Tensor::new(idx_00.as_slice(), device)?, 0)?;
            let e01 = pos_table.index_select(&Tensor::new(idx_01.as_slice(), device)?, 0)?;
            let e10 = pos_table.index_select(&Tensor::new(idx_10.as_slice(), device)?, 0)?;
            let e11 = pos_table.index_select(&Tensor::new(idx_11.as_slice(), device)?, 0)?;

            let pos_dtype = pos_table.dtype();
            let w00_t = Tensor::new(w_00.as_slice(), device)?
                .unsqueeze(1)?
                .to_dtype(pos_dtype)?;
            let w01_t = Tensor::new(w_01.as_slice(), device)?
                .unsqueeze(1)?
                .to_dtype(pos_dtype)?;
            let w10_t = Tensor::new(w_10.as_slice(), device)?
                .unsqueeze(1)?
                .to_dtype(pos_dtype)?;
            let w11_t = Tensor::new(w_11.as_slice(), device)?
                .unsqueeze(1)?
                .to_dtype(pos_dtype)?;

            let pos_embed = (e00.broadcast_mul(&w00_t)?
                + e01.broadcast_mul(&w01_t)?
                + e10.broadcast_mul(&w10_t)?
                + e11.broadcast_mul(&w11_t)?)?;

            for _frame in 0..t {
                let merged_h = h / merge;
                let merged_w = w / merge;
                let hidden_size = pos_embed.dim(1)?;
                let pe = pos_embed
                    .reshape((merged_h, merge, merged_w, merge, hidden_size))?
                    .permute((0, 2, 1, 3, 4))?
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
    /// Host copy of inv_freq: cos/sin tables are built on CPU, avoiding a device
    /// download on every forward. Only the final tables are uploaded.
    inv_freq: Vec<f32>,
    mrope_section: Vec<usize>,
    device: Device,
    dtype: DType,
}

impl MRoPE {
    pub fn new(cfg: &TextConfig, device: &Device, dtype: DType) -> candle_core::Result<Self> {
        let rope_dim = cfg.rope_dim(); // 64 for Argus (partial rotary)
        let theta = cfg.rope_theta(); // 1e7 for Argus
        let half_dim = rope_dim / 2; // 32
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64) as f32)
            .collect();
        let mrope_section = cfg.rope_mrope_section(); // [11,11,10] for Argus
        Ok(Self {
            inv_freq,
            mrope_section,
            device: device.clone(),
            dtype,
        })
    }

    /// Build interleaved M-RoPE `(cos, sin)` for the partial rotary dim.
    /// Positions come straight from the host (they are computed on CPU anyway).
    /// Layout matches HF's interleaved mrope: `[T,H,W,T,H,W,...]` on the min-section
    /// prefix, then the per-section remainder in T,H,W order.
    pub fn forward_positions(
        &self,
        t_pos: &[i64],
        h_pos: &[i64],
        w_pos: &[i64],
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let device = &self.device;
        let half_dim = self.inv_freq.len();
        let seq_len = t_pos.len();
        let sections = &self.mrope_section;
        let inv_freq_vec = &self.inv_freq;

        let min_section = *sections.iter().min().unwrap();
        let mut output = vec![vec![0f32; half_dim]; seq_len];
        let interleaved_end = min_section * 3;

        for s in 0..seq_len {
            let dim_pos = [t_pos[s] as f32, h_pos[s] as f32, w_pos[s] as f32];

            for i in 0..min_section {
                let base = i * 3;
                output[s][base] = dim_pos[0] * inv_freq_vec[base];
                output[s][base + 1] = dim_pos[1] * inv_freq_vec[base + 1];
                output[s][base + 2] = dim_pos[2] * inv_freq_vec[base + 2];
            }

            let mut f = interleaved_end;
            for (d, &sec_len) in sections.iter().enumerate() {
                for _ in 0..(sec_len - min_section) {
                    output[s][f] = dim_pos[d] * inv_freq_vec[f];
                    f += 1;
                }
            }
        }

        let output = Tensor::new(output, device)?;
        Ok((
            output.cos()?.to_dtype(self.dtype)?,
            output.sin()?.to_dtype(self.dtype)?,
        ))
    }
}

// ── Text: Gated RMSNorm (GDN output gate) ────────────────────────────

/// `rms_norm(x) * silu(gate)`. Computed in F32; weight used AS-IS (no +1).
struct RmsNormGated {
    weight: Tensor,
    eps: f64,
    out_dtype: DType,
}

impl RmsNormGated {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?;
        let out_dtype = weight.dtype();
        Ok(Self {
            weight,
            eps,
            out_dtype,
        })
    }

    /// `x` and `gate`: `(.., size)`.
    fn forward(&self, x: &Tensor, gate: &Tensor) -> candle_core::Result<Tensor> {
        // Fast path: fused rmsnorm(x)*weight*silu(gate) in one CUDA kernel.
        #[cfg(feature = "cuda")]
        {
            if x.device().is_cuda() && self.out_dtype == DType::BF16 {
                return crate::fused_ops::fused_rmsnorm_gated(x, gate, &self.weight, self.eps as f32);
            }
        }
        let x = x.to_dtype(DType::F32)?;
        let gate = gate.to_dtype(DType::F32)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let x_scaled = x_norm.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        let g = Activation::Silu.forward(&gate)?;
        x_scaled.mul(&g)?.to_dtype(self.out_dtype)
    }
}

// ── Text: Gated Delta Net (linear attention) ─────────────────────────

// ── Lightweight GDN profiler (CRANE_PROFILE=1) ───────────────────────
thread_local! {
    static GDN_PROF: std::cell::RefCell<Vec<(&'static str, f64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}
fn gdn_prof_on() -> bool {
    std::env::var("CRANE_PROFILE").map(|v| v == "1").unwrap_or(false)
}
/// Whether the cross-chunk scan GEMMs + state recurrence run in bf16 (tensor-core)
/// instead of F32. Decay math (cumsum/exp/decay_mask) and the intra-chunk inverse
/// stay F32. Saves ~36ms/page. Global embedding cos ~0.998 vs F32 (a few
/// L2-normalized token vectors drift), BUT a full retrieval eval over 182 real
/// queries showed NO meaningful change: top-1 identical 96%, and every top-1 flip
/// is a score tie (gap ≤0.125); overlap with the ops anchor is 170 vs 169 (equal).
/// So it is **on by default**; set `CRANE_GDN_BF16=0` to force the F32 path.
fn gdn_bf16_on() -> bool {
    std::env::var("CRANE_GDN_BF16").map(|v| v != "0").unwrap_or(true)
}
/// Whether the MLP projections run in FP8 W8A8 (cuBLASLt). Off by default —
/// opt in with `CRANE_FP8=1`. Weights are quantized once at load.
#[cfg(feature = "cuda")]
fn fp8_enabled() -> bool {
    std::env::var("CRANE_FP8").map(|v| v == "1").unwrap_or(false)
}
fn gdn_prof_add(key: &'static str, s: f64) {
    GDN_PROF.with(|m| {
        let mut v = m.borrow_mut();
        match v.iter_mut().find(|(k, _)| *k == key) {
            Some(e) => e.1 += s,
            None => v.push((key, s)),
        }
    });
}
/// Time `f` (with device syncs) under key `key` when profiling is on.
fn gdn_time<T>(
    dev: &Device,
    key: &'static str,
    f: impl FnOnce() -> candle_core::Result<T>,
) -> candle_core::Result<T> {
    if !gdn_prof_on() {
        return f();
    }
    dev.synchronize()?;
    let t = std::time::Instant::now();
    let r = f()?;
    dev.synchronize()?;
    gdn_prof_add(key, t.elapsed().as_secs_f64());
    Ok(r)
}
/// Print + reset the accumulated per-section GDN timings (call once per encode).
pub fn gdn_prof_report() {
    if !gdn_prof_on() {
        return;
    }
    GDN_PROF.with(|m| {
        let mut v = m.borrow_mut();
        if v.is_empty() {
            return;
        }
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut line = String::from("GDN_PROF (24 layers):");
        for (k, s) in v.iter() {
            line.push_str(&format!(" {}={:.0}ms", k, s * 1e3));
        }
        eprintln!("{}", line);
        v.clear();
    });
}

/// Forward-only (prefill) Gated Delta Net. No conv/recurrent state is retained.
struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv1d_w: Tensor, // (conv_dim, kernel_size)
    a_log: Tensor,    // (num_v_heads,)
    dt_bias: Tensor,  // (num_v_heads,)
    norm: RmsNormGated,
    out_proj: Linear,
    num_v_heads: usize,
    num_k_heads: usize,
    key_dim: usize,
    value_dim: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_ks: usize,
}

impl GatedDeltaNet {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let nkh = cfg.linear_num_key_heads; // 16
        let nvh = cfg.linear_num_value_heads; // 32
        let hkd = cfg.linear_key_head_dim; // 128
        let hvd = cfg.linear_value_head_dim; // 128
        let ks = cfg.linear_conv_kernel_dim; // 4
        let kd = nkh * hkd; // key_dim = 2048
        let vd = nvh * hvd; // value_dim = 4096
        let conv_dim = kd + kd + vd; // 8192

        let in_proj_qkv = linear_no_bias(h, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(h, vd, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(h, nvh, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(h, nvh, vb.pp("in_proj_a"))?;

        let conv1d_raw = vb.get((conv_dim, 1, ks), "conv1d.weight")?;
        let conv1d_w = conv1d_raw.reshape((conv_dim, ks))?;

        let a_log = vb.get(nvh, "A_log")?;
        let dt_bias = vb.get(nvh, "dt_bias")?;

        let norm = RmsNormGated::new(hvd, cfg.rms_norm_eps, vb.pp("norm"))?;
        let out_proj = linear_no_bias(vd, h, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d_w,
            a_log,
            dt_bias,
            norm,
            out_proj,
            num_v_heads: nvh,
            num_k_heads: nkh,
            key_dim: kd,
            value_dim: vd,
            head_k_dim: hkd,
            head_v_dim: hvd,
            conv_ks: ks,
        })
    }

    /// Depthwise causal conv1d + SiLU on `x: (B, T, C)` → `(B, T, C)` (zero left-pad).
    /// The vectorized window-stack (broadcast_mul + sum) saturates the GPU better
    /// than candle's grouped `conv1d` at this channel count (measured), so we keep it.
    fn apply_conv1d(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let ks = self.conv_ks;
        let x_t = x.transpose(1, 2)?.contiguous()?; // (B, C, T)

        // Fast path: fused depthwise causal conv1d + SiLU in one CUDA kernel,
        // replacing the T-slice window stack (~194MB + T launches per layer).
        #[cfg(feature = "cuda")]
        {
            if x_t.device().is_cuda() && matches!(x_t.dtype(), DType::BF16 | DType::F32) {
                let w = self.conv1d_w.to_dtype(x_t.dtype())?; // (C, ks)
                let out = crate::fused_ops::causal_conv1d_silu(&x_t, &w)?; // (B, C, T)
                return out.transpose(1, 2); // (B, T, C)
            }
        }

        // Fallback (CPU / other dtypes): vectorized window stack.
        let pad = Tensor::zeros((b, c, ks - 1), x_t.dtype(), x_t.device())?;
        let x_padded = Tensor::cat(&[&pad, &x_t], 2)?; // (B, C, T + ks - 1)

        let windows: candle_core::Result<Vec<Tensor>> =
            (0..t).map(|i| x_padded.narrow(2, i, ks)).collect();
        let stacked = Tensor::stack(&windows?, 2)?; // (B, C, T, ks)
        let w = self
            .conv1d_w
            .to_dtype(stacked.dtype())?
            .unsqueeze(0)?
            .unsqueeze(2)?; // (1, C, 1, ks)
        let out = stacked.broadcast_mul(&w)?.sum(D::Minus1)?; // (B, C, T)
        let out = Activation::Silu.forward(&out)?;
        out.transpose(1, 2) // (B, T, C)
    }

    /// Repeat each head `n` times along the head axis (torch `repeat_interleave(n, dim=2)`).
    /// `x`: `(B, T, H, D)` → `(B, T, H*n, D)`.
    fn repeat_interleave_heads(x: &Tensor, n: usize) -> candle_core::Result<Tensor> {
        if n == 1 {
            return Ok(x.clone());
        }
        let (b, t, h, d) = x.dims4()?;
        x.unsqueeze(3)?
            .expand((b, t, h, n, d))?
            .contiguous()?
            .reshape((b, t, h * n, d))
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _) = xs.dims3()?;

        // 1. Projections.
        let (qkv, z, b_proj, a_proj) = gdn_time(xs.device(), "in_proj", || {
            Ok((
                self.in_proj_qkv.forward(xs)?,
                self.in_proj_z.forward(xs)?,
                self.in_proj_b.forward(xs)?,
                self.in_proj_a.forward(xs)?,
            ))
        })?;

        // 2. Causal short conv + SiLU.
        let qkv = gdn_time(xs.device(), "conv1d", || self.apply_conv1d(&qkv))?;

        // 3. Split Q, K, V.
        let q = qkv.narrow(2, 0, self.key_dim)?;
        let k = qkv.narrow(2, self.key_dim, self.key_dim)?;
        let v = qkv.narrow(2, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((b, t, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((b, t, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((b, t, self.num_v_heads, self.head_v_dim))?;

        // 4. L2-normalise Q, K (in input dtype, HF l2norm eps 1e-6).
        let q = l2_normalize(&q)?;
        let k = l2_normalize(&k)?;

        // 5. GQA: repeat K/V-mismatched Q,K up to num_v_heads.
        let repeat = self.num_v_heads / self.num_k_heads;
        let q = Self::repeat_interleave_heads(&q, repeat)?;
        let k = Self::repeat_interleave_heads(&k, repeat)?;

        // 6. Decay & beta (F32). beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias).
        let beta = candle_nn::ops::sigmoid(&b_proj)?.to_dtype(DType::F32)?; // (B,T,H)
        let a_log_f = self.a_log.to_dtype(DType::F32)?;
        let dt_f = self.dt_bias.to_dtype(DType::F32)?;
        let a_proj_f = a_proj.to_dtype(DType::F32)?;
        let g = {
            let sp = softplus(&a_proj_f.broadcast_add(&dt_f)?)?; // (B,T,H)
            let decay = a_log_f.exp()?; // (H,)
            sp.broadcast_mul(&decay)?.neg()? // (B,T,H) in (-inf, 0]
        };
        // 7. Delta-rule scan in F32. q scaled by 1/sqrt(head_k_dim).
        let scale = 1.0 / (self.head_k_dim as f64).sqrt();
        let q = (q.to_dtype(DType::F32)? * scale)?.transpose(1, 2)?.contiguous()?; // (B,H,T,Hk)
        let k = k.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
        let v = v.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
        let g = g.transpose(1, 2)?.contiguous()?; // (B,H,T) raw log-decay
        let beta = beta.transpose(1, 2)?.contiguous()?; // (B,H,T)

        // Chunked parallel scan by default (HF's prefill path, algebraically
        // identical); the sequential recurrent form is kept behind a flag for
        // parity checks (CRANE_GDN_RECURRENT=1).
        let core_out = gdn_time(xs.device(), "scan", || {
            if std::env::var("CRANE_GDN_RECURRENT").map(|s| s == "1").unwrap_or(false) {
                self.recurrent_scan(&q, &k, &v, &g, &beta)
            } else {
                self.chunk_scan(&q, &k, &v, &g, &beta)
            }
        })?;

        // 8. Gated RMSNorm (per value head), then output projection.
        let bth = b * t * self.num_v_heads;
        let co_flat = core_out.reshape((bth, self.head_v_dim))?;
        let z_flat = z
            .reshape((b * t, self.num_v_heads, self.head_v_dim))?
            .reshape((bth, self.head_v_dim))?;
        let normed = gdn_time(xs.device(), "gated_norm", || self.norm.forward(&co_flat, &z_flat))?;
        let normed = normed.reshape((b, t, self.value_dim))?;

        gdn_time(xs.device(), "out_proj", || self.out_proj.forward(&normed))
    }

    /// Sequential recurrent gated delta rule (F32). Mirrors HF
    /// `torch_recurrent_gated_delta_rule`. Inputs (B,H,T,·)/(B,H,T), q pre-scaled,
    /// `g` raw log-decay. Returns core_out (B,T,H,Hv). Kept for parity checks.
    fn recurrent_scan(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b, _h, t, _) = q.dims4()?;
        let g_exp = g.exp()?; // (B,H,T)
        let mut state = Tensor::zeros(
            (b, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            DType::F32,
            q.device(),
        )?;
        let mut step_outputs = Vec::with_capacity(t);
        for pos in 0..t {
            let q_t = q.narrow(2, pos, 1)?.squeeze(2)?;
            let k_t = k.narrow(2, pos, 1)?.squeeze(2)?;
            let v_t = v.narrow(2, pos, 1)?.squeeze(2)?;
            let g_t = g_exp.narrow(2, pos, 1)?.squeeze(2)?;
            let beta_t = beta.narrow(2, pos, 1)?.squeeze(2)?;
            let g4d = g_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            let beta2d = beta_t.unsqueeze(D::Minus1)?;
            state = state.broadcast_mul(&g4d)?;
            let kv_pred = state.broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?.sum(D::Minus2)?;
            let delta = v_t.sub(&kv_pred)?.broadcast_mul(&beta2d)?;
            let outer = k_t.unsqueeze(D::Minus1)?.broadcast_mul(&delta.unsqueeze(D::Minus2)?)?;
            state = state.add(&outer)?;
            let out_t = state.broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?.sum(D::Minus2)?;
            step_outputs.push(out_t.unsqueeze(1)?); // (B,1,H,Hv)
        }
        Tensor::cat(&step_outputs, 1) // (B,T,H,Hv)
    }

    /// Chunked parallel gated delta rule (F32). Mirrors HF
    /// `torch_chunk_gated_delta_rule` (chunk_size=64) — algebraically identical
    /// to the recurrent form but processes chunks with batched matmuls instead of
    /// a per-token loop, cutting ~T sequential steps to ~T/64 + 64. Inputs
    /// (B,H,T,·)/(B,H,T), q pre-scaled, `g` raw log-decay. Returns (B,T,H,Hv).
    fn chunk_scan(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let dev = q.device();
        let prof = gdn_prof_on();
        let mut _pt = std::time::Instant::now();
        let (b, h, t, hk) = q.dims4()?;
        let hv = v.dim(3)?;
        let c = 64usize;
        let pad = (c - t % c) % c;
        let tp = t + pad;
        let nc = tp / c;
        let bh = b * h;
        let gg = bh * nc;

        // Pad along T so the sequence splits into whole chunks.
        let q = q.pad_with_zeros(2, 0, pad)?;
        let k = k.pad_with_zeros(2, 0, pad)?;
        let v = v.pad_with_zeros(2, 0, pad)?;
        let g = g.pad_with_zeros(2, 0, pad)?; // (B,H,Tp)
        let beta = beta.pad_with_zeros(2, 0, pad)?;

        let beta_e = beta.unsqueeze(3)?; // (B,H,Tp,1)
        let v_beta = v.broadcast_mul(&beta_e)?;
        let k_beta = k.broadcast_mul(&beta_e)?;

        // Cumulative decay within each chunk.
        let g_cs = g.reshape((b, h, nc, c))?.cumsum(3)?; // (B,H,Nc,C)
        let g_cs_g = g_cs.reshape((gg, c))?;

        // attn0 = -((k_beta @ key^T) * decay_mask) on strict-lower, where
        // decay_mask[.,i,j] = exp(g_i - g_j) for i>=j else 0. decay_mask is reused
        // below for attn_all. On CUDA the mask/exp/neg glue (7 elementwise launches
        // over (G,C,C) + a `diff` intermediate) collapses into two grid-stride kernels.
        let k_beta_g = k_beta.reshape((gg, c, hk))?;
        let key_g = k.reshape((gg, c, hk))?;
        let kk = k_beta_g.matmul(&key_g.transpose(1, 2)?.contiguous()?)?; // (G,C,C)

        #[cfg(feature = "cuda")]
        let (decay_mask, attn0) = if dev.is_cuda() {
            let dm = crate::fused_ops::gdn_decay_mask(&g_cs_g, gg, c)?;
            let a0 = crate::fused_ops::gdn_neg_lower_mul(&kk, &dm)?;
            (dm, a0)
        } else {
            decay_mask_attn0_fallback(&g_cs_g, &kk, c, dev)?
        };
        #[cfg(not(feature = "cuda"))]
        let (decay_mask, attn0) = decay_mask_attn0_fallback(&g_cs_g, &kk, c, dev)?;

        if prof {
            dev.synchronize()?;
            gdn_prof_add("scan_prep", _pt.elapsed().as_secs_f64());
            _pt = std::time::Instant::now();
        }
        // Intra-chunk inverse via forward substitution. CUDA path runs one
        // shared-memory kernel per group; otherwise a slice_assign loop.
        #[cfg(feature = "cuda")]
        let attn = if dev.is_cuda() {
            crate::fused_ops::chunk_delta_invert(&attn0, gg, c)?
        } else {
            let eye = tri_mask(c, |i, j| i == j, dev)?;
            substitute_fallback(attn0, gg, c, &eye)?
        };
        #[cfg(not(feature = "cuda"))]
        let attn = {
            let eye = tri_mask(c, |i, j| i == j, dev)?;
            substitute_fallback(attn0, gg, c, &eye)?
        };
        if prof {
            dev.synchronize()?;
            gdn_prof_add("scan_subst", _pt.elapsed().as_secs_f64());
            _pt = std::time::Instant::now();
        }

        // Cross-chunk GEMMs + the state recurrence run in `mm` (bf16 tensor-core when
        // enabled, else F32). The win (~36ms/page) comes from the bf16 recurrence
        // loop; the trade-off is per-token drift (a few L2-normalized token vectors
        // degrade, cos ~0.998 global vs F32). Decay math (exp/cumsum/decay_mask) and
        // the intra-chunk inverse stay F32. OFF by default — opt in with
        // CRANE_GDN_BF16=1. `mm == F32` reproduces the old path byte-for-byte.
        let mm = if dev.is_cuda() && gdn_bf16_on() { DType::BF16 } else { DType::F32 };

        let v_beta_g = v_beta.reshape((gg, c, hv))?;
        let attn_mm = attn.to_dtype(mm)?;
        let u = attn_mm.matmul(&v_beta_g.to_dtype(mm)?)?; // (G,C,Hv) pseudo-values, mm
        let g_exp_col = g_cs_g.exp()?.unsqueeze(2)?; // (G,C,1) F32
        let kbg = k_beta_g.broadcast_mul(&g_exp_col)?; // (G,C,Hk) F32
        let k_cumdecay = attn_mm.matmul(&kbg.to_dtype(mm)?)?; // (G,C,Hk) mm

        // Views for the cross-chunk recurrence: BH batch, iterate over Nc.
        let q_c = q.reshape((bh, nc, c, hk))?;
        let k_c = k.reshape((bh, nc, c, hk))?;
        let u_c = u.reshape((bh, nc, c, hv))?; // mm
        let g_c = g_cs.reshape((bh, nc, c))?;
        let kcd_c = k_cumdecay.reshape((bh, nc, c, hk))?; // mm

        // Batched precompute of the state-independent per-chunk quantities, so the
        // sequential recurrence only carries `state`. Decay scalings are applied in
        // F32, then cast to `mm` for the recurrence matmuls.
        let bhnc = bh * nc;
        let qk = q_c
            .reshape((bhnc, c, hk))?
            .to_dtype(mm)?
            .matmul(&k_c.reshape((bhnc, c, hk))?.transpose(1, 2)?.contiguous()?.to_dtype(mm)?)?;
        let attn_all = qk
            .to_dtype(DType::F32)?
            .broadcast_mul(&decay_mask)?
            .reshape((bh, nc, c, c))?
            .to_dtype(mm)?;
        let g_exp = g_c.exp()?; // (bh,nc,c) F32
        let qg_all = q_c.broadcast_mul(&g_exp.unsqueeze(3)?)?.to_dtype(mm)?; // q_i * exp(g_cumsum)
        let g_last = g_c.narrow(2, c - 1, 1)?; // (bh,nc,1)
        let glast_exp = g_last.squeeze(2)?.exp()?.to_dtype(mm)?; // (bh,nc)
        let coef = g_last.broadcast_sub(&g_c)?.exp()?; // (bh,nc,c) F32
        let ks_all = k_c.broadcast_mul(&coef.unsqueeze(3)?)?.to_dtype(mm)?; // k_i * exp(g_last - g_i)

        // Cross-chunk recurrence. The candle loop (cuBLAS batched matmuls) is the
        // default and fastest here; the single-launch fused kernel
        // (CRANE_GDN_FUSED_SCAN=1, F32-only) is kept for reference but is
        // latency-bound at this shape (low occupancy) and slower.
        #[cfg(feature = "cuda")]
        let core_all = if dev.is_cuda()
            && mm == DType::F32
            && std::env::var("CRANE_GDN_FUSED_SCAN").map(|v| v == "1").unwrap_or(false)
        {
            crate::fused_ops::fused_chunk_recurrence(
                &attn_all, &qg_all, &kcd_c, &ks_all, &u_c, &glast_exp, bh, nc, c, hk, hv,
            )?
        } else {
            chunk_recurrence_loop(&attn_all, &qg_all, &kcd_c, &ks_all, &u_c, &glast_exp, bh, nc, c, hk, hv, dev)?
        };
        #[cfg(not(feature = "cuda"))]
        let core_all = chunk_recurrence_loop(
            &attn_all, &qg_all, &kcd_c, &ks_all, &u_c, &glast_exp, bh, nc, c, hk, hv, dev,
        )?;

        if prof {
            dev.synchronize()?;
            gdn_prof_add("scan_chunk", _pt.elapsed().as_secs_f64());
        }
        let core = core_all.reshape((bh, tp, hv))?; // (BH,Tp,Hv)
        let core = core.narrow(1, 0, t)?; // (BH,T,Hv)
        core.reshape((b, h, t, hv))?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32) // (B,T,H,Hv), F32 for the gated norm downstream
    }
}

/// Candle fallback for the chunked delta-rule cross-chunk recurrence (mirrors
/// `fused_chunk_recurrence`). Consumes the batched precomputed tensors and
/// carries `state` sequentially. Returns core_all (BH,Nc,C,Hv).
#[allow(clippy::too_many_arguments)]
fn chunk_recurrence_loop(
    attn_all: &Tensor,
    qg_all: &Tensor,
    kcd_c: &Tensor,
    ks_all: &Tensor,
    u_c: &Tensor,
    glast_exp: &Tensor,
    bh: usize,
    nc: usize,
    c: usize,
    hk: usize,
    hv: usize,
    dev: &Device,
) -> candle_core::Result<Tensor> {
    // State carries the recurrence in the matmul dtype (bf16 or F32) so every
    // matmul below stays same-dtype; inputs are pre-cast by the caller.
    let mut state = Tensor::zeros((bh, hk, hv), attn_all.dtype(), dev)?;
    let mut outs = Vec::with_capacity(nc);
    for i in 0..nc {
        let attn_i = attn_all.narrow(1, i, 1)?.squeeze(1)?.contiguous()?; // (BH,C,C)
        let qg_i = qg_all.narrow(1, i, 1)?.squeeze(1)?.contiguous()?; // (BH,C,Hk)
        let kcd_i = kcd_c.narrow(1, i, 1)?.squeeze(1)?.contiguous()?; // (BH,C,Hk)
        let ks_i = ks_all.narrow(1, i, 1)?.squeeze(1)?.contiguous()?; // (BH,C,Hk)
        let v_i = u_c.narrow(1, i, 1)?.squeeze(1)?; // (BH,C,Hv)
        let gl_i = glast_exp.narrow(1, i, 1)?; // (BH,1)

        let v_prime = kcd_i.matmul(&state)?; // (BH,C,Hv)
        let v_new = (v_i - v_prime)?.contiguous()?;
        let attn_inter = qg_i.matmul(&state)?; // (BH,C,Hv)
        let core_i = (attn_inter + attn_i.matmul(&v_new)?)?;
        outs.push(core_i.reshape((bh, 1, c, hv))?);

        let decay_state = state.broadcast_mul(&gl_i.unsqueeze(2)?)?; // *(BH,1,1)
        let kv = ks_i.transpose(1, 2)?.contiguous()?.matmul(&v_new)?; // (BH,Hk,Hv)
        state = (decay_state + kv)?;
    }
    Tensor::cat(&outs, 1) // (BH,Nc,C,Hv)
}

/// Candle fallback for the GDN decay-mask + strict-lower `attn0` glue (the CUDA
/// path fuses this into `gdn_decay_mask` + `gdn_neg_lower_mul`). Returns
/// `(decay_mask, attn0)`, both `(G,C,C)`. Semantics identical to the original
/// candle op chain so CPU parity is preserved.
fn decay_mask_attn0_fallback(
    g_cs_g: &Tensor,
    kk: &Tensor,
    c: usize,
    dev: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let lower_incl = tri_mask(c, |i, j| i >= j, dev)?;
    let strict_lower = tri_mask(c, |i, j| i > j, dev)?;
    // mask the diff BEFORE exp so masked-out upper entries can't overflow to inf.
    let diff = g_cs_g.unsqueeze(2)?.broadcast_sub(&g_cs_g.unsqueeze(1)?)?; // (G,C,C)
    let decay_mask = diff
        .broadcast_mul(&lower_incl)?
        .exp()?
        .broadcast_mul(&lower_incl)?; // (G,C,C)
    let attn0 = kk
        .broadcast_mul(&decay_mask)?
        .neg()?
        .broadcast_mul(&strict_lower)?;
    Ok((decay_mask, attn0))
}

/// CPU/non-CUDA fallback for the chunked delta-rule intra-chunk inverse:
/// sequential forward substitution + identity via slice_assign.
fn substitute_fallback(
    mut attn: Tensor,
    gg: usize,
    c: usize,
    eye: &Tensor,
) -> candle_core::Result<Tensor> {
    for i in 1..c {
        let row = attn.narrow(1, i, 1)?.narrow(2, 0, i)?.contiguous()?; // (G,1,i)
        let sub = attn.narrow(1, 0, i)?.narrow(2, 0, i)?.contiguous()?; // (G,i,i)
        let new_row = (&row + row.matmul(&sub)?)?; // (G,1,i)
        attn = attn.slice_assign(&[0..gg, i..i + 1, 0..i], &new_row)?;
    }
    attn.broadcast_add(eye)
}

/// (C,C) F32 mask, 1.0 where `keep(row, col)` else 0.0.
fn tri_mask<F: Fn(usize, usize) -> bool>(
    c: usize,
    keep: F,
    dev: &Device,
) -> candle_core::Result<Tensor> {
    let mut v = vec![0f32; c * c];
    for i in 0..c {
        for j in 0..c {
            if keep(i, j) {
                v[i * c + j] = 1.0;
            }
        }
    }
    Tensor::from_vec(v, (c, c), dev)
}

// ── Text: Gated full attention (partial RoPE, QK-norm, GQA) ──────────

struct TextAttention {
    q_proj: Linear, // out = 2 * (num_heads * head_dim) when gated
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    q_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    attn_output_gate: bool,
}

impl TextAttention {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let gate = cfg.attn_output_gate;

        let q_proj_out = if gate { q_dim * 2 } else { q_dim };
        Ok(Self {
            q_proj: linear_no_bias(h, q_proj_out, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(h, kv_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(h, kv_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(q_dim, h, vb.pp("o_proj"))?,
            q_norm: rms_norm_1plusw(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm_1plusw(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            q_dim,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            rope_dim: cfg.rope_dim(),
            attn_output_gate: gate,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;

        // Gated q_proj: reshape (.., num_heads, 2*head_dim) then chunk → (query, gate).
        let (q, gate_signal, k, v) = if self.attn_output_gate {
            let q_full = self.q_proj.forward(xs)?; // (B, T, num_heads * head_dim * 2)
            let q_full_r = q_full.reshape((b, seq_len, self.num_heads, self.head_dim * 2))?;
            let q_half = q_full_r
                .narrow(D::Minus1, 0, self.head_dim)?
                .contiguous()?
                .reshape((b, seq_len, self.q_dim))?;
            let gate = q_full_r
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .contiguous()?
                .reshape((b, seq_len, self.q_dim))?;
            (
                q_half,
                Some(gate),
                self.k_proj.forward(xs)?,
                self.v_proj.forward(xs)?,
            )
        } else {
            (
                self.q_proj.forward(xs)?,
                None,
                self.k_proj.forward(xs)?,
                self.v_proj.forward(xs)?,
            )
        };

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;

        // Per-head QK-norm.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Partial RoPE in (B,S,H,D) via rope_thd: rotate first rope_dim dims, pass the rest.
        let (q, k) = if self.rope_dim < self.head_dim {
            let rd = self.rope_dim;
            let pass = self.head_dim - rd;
            let q_r = candle_nn::rotary_emb::rope_thd(
                &q.narrow(D::Minus1, 0, rd)?.contiguous()?,
                cos,
                sin,
            )?;
            let q_p = q.narrow(D::Minus1, rd, pass)?.contiguous()?;
            let k_r = candle_nn::rotary_emb::rope_thd(
                &k.narrow(D::Minus1, 0, rd)?.contiguous()?,
                cos,
                sin,
            )?;
            let k_p = k.narrow(D::Minus1, rd, pass)?.contiguous()?;
            (
                Tensor::cat(&[&q_r, &q_p], D::Minus1)?,
                Tensor::cat(&[&k_r, &k_p], D::Minus1)?,
            )
        } else {
            (
                candle_nn::rotary_emb::rope_thd(&q.contiguous()?, cos, sin)?,
                candle_nn::rotary_emb::rope_thd(&k.contiguous()?, cos, sin)?,
            )
        };
        let v = v.contiguous()?;

        // Causal SDPA in (B,S,H,D); flash-attn handles the mask / GQA when active.
        let attn_output =
            crate::fused_ops::attention::scaled_dot_product_attention_bshd(&q, &k, &v, true)?;
        let out = attn_output.reshape((b, seq_len, self.num_heads * self.head_dim))?;

        // Sigmoid output gate, then o_proj.
        let out = match gate_signal {
            Some(g) => out.mul(&candle_nn::ops::sigmoid(&g)?)?,
            None => out,
        };
        self.o_proj.forward(&out)
    }
}

// ── Text: MLP (SwiGLU, fused gate+up) ────────────────────────────────

/// FP8-quantized MLP weights (built once at load when CRANE_FP8=1).
#[cfg(feature = "cuda")]
struct Fp8Mlp {
    gu_w: Tensor,
    gu_scale: Tensor,
    down_w: Tensor,
    down_scale: Tensor,
}

struct TextMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
    #[cfg(feature = "cuda")]
    fp8: Option<Fp8Mlp>,
}

impl TextMLP {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = linear_no_bias(h, i, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(h, i, vb.pp("up_proj"))?;
        let gu_w = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
        let down_proj = linear_no_bias(i, h, vb.pp("down_proj"))?;

        #[cfg(feature = "cuda")]
        {
            if fp8_enabled() && gu_w.device().is_cuda() {
                let (guq, gus) = crate::fused_ops::fp8::quantize_weight_e4m3(&gu_w)?;
                let (dq, ds) =
                    crate::fused_ops::fp8::quantize_weight_e4m3(down_proj.weight())?;
                // Drop the bf16 weights (replace with 1-elem dummies) — the FP8 copies
                // are half the size, so the model footprint DROPS. Keeping both would
                // add ~5GB and thrash the allocator (slows even the GDN scan).
                let dummy = Tensor::zeros((1, 1), gu_w.dtype(), gu_w.device())?;
                return Ok(Self {
                    gate_up_proj: Linear::new(dummy.clone(), None),
                    down_proj: Linear::new(dummy, None),
                    intermediate_size: i,
                    fp8: Some(Fp8Mlp { gu_w: guq, gu_scale: gus, down_w: dq, down_scale: ds }),
                });
            }
        }

        Ok(Self {
            gate_up_proj: Linear::new(gu_w, None),
            down_proj,
            intermediate_size: i,
            #[cfg(feature = "cuda")]
            fp8: None,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            if let Some(f) = &self.fp8 {
                let gu = crate::fused_ops::fp8::fp8_linear(x, &f.gu_w, &f.gu_scale)?;
                let activated =
                    crate::fused_ops::fused_silu_mul(&gu.contiguous()?, self.intermediate_size)?;
                return crate::fused_ops::fp8::fp8_linear(&activated, &f.down_w, &f.down_scale);
            }
        }
        let gu = self.gate_up_proj.forward(x)?;
        #[cfg(feature = "cuda")]
        {
            if gu.device().is_cuda() {
                let activated =
                    crate::fused_ops::fused_silu_mul(&gu.contiguous()?, self.intermediate_size)?;
                return self.down_proj.forward(&activated);
            }
        }
        let gate = gu.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gu.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let gate = Activation::Silu.forward(&gate)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── Text: Hybrid decoder layers ──────────────────────────────────────

struct FullDecoderLayer {
    self_attn: TextAttention,
    mlp: TextMLP,
    input_ln: RmsNorm,
    post_attn_ln_weight: Tensor,
    rms_norm_eps: f64,
}

impl FullDecoderLayer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            self_attn: TextAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: TextMLP::new(cfg, vb.pp("mlp"))?,
            input_ln: rms_norm_1plusw(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attn_ln_weight: ln_weight_1plusw(
                cfg.hidden_size,
                vb.pp("post_attention_layernorm"),
            )?,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = gdn_time(xs.device(), "full_attn", || self.self_attn.forward(&xs, cos, sin))?;
        let (new_residual, h) = crate::fused_ops::fused_add_rmsnorm(
            residual,
            &xs,
            &self.post_attn_ln_weight,
            self.rms_norm_eps,
        )?;
        let h = gdn_time(h.device(), "mlp", || self.mlp.forward(&h))?;
        &new_residual + h
    }
}

struct LinearDecoderLayer {
    linear_attn: GatedDeltaNet,
    mlp: TextMLP,
    input_ln: RmsNorm,
    post_attn_ln_weight: Tensor,
    rms_norm_eps: f64,
}

impl LinearDecoderLayer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            linear_attn: GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?,
            mlp: TextMLP::new(cfg, vb.pp("mlp"))?,
            input_ln: rms_norm_1plusw(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attn_ln_weight: ln_weight_1plusw(
                cfg.hidden_size,
                vb.pp("post_attention_layernorm"),
            )?,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = self.linear_attn.forward(&xs)?;
        let (new_residual, h) = crate::fused_ops::fused_add_rmsnorm(
            residual,
            &xs,
            &self.post_attn_ln_weight,
            self.rms_norm_eps,
        )?;
        let h = gdn_time(h.device(), "mlp", || self.mlp.forward(&h))?;
        &new_residual + h
    }
}

enum HybridDecoderLayer {
    Full(FullDecoderLayer),
    Linear(LinearDecoderLayer),
}

impl HybridDecoderLayer {
    fn new(cfg: &TextConfig, layer_idx: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        if cfg.is_full_attention(layer_idx) {
            Ok(Self::Full(FullDecoderLayer::new(cfg, vb)?))
        } else {
            Ok(Self::Linear(LinearDecoderLayer::new(cfg, vb)?))
        }
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Full(l) => l.forward(xs, cos, sin),
            Self::Linear(l) => l.forward(xs),
        }
    }
}

// ── Text: Decoder ────────────────────────────────────────────────────

pub struct TextDecoder {
    embed_tokens: Embedding,
    layers: Vec<HybridDecoderLayer>,
    norm: RmsNorm,
}

impl TextDecoder {
    /// `vb` must already be prefixed to the text stack (e.g. `vb.pp("language_model")`).
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        #[cfg(feature = "cuda")]
        crate::fused_ops::ensure_mempool_cached(&vb.device());
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(HybridDecoderLayer::new(
                cfg,
                i,
                vb.pp(&format!("layers.{}", i)),
            )?);
        }
        let norm = rms_norm_1plusw(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Token-embedding lookup. `input_ids`: `(1, seq)` → `(1, seq, hidden)`.
    pub fn embed(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Full hybrid prefill: all layers (GDN + gated-full dispatch by layer_types),
    /// applying MRoPE `cos`/`sin` to the full-attention layers, then the final norm.
    ///
    /// Returns `(final_hidden (1, seq, hidden), captured)`.
    ///
    /// `capture_hidden_neg_index` replicates HF `output_hidden_states` negative
    /// indexing. The HF hidden_states tuple is `[embeddings, after_layer_0, ...,
    /// after_layer_{N-1}]` (length N+1). A value of `-5` returns the SAME tensor as
    /// `hidden_states[-5]` in Python — the output right AFTER decoder layer N-5,
    /// BEFORE the final norm. The captured tensor is returned WITHOUT the final norm.
    pub fn forward_hidden(
        &self,
        input_embeds: Tensor,
        cos: &Tensor,
        sin: &Tensor,
        capture_hidden_neg_index: Option<i32>,
    ) -> candle_core::Result<(Tensor, Option<Tensor>)> {
        let total_states = self.layers.len() as i32 + 1; // embeddings + N layer outputs
        // Python index into the hidden_states tuple (0 = embeddings, i+1 = after layer i).
        let capture_py = capture_hidden_neg_index.map(|neg| {
            let idx = if neg < 0 { total_states + neg } else { neg };
            idx
        });

        let mut captured: Option<Tensor> = None;
        if capture_py == Some(0) {
            captured = Some(input_embeds.clone());
        }

        let mut h = input_embeds;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, cos, sin)?;
            if capture_py == Some(i as i32 + 1) {
                captured = Some(h.clone());
            }
        }

        let final_hidden = self.norm.forward(&h)?;
        Ok((final_hidden, captured))
    }
}
