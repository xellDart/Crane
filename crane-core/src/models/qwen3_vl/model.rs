//! Qwen3-VL backbone for ColQwen3 multi-vector embeddings.
//!
//! Forward-only path: vision encoder + text decoder hidden states (no lm_head,
//! no autoregressive decode, no KV cache, no GDN/hybrid, no batch infra).
//!
//! Architecture:
//!   - Vision: Conv3D patch embed + 24-layer ViT + spatial merge + DeepStack
//!   - Text:   36-layer Qwen3 decoder with M-RoPE, QK-norm, GQA

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{
    self, linear, linear_no_bias, Embedding, LayerNorm, Linear, RmsNorm, VarBuilder,
};
use serde::Deserialize;

pub use super::config::PreprocessorConfig;

fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<RmsNorm> {
    let w = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?;
    Ok(RmsNorm::new(w, eps))
}

// ── Configs ───────────────────────────────────────────────────────────

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
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

impl TextConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().map(|p| p.rope_theta))
            .unwrap_or(1_000_000.0)
    }

    /// Number of head dimensions that receive rotary position embeddings.
    pub fn rope_dim(&self) -> usize {
        let factor = self
            .rope_parameters
            .as_ref()
            .map(|p| p.partial_rotary_factor)
            .unwrap_or(1.0);
        ((self.head_dim as f64 * factor).round() as usize).max(2)
    }

    /// mrope_section — from rope_parameters or rope_scaling.
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
        vec![24, 20, 20]
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

fn default_partial_rotary_factor() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub mrope_interleaved: bool,
}

fn default_head_dim() -> usize {
    128
}

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

        let out = out.contiguous()?.reshape((seq_len, self.num_heads * self.head_dim))?;
        self.proj.forward(&out)
    }

    fn apply_vision_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
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
        let x = self.fc1.forward(x)?.gelu_erf()?; // gelu_pytorch_tanh
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
    fn new(vb: VarBuilder, cfg: &VisionConfig, post_shuffle_norm: bool) -> candle_core::Result<Self> {
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

    /// Bilinear interpolation of position embeddings for variable resolution.
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
    /// Host copy: cos/sin tables are built on CPU, so keeping inv_freq on the
    /// host avoids a device→host download on every forward.
    inv_freq: Vec<f32>,
    mrope_section: Vec<usize>,
    device: Device,
    dtype: DType,
}

impl MRoPE {
    pub fn new(cfg: &TextConfig, device: &Device, dtype: DType) -> candle_core::Result<Self> {
        let rope_dim = cfg.rope_dim();
        let theta = cfg.rope_theta();
        let half_dim = rope_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64) as f32)
            .collect();
        let mrope_section = cfg.rope_mrope_section();
        Ok(Self { inv_freq, mrope_section, device: device.clone(), dtype })
    }

    /// Positions come straight from the host (they are computed on CPU by the
    /// callers anyway): no device round-trip. Only the final cos/sin tables
    /// are uploaded. Bit-identical to the old Tensor-based path.
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
        Ok((output.cos()?.to_dtype(self.dtype)?, output.sin()?.to_dtype(self.dtype)?))
    }
}

// ── Text: Attention (forward-only, no KV cache) ──────────────────────

struct TextAttention {
    qkv_proj: Linear, // fused QKV
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    q_dim: usize,
    kv_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
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

        // Fused QKV: merge weights into single matmul.
        let qkv_w = Tensor::cat(&[q_proj.weight(), k_proj.weight(), v_proj.weight()], 0)?;
        let qkv_proj = Linear::new(qkv_w, None);

        Ok(Self {
            qkv_proj,
            o_proj: linear_no_bias(q_dim, h, vb.pp("o_proj"))?,
            q_norm: rms_norm(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            q_dim,
            kv_dim,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            rope_dim: cfg.rope_dim(),
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;

        // Fused Q/K/V projection
        let qkv = self.qkv_proj.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, self.q_dim)?;
        let k = qkv.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
        let v = qkv.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;

        // QK-norm (per head)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Everything stays in (b, seq, heads, dim) — flash-attn's native layout.
        // rope_thd applies the same non-interleaved rotation as rope, indexed
        // for this layout, so no transpose round-trips are needed anywhere.
        let (q, k) = if self.rope_dim < self.head_dim {
            let rd = self.rope_dim;
            let pass = self.head_dim - rd;
            let q_r = candle_nn::rotary_emb::rope_thd(&q.narrow(D::Minus1, 0, rd)?.contiguous()?, cos, sin)?;
            let q_p = q.narrow(D::Minus1, rd, pass)?.contiguous()?;
            let k_r = candle_nn::rotary_emb::rope_thd(&k.narrow(D::Minus1, 0, rd)?.contiguous()?, cos, sin)?;
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

        // SDPA in (b, seq, heads, dim); causal handled inside flash-attn.
        let attn_output =
            crate::fused_ops::attention::scaled_dot_product_attention_bshd(&q, &k, &v, true)?;

        let out = attn_output.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

// ── Text: MLP (SwiGLU, fused gate+up) ───────────────────────────────

struct TextMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl TextMLP {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = linear_no_bias(h, i, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(h, i, vb.pp("up_proj"))?;
        let gu_w = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
        Ok(Self {
            gate_up_proj: Linear::new(gu_w, None),
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
            intermediate_size: i,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gu = self.gate_up_proj.forward(x)?;
        #[cfg(feature = "cuda")]
        {
            if gu.device().is_cuda() {
                let activated = crate::fused_ops::fused_silu_mul(&gu.contiguous()?, self.intermediate_size)?;
                return self.down_proj.forward(&activated);
            }
        }
        let gate = gu.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gu.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let gate = candle_nn::Activation::Silu.forward(&gate)?;
        self.down_proj.forward(&(gate * up)?)
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

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin)?;

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
    let num_vis = vis_feat.dim(0)?;
    let positions: Vec<usize> = mask_vec
        .iter()
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
    device: Device,
    dtype: DType,
    hidden_size: usize,
}

impl TextDecoder {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        #[cfg(feature = "cuda")]
        crate::fused_ops::ensure_mempool_cached(&vb.device());
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(TextDecoderLayer::new(cfg, vb.pp(&format!("layers.{}", i)))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
        })
    }

    pub fn embed(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        self.embed_tokens.forward(ids)
    }

    /// Forward pass returning final hidden states `(B, S, H)` after the last RMSNorm.
    /// No mask is built — flash-attn-2 handles causal masking internally for prefill.
    /// `vision_mask` is host data: the scatter needs CPU positions, so taking a
    /// device tensor here forced a download per forward.
    pub fn forward_hidden(
        &self,
        xs: Tensor,
        cos: &Tensor,
        sin: &Tensor,
        deepstack_features: Option<&[Tensor]>,
        vision_mask: Option<&[f32]>,
    ) -> candle_core::Result<Tensor> {
        let (_b, seq_len, _) = xs.dims3()?;

        // Pre-compute scattered deepstack features once
        let scattered_ds = if let (Some(ds), Some(mask_vec)) = (deepstack_features, vision_mask) {
            let mut scattered = Vec::with_capacity(ds.len());
            for feat in ds.iter() {
                let padded = scatter_vision_features(
                    feat,
                    mask_vec,
                    seq_len,
                    self.hidden_size,
                    self.dtype,
                    &self.device,
                )?;
                scattered.push(padded);
            }
            Some(scattered)
        } else {
            None
        };

        let mut h = xs;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, cos, sin)?;
            if let Some(ref scattered) = scattered_ds {
                if i < scattered.len() {
                    h = (h + &scattered[i])?;
                }
            }
        }

        self.norm.forward(&h)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

