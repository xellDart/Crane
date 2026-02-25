use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{self, Activation, Embedding, VarBuilder};
use candle_transformers::models::clip::vision_model::{ClipVisionConfig, ClipVisionTransformer};
use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

// ── Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct LlavaQwenConfig {
    pub image_token_index: usize,
    #[serde(default = "default_image_seq_length")]
    pub image_seq_length: usize,
    pub projector_hidden_act: String,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub vision_feature_layer: isize,
    #[serde(default = "default_select_strategy")]
    pub vision_feature_select_strategy: String,
}

fn default_image_seq_length() -> usize {
    576
}
fn default_select_strategy() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub hidden_act: HiddenAct,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    #[default]
    Silu,
    Gelu,
    Relu,
}

impl From<HiddenAct> for Activation {
    fn from(act: HiddenAct) -> Self {
        match act {
            HiddenAct::Silu => Activation::Silu,
            HiddenAct::Gelu => Activation::Gelu,
            HiddenAct::Relu => Activation::Relu,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    pub projection_dim: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
}

fn default_num_channels() -> usize {
    3
}

impl VisionConfig {
    fn to_clip_config(&self) -> ClipVisionConfig {
        use candle_transformers::models::clip::text_model::Activation as ClipActivation;
        ClipVisionConfig {
            embed_dim: self.hidden_size,
            activation: ClipActivation::QuickGelu,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            projection_dim: self.projection_dim,
            num_channels: self.num_channels,
            image_size: self.image_size,
            patch_size: self.patch_size,
        }
    }
}

// ── Qwen2 decoder (with embed access) ───────────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &TextConfig, dev: &Device) -> candle_core::Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Qwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen2Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = h / nh;
        Ok(Self {
            q_proj: linear(h, nh * hd, vb.pp("q_proj"))?,
            k_proj: linear(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: linear(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(nh * hd, h, vb.pp("o_proj"))?,
            num_heads: nh,
            num_kv_heads: nkv,
            num_kv_groups: nh / nkv,
            head_dim: hd,
            hidden_size: h,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> candle_core::Result<Tensor> {
        let (b, q_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?.reshape((b, q_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(xs)?.reshape((b, q_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(xs)?.reshape((b, q_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        let k = candle_transformers::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)?.transpose(1, 2)?.reshape((b, q_len, self.hidden_size))?.apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
struct Qwen2Layer {
    self_attn: Qwen2Attention,
    mlp_gate: Linear,
    mlp_up: Linear,
    mlp_down: Linear,
    act: Activation,
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
}

impl Qwen2Layer {
    fn new(rotary: Arc<RotaryEmbedding>, cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            self_attn: Qwen2Attention::new(rotary, cfg, vb.pp("self_attn"))?,
            mlp_gate: linear_no_bias(h, i, vb.pp("mlp.gate_proj"))?,
            mlp_up: linear_no_bias(h, i, vb.pp("mlp.up_proj"))?,
            mlp_down: linear_no_bias(i, h, vb.pp("mlp.down_proj"))?,
            act: cfg.hidden_act.clone().into(),
            input_ln: RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attn_ln: RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = self.self_attn.forward(&xs, mask, offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let h = xs.apply(&self.post_attn_ln)?;
        let gate = h.apply(&self.mlp_gate)?.apply(&self.act)?;
        let up = h.apply(&self.mlp_up)?;
        let mlp = (gate * up)?.apply(&self.mlp_down)?;
        residual + mlp
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

struct Qwen2Decoder {
    embed_tokens: Embedding,
    layers: Vec<Qwen2Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl Qwen2Decoder {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen2Layer::new(rotary.clone(), cfg, vb_l.pp(i))?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn embed(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> candle_core::Result<Tensor> {
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
        mask.expand((b, 1, tgt, tgt + offset))?.to_dtype(self.dtype)
    }

    fn forward_embeds(&mut self, xs: Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(self.causal_mask(b, seq_len, offset)?)
        };
        let mut h = xs;
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, mask.as_ref(), offset)?;
        }
        let h = h.apply(&self.norm)?;
        h.narrow(1, seq_len - 1, 1)?.apply(&self.lm_head)
    }

    fn forward_ids(&mut self, input_ids: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.forward_embeds(embeds, offset)
    }

    fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

// ── CLIP Vision Tower ────────────────────────────────────────────────

struct VisionTower {
    model: ClipVisionTransformer,
    select_layer: isize,
    select_feature: String,
}

impl VisionTower {
    fn new(vb: VarBuilder, cfg: &VisionConfig, select_layer: isize, select_feature: &str) -> candle_core::Result<Self> {
        let clip_cfg = cfg.to_clip_config();
        let model = ClipVisionTransformer::new(vb, &clip_cfg)?;
        Ok(Self {
            model,
            select_layer,
            select_feature: select_feature.to_string(),
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let hidden_states = self.model.output_hidden_states(x)?;
        let idx = (hidden_states.len() as isize + self.select_layer) as usize;
        let features = hidden_states[idx].clone();
        if self.select_feature == "cls_patch" || self.select_feature == "full" {
            Ok(features)
        } else {
            // "default" / "patch" → skip CLS token
            features.i((.., 1..))
        }
    }
}

// ── MM Projector ─────────────────────────────────────────────────────

struct MMProjector {
    linear1: Linear,
    linear2: Linear,
}

impl MMProjector {
    fn new(vision_hidden: usize, text_hidden: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            linear1: linear(vision_hidden, text_hidden, vb.pp("linear_1"))?,
            linear2: linear(text_hidden, text_hidden, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.gelu()?;
        self.linear2.forward(&x)
    }
}

// ── LLaVA-Qwen2 Model ───────────────────────────────────────────────

pub struct LlavaQwen {
    vision_tower: VisionTower,
    mm_projector: MMProjector,
    decoder: Qwen2Decoder,
    tokenizer: Tokenizer,
    config: LlavaQwenConfig,
    pub device: Device,
    dtype: DType,
}

pub struct LlavaResult {
    pub text: String,
    pub tokens_generated: usize,
    pub duration_secs: f32,
}

impl LlavaQwen {
    pub fn from_local(path: impl AsRef<Path>, cpu: bool, bf16: bool) -> Result<Self> {
        let device = if cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
        let dtype = if bf16 && device.is_cuda() { DType::BF16 } else { DType::F32 };

        let base = path.as_ref();
        let config: LlavaQwenConfig =
            serde_json::from_str(&std::fs::read_to_string(base.join("config.json"))?)?;
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

        println!("Loading vision tower...");
        let vision_tower = VisionTower::new(
            vb.pp("vision_tower.vision_model"),
            &config.vision_config,
            config.vision_feature_layer,
            &config.vision_feature_select_strategy,
        )?;

        println!("Loading MM projector...");
        let mm_projector = MMProjector::new(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("multi_modal_projector"),
        )?;

        println!("Loading language model...");
        let decoder = Qwen2Decoder::new(&config.text_config, vb.pp("language_model"))?;

        println!("Model loaded!");

        Ok(Self {
            vision_tower,
            mm_projector,
            decoder,
            tokenizer,
            config,
            device,
            dtype,
        })
    }

    fn encode_image(&self, pixel_values: &Tensor) -> candle_core::Result<Tensor> {
        let features = self.vision_tower.forward(pixel_values)?;
        self.mm_projector.forward(&features)
    }

    fn prepare_multimodal_input(
        &self,
        input_ids: &[u32],
        image_features_list: &[Tensor],
    ) -> candle_core::Result<Tensor> {
        let token_idx = self.config.image_token_index as u32;
        let seq_len = self.config.image_seq_length;

        // Split input_ids into segments: text and image-placeholder runs.
        // A consecutive run of image tokens is split into chunks of image_seq_length,
        // each chunk corresponding to one image.
        let mut parts: Vec<Tensor> = Vec::new();
        let mut current_text: Vec<u32> = Vec::new();
        let mut img_run_len = 0usize;
        let mut img_idx = 0usize;

        for &id in input_ids {
            if id == token_idx {
                if img_run_len == 0 {
                    // Flush pending text before starting image run
                    if !current_text.is_empty() {
                        let ids = Tensor::new(current_text.as_slice(), &self.device)?;
                        parts.push(self.decoder.embed(&ids)?);
                        current_text.clear();
                    }
                }
                img_run_len += 1;
                // Every image_seq_length tokens → one image
                if img_run_len == seq_len {
                    if img_idx < image_features_list.len() {
                        let feats = if image_features_list[img_idx].dims().len() == 3 {
                            image_features_list[img_idx].squeeze(0)?
                        } else {
                            image_features_list[img_idx].clone()
                        };
                        parts.push(feats);
                    }
                    img_idx += 1;
                    img_run_len = 0;
                }
            } else {
                img_run_len = 0;
                current_text.push(id);
            }
        }
        // Flush trailing text
        if !current_text.is_empty() {
            let ids = Tensor::new(current_text.as_slice(), &self.device)?;
            parts.push(self.decoder.embed(&ids)?);
        }

        let combined = Tensor::cat(&parts, 0)?;
        combined.unsqueeze(0) // [1, total_seq_len, hidden_dim]
    }

    pub fn recognize<P: AsRef<Path>>(
        &mut self,
        image_paths: &[P],
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<LlavaResult> {
        let start = Instant::now();

        // Load, preprocess and encode each image
        let mut image_features_list = Vec::with_capacity(image_paths.len());
        for path in image_paths {
            let pixel_values = load_and_preprocess_image(
                path.as_ref(),
                self.config.vision_config.image_size,
                &self.device,
                self.dtype,
            )?;
            image_features_list.push(self.encode_image(&pixel_values)?);
        }

        // Tokenize prompt
        let input_ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| E::msg(format!("Tokenizer: {}", e)))?
            .get_ids()
            .to_vec();

        // Prepare multimodal input
        self.decoder.clear_kv_cache();
        let input_embeds = self.prepare_multimodal_input(&input_ids, &image_features_list)?;

        // First forward pass with images
        let logits = self.decoder.forward_embeds(input_embeds, 0)?;
        let next_token = logits.flatten_all()?.argmax(D::Minus1)?.to_dtype(DType::U32)?.to_scalar::<u32>()?;

        let eos_id = self.tokenizer.token_to_id("<|im_end|>").unwrap_or(
            self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(151645),
        );

        let mut generated = vec![next_token];
        if next_token == eos_id {
            let text = self.tokenizer.decode(&generated, true).map_err(E::msg)?;
            return Ok(LlavaResult {
                text,
                tokens_generated: 1,
                duration_secs: start.elapsed().as_secs_f32(),
            });
        }

        // Calculate seq offset: non-image text tokens + N * image_seq_length
        let total_first_len = {
            let non_image_tokens = input_ids.iter().filter(|&&id| id != self.config.image_token_index as u32).count();
            non_image_tokens + image_paths.len() * self.config.image_seq_length
        };
        let mut offset = total_first_len;

        // Autoregressive generation
        for _ in 1..max_new_tokens {
            let token_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.decoder.forward_ids(&token_tensor, offset)?;
            let next = logits.flatten_all()?.argmax(D::Minus1)?.to_dtype(DType::U32)?.to_scalar::<u32>()?;
            generated.push(next);
            if next == eos_id {
                break;
            }
            offset += 1;
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(E::msg)?
            .trim()
            .to_string();

        Ok(LlavaResult {
            text,
            tokens_generated: generated.len(),
            duration_secs: start.elapsed().as_secs_f32(),
        })
    }

    pub fn recognize_stream<P: AsRef<Path>, F>(
        &mut self,
        image_paths: &[P],
        prompt: &str,
        max_new_tokens: usize,
        mut callback: F,
    ) -> Result<LlavaResult>
    where
        F: FnMut(&str),
    {
        let start = Instant::now();

        // Load, preprocess and encode each image
        let mut image_features_list = Vec::with_capacity(image_paths.len());
        for path in image_paths {
            let pixel_values = load_and_preprocess_image(
                path.as_ref(),
                self.config.vision_config.image_size,
                &self.device,
                self.dtype,
            )?;
            image_features_list.push(self.encode_image(&pixel_values)?);
        }

        let input_ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| E::msg(format!("Tokenizer: {}", e)))?
            .get_ids()
            .to_vec();

        self.decoder.clear_kv_cache();
        let input_embeds = self.prepare_multimodal_input(&input_ids, &image_features_list)?;

        let logits = self.decoder.forward_embeds(input_embeds, 0)?;
        let mut next_token = logits.flatten_all()?.argmax(D::Minus1)?.to_dtype(DType::U32)?.to_scalar::<u32>()?;

        let eos_id = self.tokenizer.token_to_id("<|im_end|>").unwrap_or(
            self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(151645),
        );

        let mut generated = Vec::new();
        let total_first_len = {
            let non_image_tokens = input_ids.iter().filter(|&&id| id != self.config.image_token_index as u32).count();
            non_image_tokens + image_paths.len() * self.config.image_seq_length
        };
        let mut offset = total_first_len;
        let mut text = String::new();

        for _ in 0..max_new_tokens {
            generated.push(next_token);
            if next_token == eos_id {
                break;
            }
            if let Ok(s) = self.tokenizer.decode(&[next_token], false) {
                callback(&s);
                text.push_str(&s);
            }

            let token_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.decoder.forward_ids(&token_tensor, offset)?;
            next_token = logits.flatten_all()?.argmax(D::Minus1)?.to_dtype(DType::U32)?.to_scalar::<u32>()?;
            offset += 1;
        }

        Ok(LlavaResult {
            text,
            tokens_generated: generated.len(),
            duration_secs: start.elapsed().as_secs_f32(),
        })
    }
}

// ── Image loading ────────────────────────────────────────────────────

fn load_and_preprocess_image(
    path: &Path,
    image_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::open(path)?.to_rgb8();
    let (orig_w, orig_h) = (img.width(), img.height());
    let target = image_size as u32;

    // CLIP preprocessing: resize shortest edge to target, then center crop
    let (new_w, new_h) = if orig_w < orig_h {
        (target, (orig_h as f32 * target as f32 / orig_w as f32).round() as u32)
    } else {
        ((orig_w as f32 * target as f32 / orig_h as f32).round() as u32, target)
    };

    let resized = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::CatmullRom);

    // Center crop
    let crop_x = (new_w - target) / 2;
    let crop_y = (new_h - target) / 2;
    let cropped = image::imageops::crop_imm(&resized, crop_x, crop_y, target, target).to_image();

    // CLIP normalization
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.26130258, 0.27577711];

    let s = image_size;
    let mut data = Vec::with_capacity(3 * s * s);
    for c in 0..3 {
        for y in 0..s {
            for x in 0..s {
                let pixel = cropped.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
                data.push((pixel - mean[c]) / std[c]);
            }
        }
    }

    let tensor = Tensor::from_vec(data, (1, 3, s, s), device)?.to_dtype(dtype)?;
    Ok(tensor)
}
