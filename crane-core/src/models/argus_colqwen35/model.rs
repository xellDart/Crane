//! Argus-Colqwen3.5-9B multi-vector retriever.
//!
//! Pipeline mirrors `colqwen3_emb` (pipelined encode, on-device patchify,
//! host-side M-RoPE, MaxSim scoring) but sits on the Qwen3.5-VL *hybrid*
//! backbone (`super::super::qwen3_5_vl`) and adds the Argus region-level
//! Mixture-of-Experts head before the retrieval projection.
//!
//! Doc encode: backbone → capture hidden@router_layer + final hidden → region
//! MoE fusion over the vision-token grid → custom_text_proj → L2 norm →
//! (optionally) mask non-image tokens.
//! Query encode: backbone → custom_text_proj → L2 norm (no MoE).

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Module, Shape, Tensor, D};
use candle_nn::{layer_norm, linear, linear_no_bias, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

use super::super::qwen3_5_vl::{MRoPE, TextConfig, TextDecoder, VisionConfig, VisionModel};

// ── Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ArgusConfig {
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    pub image_token_id: u32,
    #[serde(default)]
    pub video_token_id: u32,
    #[serde(default)]
    pub vision_start_token_id: u32,
    #[serde(default)]
    pub vision_end_token_id: u32,

    #[serde(default = "default_retrieval_dim")]
    pub retrieval_dim: usize,
    #[serde(default = "default_num_specialists")]
    pub num_specialists: usize,
    #[serde(default = "default_top_k")]
    pub top_k_experts: usize,
    #[serde(default = "default_region_size")]
    pub region_size: usize,
    #[serde(default = "default_router_layer_index")]
    pub router_layer_index: i32,
    #[serde(default = "default_router_temperature")]
    pub router_temperature: f64,
    #[serde(default)]
    pub router_noise_std: f64,
    #[serde(default = "default_true")]
    pub mask_non_image_embeddings: bool,
    #[serde(default)]
    pub shared_gate_init: f64,
    #[serde(default)]
    pub specialist_gate_init: f64,
}

fn default_retrieval_dim() -> usize { 1024 }
fn default_num_specialists() -> usize { 4 }
fn default_top_k() -> usize { 2 }
fn default_region_size() -> usize { 4 }
fn default_router_layer_index() -> i32 { -5 }
fn default_router_temperature() -> f64 { 0.8 }
fn default_true() -> bool { true }

/// `processor_config.json` → `image_processor` sub-object (Argus ships no
/// standalone `preprocessor_config.json`).
#[derive(Debug, Clone, Deserialize)]
struct ProcessorConfig {
    image_processor: ImageProcessorConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct ImageProcessorConfig {
    image_mean: [f64; 3],
    image_std: [f64; 3],
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    size: SizeConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct SizeConfig {
    shortest_edge: usize,
    longest_edge: usize,
}

// ── Argus MoE head ───────────────────────────────────────────────────

/// A `Sequential(LayerNorm, Linear, GELU, Linear)` expert. GELU is the exact
/// (erf) variant to match `nn.GELU()` in the reference.
struct MlpExpert {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl MlpExpert {
    /// `vb` points at the `net` Sequential (indices 0=LN, 1=Linear, 3=Linear).
    fn new(vb: VarBuilder, hidden: usize, expansion: usize) -> Result<Self> {
        let norm = layer_norm(hidden, 1e-5, vb.pp("0"))?;
        let fc1 = linear(hidden, hidden * expansion, vb.pp("1"))?;
        let fc2 = linear(hidden * expansion, hidden, vb.pp("3"))?;
        Ok(Self { norm, fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.norm.forward(x)?;
        let h = self.fc1.forward(&h)?;
        let h = h.gelu_erf()?;
        self.fc2.forward(&h)
    }
}

/// `Sequential(LayerNorm, Linear(h,h), GELU, Linear(h, num_specialists))`.
struct RegionRouter {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl RegionRouter {
    fn new(vb: VarBuilder, hidden: usize, num_specialists: usize) -> Result<Self> {
        let norm = layer_norm(hidden, 1e-5, vb.pp("0"))?;
        let fc1 = linear(hidden, hidden, vb.pp("1"))?;
        let fc2 = linear(hidden, num_specialists, vb.pp("3"))?;
        Ok(Self { norm, fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.norm.forward(x)?;
        let h = self.fc1.forward(&h)?;
        let h = h.gelu_erf()?;
        self.fc2.forward(&h)
    }
}

struct ArgusHead {
    custom_text_proj: Linear,
    shared_expert: MlpExpert,
    latent_experts: Vec<MlpExpert>,
    region_router: RegionRouter,
    region_coord_proj: Linear, // Linear(4, hidden, bias=False)
    gate_shared: f64,          // raw scalar (sigmoid applied at use)
    gate_specialist: f64,
}

impl ArgusHead {
    fn new(vb: VarBuilder, cfg: &ArgusConfig, hidden: usize) -> Result<Self> {
        let custom_text_proj = linear(hidden, cfg.retrieval_dim, vb.pp("custom_text_proj"))?;
        let shared_expert = MlpExpert::new(vb.pp("shared_expert").pp("net"), hidden, 4)?;
        let mut latent_experts = Vec::with_capacity(cfg.num_specialists);
        for i in 0..cfg.num_specialists {
            latent_experts.push(MlpExpert::new(
                vb.pp("latent_experts").pp(i).pp("net"),
                hidden,
                2,
            )?);
        }
        let region_router = RegionRouter::new(vb.pp("region_router"), hidden, cfg.num_specialists)?;
        let region_coord_proj = linear_no_bias(4, hidden, vb.pp("region_coord_proj"))?;

        // gate_scalars are 0-dim fp32 params.
        let gate_shared = scalar_param(&vb, "gate_scalars.shared")?;
        let gate_specialist = scalar_param(&vb, "gate_scalars.specialist")?;

        Ok(Self {
            custom_text_proj,
            shared_expert,
            latent_experts,
            region_router,
            region_coord_proj,
            gate_shared,
            gate_specialist,
        })
    }
}

fn scalar_param(vb: &VarBuilder, name: &str) -> Result<f64> {
    // gate scalars are rank-0 tensors; read as a single f32.
    let t = vb.get((), name).or_else(|_| vb.get(1, name))?;
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(v.first().copied().unwrap_or(0.0) as f64)
}

// ── Model ────────────────────────────────────────────────────────────

pub struct ArgusColqwen35Emb {
    vision: VisionModel,
    decoder: TextDecoder,
    mrope: MRoPE,
    head: ArgusHead,
    tokenizer: Tokenizer,
    config: ArgusConfig,
    proc: ImageProcessorConfig,
    dims: usize,
    pub device: Device,
    dtype: DType,
    img_mean: Tensor,
    img_std: Tensor,
}

impl ArgusColqwen35Emb {
    pub fn from_local(path: impl AsRef<Path>, cpu: bool, bf16: bool) -> Result<Self> {
        let device = if cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
        let dtype = if bf16 && device.is_cuda() { DType::BF16 } else { DType::F32 };

        let base = path.as_ref();
        let config: ArgusConfig =
            serde_json::from_str(&std::fs::read_to_string(base.join("config.json"))?)?;
        let proc_cfg: ProcessorConfig =
            serde_json::from_str(&std::fs::read_to_string(base.join("processor_config.json"))?)?;
        let proc = proc_cfg.image_processor;
        let tokenizer = Tokenizer::from_file(base.join("tokenizer.json")).map_err(E::msg)?;

        let safetensors: Vec<std::path::PathBuf> = std::fs::read_dir(base)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        if safetensors.is_empty() {
            anyhow::bail!("No safetensors files found in {}", base.display());
        }
        let refs: Vec<&std::path::Path> = safetensors.iter().map(|p| p.as_path()).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&refs, dtype, &device)? };

        println!("Loading Qwen3.5-VL vision encoder...");
        let vision = VisionModel::new(vb.pp("visual"), &config.vision_config)?;

        println!("Loading Qwen3.5 hybrid text decoder (embedding mode)...");
        let decoder = TextDecoder::new(&config.text_config, vb.pp("language_model"))?;

        println!("Loading Argus MoE retrieval head...");
        let head = ArgusHead::new(vb.clone(), &config, config.text_config.hidden_size)?;

        println!("Initializing M-RoPE...");
        let mrope = MRoPE::new(&config.text_config, &device, dtype)?;

        let img_mean = Tensor::new(
            &[proc.image_mean[0] as f32, proc.image_mean[1] as f32, proc.image_mean[2] as f32],
            &device,
        )?
        .reshape((3, 1, 1))?;
        let img_std = Tensor::new(
            &[proc.image_std[0] as f32, proc.image_std[1] as f32, proc.image_std[2] as f32],
            &device,
        )?
        .reshape((3, 1, 1))?;

        let dims = config.retrieval_dim;
        println!("Argus-Colqwen3.5 model loaded! retrieval_dim={}", dims);

        Ok(Self {
            vision,
            decoder,
            mrope,
            head,
            tokenizer,
            config,
            proc,
            dims,
            device,
            dtype,
            img_mean,
            img_std,
        })
    }

    /// Matryoshka-style truncation (Argus is trained at retrieval_dim, so this
    /// is off by default; kept for API parity with ColQwen3Emb).
    pub fn set_dims(&mut self, dims: usize) {
        assert!(
            dims <= self.config.retrieval_dim,
            "target dims ({}) > retrieval_dim ({})",
            dims,
            self.config.retrieval_dim
        );
        self.dims = dims;
    }

    // ── Image encoding ────────────────────────────────────────────────

    pub fn encode_images<P: AsRef<Path> + Sync>(&mut self, image_paths: &[P]) -> Result<Vec<Tensor>> {
        let params = self.preproc_params();
        let jobs: Vec<std::path::PathBuf> =
            image_paths.iter().map(|p| p.as_ref().to_path_buf()).collect();
        self.encode_pipelined(jobs, move |path| {
            let img = image::open(&path)?.to_rgb8();
            resize_image_cpu(img, params)
        })
    }

    pub fn encode_images_from_bytes(&mut self, images: &[&[u8]]) -> Result<Vec<Tensor>> {
        let params = self.preproc_params();
        let jobs: Vec<Vec<u8>> = images.iter().map(|b| b.to_vec()).collect();
        self.encode_pipelined(jobs, move |bytes| {
            let img = image::load_from_memory(&bytes)
                .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?
                .to_rgb8();
            resize_image_cpu(img, params)
        })
    }

    fn encode_pipelined<T, F>(&mut self, jobs: Vec<T>, decode: F) -> Result<Vec<Tensor>>
    where
        T: Send + 'static,
        F: Fn(T) -> Result<DecodedImage> + Send + Sync + 'static,
    {
        let n = jobs.len();
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Result<DecodedImage>)>(8);
        let decode = std::sync::Arc::new(decode);
        for (i, job) in jobs.into_iter().enumerate() {
            let tx = tx.clone();
            let decode = decode.clone();
            rayon::spawn(move || {
                let _ = tx.send((i, decode(job)));
            });
        }
        drop(tx);

        let mut slots: Vec<Option<Tensor>> = std::iter::repeat_with(|| None).take(n).collect();
        while let Ok((i, dec)) = rx.recv() {
            let emb = self.encode_one_decoded(dec?)?;
            slots[i] = Some(emb);
        }

        slots
            .into_iter()
            .map(|s| s.ok_or_else(|| E::msg("image decode task vanished")))
            .collect()
    }

    /// GPU stage for one image: patchify → vision → decoder (capturing the
    /// router-layer hidden) → Argus MoE fusion → project + normalize + mask.
    fn encode_one_decoded(&mut self, dec: DecodedImage) -> Result<Tensor> {
        let merge = self.config.vision_config.spatial_merge_size as u32;

        let (pixel_values, grid_tensor, grid) = self.patchify_on_device(dec)?;
        let (image_embeds, _deepstack) = self.vision.forward(&pixel_values, &grid_tensor)?;

        let t = grid[0];
        let h_m = grid[1] / merge;
        let w_m = grid[2] / merge;
        let n_vis_tokens = (t * h_m * w_m) as usize;

        let prompt = self.build_visual_prompt(n_vis_tokens);
        let input_ids = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| E::msg(format!("Tokenizer: {}", e)))?
            .get_ids()
            .to_vec();

        let input_embeds = self.merge_embeddings(&input_ids, &image_embeds)?;

        let grid_rows = [grid.to_vec()];
        let (t_pos, h_pos, w_pos) = self.compute_mrope_positions(&input_ids, &grid_rows);
        let (cos, sin) = self.mrope.forward_positions(&t_pos, &h_pos, &w_pos)?;

        // Final hidden + router-layer hidden (HF hidden_states[router_layer_index]).
        let (final_hidden, router_hidden) = self.decoder.forward_hidden(
            input_embeds,
            &cos,
            &sin,
            Some(self.config.router_layer_index),
        )?;
        let router_hidden = router_hidden
            .ok_or_else(|| E::msg("backbone did not return the router-layer hidden state"))?;

        // Isolation flag: CRANE_ARGUS_NO_MOE=1 skips the region-MoE fusion and
        // projects the raw backbone hidden states — used to separate MoE drift
        // from vision/backbone drift during parity validation.
        let fused = if std::env::var("CRANE_ARGUS_NO_MOE").map(|v| v == "1").unwrap_or(false) {
            final_hidden
        } else {
            self.apply_region_moe(
                &final_hidden,
                &router_hidden,
                &input_ids,
                [t as usize, h_m as usize, w_m as usize],
            )?
        };

        // Project + L2 normalize, then mask to image tokens (doc side).
        let emb = self.project_and_normalize(&fused)?.squeeze(0)?; // (seq, dims)
        let emb = if self.config.mask_non_image_embeddings {
            let mask = self.image_token_mask(&input_ids)?; // (seq, 1)
            emb.broadcast_mul(&mask)?
        } else {
            emb
        };
        Ok(emb)
    }

    // ── Query encoding ────────────────────────────────────────────────

    pub fn encode_queries(&mut self, queries: &[&str]) -> Result<Vec<Tensor>> {
        // The shipped `model.encode_queries` uses `process_texts`: the raw query
        // text tokenized with NO prefix, NO augmentation, NO special tokens.
        let mut out = Vec::new();
        for query in queries {
            let input_ids = self
                .tokenizer
                .encode(*query, false)
                .map_err(|e| E::msg(format!("Tokenizer: {}", e)))?
                .get_ids()
                .to_vec();
            let seq_len = input_ids.len();

            let ids_tensor = Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let input_embeds = self.decoder.embed(&ids_tensor)?;

            let positions: Vec<i64> = (0..seq_len as i64).collect();
            let (cos, sin) = self.mrope.forward_positions(&positions, &positions, &positions)?;

            // No MoE for queries; no router-layer capture.
            let (hidden, _) = self.decoder.forward_hidden(input_embeds, &cos, &sin, None)?;
            let proj = self.project_and_normalize(&hidden)?;
            out.push(proj.squeeze(0)?); // (seq, dims), all tokens kept
        }
        Ok(out)
    }

    // ── Argus region-level MoE ────────────────────────────────────────

    /// Reproduces `ArgusForRetrieval._apply_query_conditioned_moe` (query_context
    /// = None) for a single image, writing the fused vision tokens back into
    /// `final_hidden`. Returns the mutated (1, seq, H) hidden states.
    fn apply_region_moe(
        &self,
        final_hidden: &Tensor,
        router_hidden: &Tensor,
        input_ids: &[u32],
        grid: [usize; 3], // [t, h_merged, w_merged]
    ) -> Result<Tensor> {
        let [t, h, w] = grid;
        let hidden = self.config.text_config.hidden_size;
        let num_image_tokens = t * h * w;

        // Contiguous run of image tokens in the prompt.
        let start = input_ids
            .iter()
            .position(|&id| id == self.config.image_token_id)
            .ok_or_else(|| E::msg("no image tokens in prompt"))?;

        // early_grid / final_grid: (num_image_tokens, H) → (t,h,w,H) → mean over t.
        let sl = |src: &Tensor| -> candle_core::Result<Tensor> {
            src.i((0, start..start + num_image_tokens, ..))? // (num_image_tokens, H)
                .reshape((t, h, w, hidden))?
                .mean(0) // (h, w, H)
        };
        let early_grid = sl(router_hidden)?;
        let final_grid = sl(final_hidden)?;

        let fused_grid = self.region_moe_forward(&early_grid, &final_grid, h, w, hidden)?; // (h,w,H)

        // Expand fused grid over temporal frames and write back to image tokens.
        let fused_tokens = fused_grid
            .unsqueeze(0)? // (1,h,w,H)
            .broadcast_as((t, h, w, hidden))?
            .reshape((num_image_tokens, hidden))?
            .to_dtype(final_hidden.dtype())?
            .unsqueeze(0)?; // (1, num_image_tokens, H)

        let out = final_hidden.slice_assign(
            &[0..1, start..start + num_image_tokens, 0..hidden],
            &fused_tokens,
        )?;
        Ok(out)
    }

    fn region_moe_forward(
        &self,
        early_grid: &Tensor, // (h,w,H)
        final_grid: &Tensor, // (h,w,H)
        h: usize,
        w: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let rs = self.config.region_size;
        let num_h = h.div_ceil(rs);
        let num_w = w.div_ceil(rs);
        let n_spec = self.config.num_specialists;

        // Region pooling of early_grid → region_tokens (R, H) + host coords/counts.
        let (region_tokens, coords) = self.pool_regions(early_grid, h, w, hidden)?;

        let router_input = region_tokens.broadcast_add(&self.head.region_coord_proj.forward(&coords)?)?;
        let routing_logits = self.head.region_router.forward(&router_input)?; // (R, n_spec)
        let routing_probs = self.topk_sparse_probs(&routing_logits)?; // (R, n_spec) device, dtype

        let shared_out = self.head.shared_expert.forward(final_grid)?; // (h,w,H)
        let mut specialist_stack = Vec::with_capacity(n_spec);
        for e in &self.head.latent_experts {
            specialist_stack.push(e.forward(final_grid)?); // (h,w,H)
        }
        let specialist_outputs = Tensor::stack(&specialist_stack, 2)?; // (h,w,n_spec,H)

        let patch_probs = self.broadcast_region_probs(&routing_probs, num_h, num_w, h, w)?; // (h,w,n_spec)
        let specialist_out = specialist_outputs
            .broadcast_mul(&patch_probs.unsqueeze(D::Minus1)?)? // (h,w,n_spec,H)
            .sum(2)?; // (h,w,H)

        let shared_sig = sigmoid_scalar(self.head.gate_shared);
        let spec_sig = sigmoid_scalar(self.head.gate_specialist);

        let fused = (final_grid
            + (shared_out * shared_sig)?)?
            .add(&(specialist_out * spec_sig)?)?;
        Ok(fused)
    }

    /// Non-overlapping rs×rs average pool over a (h,w,H) grid → (R,H), plus the
    /// per-region normalized [x0,y0,x1,y1] coords as a (R,4) device tensor.
    /// Zero-padding cells contribute nothing; division uses the true valid count.
    fn pool_regions(
        &self,
        grid: &Tensor,
        h: usize,
        w: usize,
        hidden: usize,
    ) -> Result<(Tensor, Tensor)> {
        let rs = self.config.region_size;
        let num_h = h.div_ceil(rs);
        let num_w = w.div_ceil(rs);
        let hp = num_h * rs;
        let wp = num_w * rs;

        // Pad grid to (hp,wp,H) with zeros.
        let padded = if hp == h && wp == w {
            grid.clone()
        } else {
            let z = Tensor::zeros((hp, wp, hidden), grid.dtype(), grid.device())?;
            z.slice_assign(&[0..h, 0..w, 0..hidden], grid)?
        };

        // (num_h, rs, num_w, rs, H) → (num_h, num_w, rs, rs, H) → (R, rs*rs, H)
        let blocks = padded
            .reshape((num_h, rs, num_w, rs, hidden))?
            .permute((0, 2, 1, 3, 4))?
            .reshape((num_h * num_w, rs * rs, hidden))?
            .contiguous()?;
        let summed = blocks.sum(1)?; // (R, H)

        // Host-side per-region valid counts and coords.
        let mut counts = Vec::with_capacity(num_h * num_w);
        let mut coords = Vec::with_capacity(num_h * num_w * 4);
        for ry in 0..num_h {
            for rx in 0..num_w {
                let rows = ((ry + 1) * rs).min(h).saturating_sub(ry * rs);
                let cols = ((rx + 1) * rs).min(w).saturating_sub(rx * rs);
                counts.push((rows * cols).max(1) as f32);
                let y0 = (ry * rs) as f32 / (h.max(1) as f32);
                let x0 = (rx * rs) as f32 / (w.max(1) as f32);
                let y1 = ((ry + 1) * rs).min(h) as f32 / (h.max(1) as f32);
                let x1 = ((rx + 1) * rs).min(w) as f32 / (w.max(1) as f32);
                coords.extend_from_slice(&[x0, y0, x1, y1]);
            }
        }
        let r = num_h * num_w;
        let counts_t = Tensor::from_vec(counts, (r, 1), grid.device())?.to_dtype(grid.dtype())?;
        let region_tokens = summed.broadcast_div(&counts_t)?; // (R, H)
        let coords_t = Tensor::from_vec(coords, (r, 4), grid.device())?.to_dtype(grid.dtype())?;
        Ok((region_tokens, coords_t))
    }

    /// Top-k sparse softmax over specialists, on the host for exactness (R is
    /// small). Matches `_topk_sparse_probs`: -inf mask on non-top-k, softmax
    /// with `router_temperature`.
    fn topk_sparse_probs(&self, logits: &Tensor) -> Result<Tensor> {
        let (r, e) = logits.dims2()?;
        let data = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let temp = (self.config.router_temperature).max(1e-6) as f32;
        let k = self.config.top_k_experts.min(e).max(1);

        let mut probs = vec![0f32; r * e];
        for (ri, row) in data.iter().enumerate() {
            // indices of the top-k logits.
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
            let keep = &idx[..k];
            // softmax over kept logits / temp; others = 0.
            let maxv = keep.iter().map(|&i| row[i]).fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            for &i in keep {
                denom += ((row[i] - maxv) / temp).exp();
            }
            for &i in keep {
                probs[ri * e + i] = ((row[i] - maxv) / temp).exp() / denom;
            }
        }
        let t = Tensor::from_vec(probs, (r, e), logits.device())?.to_dtype(logits.dtype())?;
        Ok(t)
    }

    /// Broadcast per-region probs (R, n_spec) back onto the (h,w) patch grid.
    fn broadcast_region_probs(
        &self,
        region_probs: &Tensor,
        num_h: usize,
        num_w: usize,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let rs = self.config.region_size;
        let n_spec = self.config.num_specialists;
        // (num_h,num_w,1,1,n_spec) → expand (num_h,num_w,rs,rs,n_spec)
        //   → (num_h,rs,num_w,rs,n_spec) → (hp,wp,n_spec) → crop.
        let probs = region_probs
            .reshape((num_h, num_w, 1, 1, n_spec))?
            .broadcast_as((num_h, num_w, rs, rs, n_spec))?
            .permute((0, 2, 1, 3, 4))?
            .reshape((num_h * rs, num_w * rs, n_spec))?
            .contiguous()?;
        Ok(probs.i((0..h, 0..w, ..))?)
    }

    // ── Scoring (identical math to ColQwen3Emb) ───────────────────────

    pub fn stack_passages(ps: &[Tensor]) -> Result<Tensor> {
        if ps.is_empty() {
            anyhow::bail!("Empty passage embeddings");
        }
        let device = ps[0].device();
        let dtype = ps[0].dtype();
        let dims = ps[0].dim(D::Minus1)?;
        let max_sp = ps.iter().map(|p| p.dim(0).unwrap_or(0)).max().unwrap_or(0);
        let mut ps_stacked = Tensor::zeros((ps.len(), max_sp, dims), dtype, device)?;
        for (i, p) in ps.iter().enumerate() {
            let sp = p.dim(0)?;
            ps_stacked = ps_stacked.slice_assign(&[i..i + 1, 0..sp, 0..dims], &p.unsqueeze(0)?)?;
        }
        Ok(ps_stacked.transpose(1, 2)?.contiguous()?)
    }

    pub fn score(qs: &[Tensor], ps: &[Tensor], batch_size: usize) -> Result<Tensor> {
        if qs.is_empty() || ps.is_empty() {
            anyhow::bail!("Empty query or passage embeddings");
        }
        let ps_t = Self::stack_passages(ps)?;
        Self::score_stacked(qs, &ps_t, batch_size)
    }

    pub fn score_stacked(qs: &[Tensor], ps_t: &Tensor, batch_size: usize) -> Result<Tensor> {
        if qs.is_empty() {
            anyhow::bail!("Empty query embeddings");
        }
        let dims = qs[0].dim(D::Minus1)?;
        let n_pages = ps_t.dim(0)?;
        let mut scores_list: Vec<Tensor> = Vec::new();

        for i in (0..qs.len()).step_by(batch_size) {
            let end_q = (i + batch_size).min(qs.len());
            let mut q_scores_all = Vec::new();
            for qi in i..end_q {
                let q = &qs[qi];
                let sq = q.dim(0)?;
                let mut chunk_scores = Vec::new();
                for j in (0..n_pages).step_by(batch_size) {
                    let end_p = (j + batch_size).min(n_pages);
                    let ps_chunk = ps_t.narrow(0, j, end_p - j)?;
                    let chunk_size = end_p - j;
                    let q_expanded = q.unsqueeze(0)?.expand(&[chunk_size, sq, dims])?;
                    let dots = q_expanded.matmul(&ps_chunk)?;
                    let max_sim = dots.max(D::Minus1)?;
                    let scores = max_sim.sum(D::Minus1)?;
                    chunk_scores.push(scores);
                }
                q_scores_all.push(Tensor::cat(&chunk_scores, 0)?);
            }
            scores_list.push(Tensor::stack(&q_scores_all, 0)?);
        }
        let scores = Tensor::cat(&scores_list, 0)?.to_dtype(DType::F32)?;
        Ok(scores)
    }

    // ── Internal helpers ──────────────────────────────────────────────

    fn project_and_normalize(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let proj = self.head.custom_text_proj.forward(hidden)?;
        let proj = if self.dims < self.config.retrieval_dim {
            proj.narrow(D::Minus1, 0, self.dims)?
        } else {
            proj
        };
        let norm = proj.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let norm = (norm + 1e-12)?;
        proj.broadcast_div(&norm)
    }

    /// (seq, 1) mask, 1.0 at image-token positions.
    fn image_token_mask(&self, input_ids: &[u32]) -> candle_core::Result<Tensor> {
        let vals: Vec<f32> = input_ids
            .iter()
            .map(|&id| if id == self.config.image_token_id { 1.0 } else { 0.0 })
            .collect();
        Tensor::from_vec(vals, (input_ids.len(), 1), &self.device)?.to_dtype(self.dtype)
    }

    fn build_visual_prompt(&self, n_vis_tokens: usize) -> String {
        // Matches ArgusProcessor.visual_prompt_prefix (no assistant turn).
        let mut prompt = String::from("<|im_start|>user\n<|vision_start|>");
        for _ in 0..n_vis_tokens {
            prompt.push_str("<|image_pad|>");
        }
        prompt.push_str("<|vision_end|>Describe the image.<|im_end|><|endoftext|>");
        prompt
    }

    fn compute_mrope_positions(
        &self,
        input_ids: &[u32],
        grid_thw_vec: &[Vec<u32>],
    ) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
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
        (t_pos, h_pos, w_pos)
    }

    /// One embed pass over the whole id sequence, then overwrite image_pad rows
    /// with vision embeddings (no deepstack for this model → no mask needed).
    fn merge_embeddings(&self, input_ids: &[u32], image_embeds: &Tensor) -> candle_core::Result<Tensor> {
        let image_token = self.config.image_token_id;
        let num_vis_tokens = image_embeds.dim(0)?;
        let hidden_size = image_embeds.dim(1)?;
        let n = input_ids.len();

        let ids_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let mut combined = self.decoder.embed(&ids_tensor)?; // (1, N, H)

        let mut runs: Vec<(usize, usize, usize)> = Vec::new(); // (text_pos, vis_offset, len)
        let mut vis_off = 0usize;
        let mut run_start: Option<(usize, usize)> = None;
        for (i, &id) in input_ids.iter().enumerate() {
            if id == image_token && vis_off < num_vis_tokens {
                if run_start.is_none() {
                    run_start = Some((i, vis_off));
                }
                vis_off += 1;
                if i + 1 == n || input_ids[i + 1] != image_token || vis_off == num_vis_tokens {
                    let (rs, vs) = run_start.take().unwrap();
                    runs.push((rs, vs, i - rs + 1));
                }
            }
        }
        for (text_pos, vis_offset, len) in runs {
            let block = image_embeds.narrow(0, vis_offset, len)?.unsqueeze(0)?;
            combined = combined.slice_assign(&[0..1, text_pos..text_pos + len, 0..hidden_size], &block)?;
        }
        Ok(combined)
    }

    fn preproc_params(&self) -> PreprocParams {
        PreprocParams {
            factor: self.proc.patch_size * self.proc.merge_size,
            min_pixels: self.proc.size.shortest_edge,
            max_pixels: self.proc.size.longest_edge,
        }
    }

    fn patchify_on_device(&self, dec: DecodedImage) -> Result<(Tensor, Tensor, [u32; 3])> {
        let merge_size = self.proc.merge_size;
        let patch_size = self.proc.patch_size;
        let temporal_patch_size = self.proc.temporal_patch_size;
        let DecodedImage { raw, rh, rw } = dec;

        let raw_tensor = Tensor::from_vec(raw, (rh, rw, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .to_device(&self.device)?;
        let raw_f32 = (raw_tensor.to_dtype(DType::F32)? * (1.0 / 255.0))?;
        let tensor = raw_f32
            .broadcast_sub(&self.img_mean)?
            .broadcast_div(&self.img_std)?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?;
        let tensor = Tensor::cat(&[&tensor, &tensor], 0)?;

        let grid_t = 2usize / temporal_patch_size;
        let grid_h = rh / patch_size;
        let grid_w = rw / patch_size;

        let tensor = tensor.reshape(Shape::from(vec![
            grid_t,
            temporal_patch_size,
            3,
            grid_h / merge_size,
            merge_size,
            patch_size,
            grid_w / merge_size,
            merge_size,
            patch_size,
        ]))?;
        let tensor = tensor.permute(vec![0, 3, 6, 4, 7, 2, 1, 5, 8])?;
        let tensor = tensor
            .reshape((
                grid_t * grid_h * grid_w,
                3 * temporal_patch_size * patch_size * patch_size,
            ))?
            .contiguous()?;

        let grid = [grid_t as u32, grid_h as u32, grid_w as u32];
        let grid_tensor = Tensor::from_vec(grid.to_vec(), (1, 3), &self.device)?;
        Ok((tensor, grid_tensor, grid))
    }
}

fn sigmoid_scalar(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

use candle_core::IndexOp;

// ── CPU preprocessing helpers ────────────────────────────────────────

struct DecodedImage {
    raw: Vec<u8>,
    rh: usize,
    rw: usize,
}

#[derive(Clone, Copy)]
struct PreprocParams {
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
}

fn resize_image_cpu(img: image::RgbImage, p: PreprocParams) -> Result<DecodedImage> {
    let (w, h) = (img.width(), img.height());
    let (rh, rw) =
        crate::utils::image_utils::smart_resize(h as usize, w as usize, p.factor, p.min_pixels, p.max_pixels)?;
    let img = image::imageops::resize(&img, rw as u32, rh as u32, image::imageops::FilterType::CatmullRom);
    Ok(DecodedImage { raw: img.into_raw(), rh, rw })
}
