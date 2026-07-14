//! ColQwen3.5 multi-vector retriever (ColPali-style late interaction).
//!
//! Same Qwen3.5-VL *hybrid* backbone as Argus (`super::super::qwen3_5_vl`) but
//! with the plain ColQwen retrieval head — a single `custom_text_proj` linear to
//! `dim` (320) + L2 norm — instead of Argus's region-MoE fusion.
//!
//! Faithful to `colpali_engine.models.qwen3_5.colqwen3_5` (commit 2e0b927):
//!   Doc encode:   visual_prompt → backbone → custom_text_proj → L2 norm.
//!                 `mask_non_image_embeddings` defaults to **False**, so ALL
//!                 token embeddings are kept (image + prompt text), unlike Argus.
//!   Query encode: raw query + `"<|endoftext|>"×10` (empty query_prefix) →
//!                 backbone → custom_text_proj → L2 norm.
//! Deployed at `max_num_visual_tokens = 1792` per the model card.

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Module, Shape, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

use super::super::qwen3_5_vl::{MRoPE, TextConfig, TextDecoder, VisionConfig, VisionModel};

/// Query augmentation: colpali `BaseVisualRetrieverProcessor.process_queries`
/// appends `query_augmentation_token * 10`; ColQwen3_5's `query_prefix` is "".
const QUERY_AUGMENTATION_TOKEN: &str = "<|endoftext|>";
const QUERY_AUGMENTATION_COUNT: usize = 10;
/// Deployment visual-token budget (`max_num_visual_tokens=1792` in the card).
const MAX_NUM_VISUAL_TOKENS: usize = 1792;

// ── Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ColQwen3_5Config {
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    pub image_token_id: u32,
    // parsed from config.json for completeness; not read on the retrieval path
    #[serde(default)]
    #[allow(dead_code)]
    pub video_token_id: u32,
    #[serde(default)]
    #[allow(dead_code)]
    pub vision_start_token_id: u32,
    #[serde(default)]
    #[allow(dead_code)]
    pub vision_end_token_id: u32,
    /// Retrieval projection dim (`dim` in config.json, 320 for Vultron).
    /// (config also carries a redundant `embed_dim`; we read `dim`.)
    #[serde(default = "default_dim")]
    pub dim: usize,
    /// colpali default is False → keep all token embeddings on the doc side.
    #[serde(default)]
    pub mask_non_image_embeddings: bool,
}

fn default_dim() -> usize {
    128
}

/// `processor_config.json` → `image_processor` sub-object.
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
    #[allow(dead_code)]
    longest_edge: usize,
}

// ── Model ────────────────────────────────────────────────────────────

pub struct ColQwen3_5Emb {
    vision: VisionModel,
    decoder: TextDecoder,
    mrope: MRoPE,
    custom_text_proj: Linear,
    tokenizer: Tokenizer,
    config: ColQwen3_5Config,
    proc: ImageProcessorConfig,
    dims: usize,
    pub device: Device,
    dtype: DType,
    img_mean: Tensor,
    img_std: Tensor,
}

impl ColQwen3_5Emb {
    pub fn from_local(path: impl AsRef<Path>, cpu: bool, bf16: bool) -> Result<Self> {
        let device = if cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
        let dtype = if bf16 && device.is_cuda() { DType::BF16 } else { DType::F32 };

        let base = path.as_ref();
        let config: ColQwen3_5Config =
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

        println!("Loading ColQwen3.5 retrieval projection (custom_text_proj)...");
        let custom_text_proj =
            linear(config.text_config.hidden_size, config.dim, vb.pp("custom_text_proj"))?;

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

        let dims = config.dim;
        println!(
            "ColQwen3.5 model loaded! dim={} mask_non_image={} visual_tokens={}",
            dims, config.mask_non_image_embeddings, MAX_NUM_VISUAL_TOKENS
        );

        Ok(Self {
            vision,
            decoder,
            mrope,
            custom_text_proj,
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

    /// Matryoshka-style truncation (off by default; ColQwen3.5 trains at `dim`).
    pub fn set_dims(&mut self, dims: usize) {
        assert!(dims <= self.config.dim, "target dims ({}) > dim ({})", dims, self.config.dim);
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

    /// GPU stage for one image: patchify → vision → decoder → project + norm.
    /// Keeps all token embeddings unless `mask_non_image_embeddings` is set.
    fn encode_one_decoded(&mut self, dec: DecodedImage) -> Result<Tensor> {
        let merge = self.config.vision_config.spatial_merge_size as u32;

        let profile = std::env::var("CRANE_PROFILE").map(|v| v == "1").unwrap_or(false);
        if profile {
            self.device.synchronize()?;
        }
        let t_start = std::time::Instant::now();

        let (pixel_values, grid_tensor, grid) = self.patchify_on_device(dec)?;
        let (image_embeds, _deepstack) = self.vision.forward(&pixel_values, &grid_tensor)?;
        if profile {
            self.device.synchronize()?;
        }
        let t_vis = t_start.elapsed().as_secs_f64();

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

        if profile {
            self.device.synchronize()?;
        }
        let t_dec0 = std::time::Instant::now();
        let (hidden, _) = self.decoder.forward_hidden(input_embeds, &cos, &sin, None)?;
        if profile {
            self.device.synchronize()?;
        }
        let t_dec = t_dec0.elapsed().as_secs_f64();

        let emb = self.project_and_normalize(&hidden)?.squeeze(0)?; // (seq, dims)
        if profile {
            eprintln!(
                "VULTRON_PROFILE: seq={} vision={:.0}ms decoder={:.0}ms",
                input_ids.len(),
                t_vis * 1e3,
                t_dec * 1e3
            );
            super::super::qwen3_5_vl::gdn_prof_report();
        }
        // colpali default keeps all tokens; only mask when explicitly enabled.
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
        // query_prefix="" + query + query_augmentation_token*10.
        let mut out = Vec::new();
        for query in queries {
            let processed = format!(
                "{}{}",
                query,
                QUERY_AUGMENTATION_TOKEN.repeat(QUERY_AUGMENTATION_COUNT)
            );
            let input_ids = self
                .tokenizer
                .encode(processed.as_str(), false)
                .map_err(|e| E::msg(format!("Tokenizer: {}", e)))?
                .get_ids()
                .to_vec();
            let seq_len = input_ids.len();

            let ids_tensor = Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let input_embeds = self.decoder.embed(&ids_tensor)?;

            let positions: Vec<i64> = (0..seq_len as i64).collect();
            let (cos, sin) = self.mrope.forward_positions(&positions, &positions, &positions)?;

            let (hidden, _) = self.decoder.forward_hidden(input_embeds, &cos, &sin, None)?;
            let proj = self.project_and_normalize(&hidden)?;
            out.push(proj.squeeze(0)?); // (seq, dims), all tokens kept
        }
        Ok(out)
    }

    // ── Scoring (identical MaxSim to ColQwen3Emb) ─────────────────────

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
        let proj = self.custom_text_proj.forward(hidden)?;
        let proj = if self.dims < self.config.dim {
            proj.narrow(D::Minus1, 0, self.dims)?
        } else {
            proj
        };
        // proj / ||proj||_2  (reference uses no epsilon; guard against 0).
        let norm = proj.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let norm = (norm + 1e-12)?;
        proj.broadcast_div(&norm)
    }

    fn image_token_mask(&self, input_ids: &[u32]) -> candle_core::Result<Tensor> {
        let vals: Vec<f32> = input_ids
            .iter()
            .map(|&id| if id == self.config.image_token_id { 1.0 } else { 0.0 })
            .collect();
        Tensor::from_vec(vals, (input_ids.len(), 1), &self.device)?.to_dtype(self.dtype)
    }

    fn build_visual_prompt(&self, n_vis_tokens: usize) -> String {
        // ColQwen3_5Processor.visual_prompt_prefix (image_pad repeated n times).
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
        let factor = self.proc.patch_size * self.proc.merge_size;
        // Deploy at max_num_visual_tokens=1792 (card); overridable via
        // CRANE_VISUAL_TOKENS to trade a little recall for speed.
        let max_tokens = std::env::var("CRANE_VISUAL_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&t| t > 0)
            .unwrap_or(MAX_NUM_VISUAL_TOKENS);
        PreprocParams {
            factor,
            min_pixels: self.proc.size.shortest_edge,
            max_pixels: max_tokens * factor * factor,
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
