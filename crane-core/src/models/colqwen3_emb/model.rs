//! ColQwen3 Embedding Model: multi-vector document/query embedder built on Qwen3-VL.
//!
//! Architecture: Qwen3-VL backbone (vision encoder + text decoder) + projection + L2 norm.
//! Produces per-token embeddings for ColBERT-style late interaction scoring (MaxSim).
//!
//! Reference: OpenSearch-AI/Ops-Colqwen3-4B

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Module, Shape, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

use super::super::qwen3_vl::{
    MRoPE, Qwen3VLConfig, TextConfig, TextDecoder, VisionConfig, VisionModel,
};
use super::super::qwen3_vl::config::PreprocessorConfig;

// ── Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ColQwen3Config {
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    #[serde(default = "default_dims")]
    pub dims: usize,
    #[serde(default)]
    pub mask_non_image_embeddings: bool,
}

fn default_dims() -> usize { 2560 }

impl ColQwen3Config {
    /// Convert to Qwen3VLConfig for reusing the backbone components.
    pub fn as_qwen3vl_config(&self) -> Qwen3VLConfig {
        Qwen3VLConfig {
            vision_config: self.vision_config.clone(),
            text_config: self.text_config.clone(),
            image_token_id: self.image_token_id,
            video_token_id: self.video_token_id,
            vision_start_token_id: self.vision_start_token_id,
            vision_end_token_id: self.vision_end_token_id,
        }
    }
}

// ── Model ────────────────────────────────────────────────────────────

pub struct ColQwen3Emb {
    vision: VisionModel,
    decoder: TextDecoder,
    mrope: MRoPE,
    custom_text_proj: Linear,
    tokenizer: Tokenizer,
    config: ColQwen3Config,
    preproc_cfg: PreprocessorConfig,
    dims: usize,
    pub device: Device,
    dtype: DType,
    img_mean: Tensor,
    img_std: Tensor,
}

/// Query processing constants (matching Python OpsColQwen3Processor).
const QUERY_PREFIX: &str = "Query: ";
const QUERY_AUGMENTATION_TOKEN: &str = "<|endoftext|>";
const QUERY_AUGMENTATION_COUNT: usize = 10;

impl ColQwen3Emb {
    pub fn from_local(path: impl AsRef<Path>, cpu: bool, bf16: bool) -> Result<Self> {
        let device = if cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
        let dtype = if bf16 && device.is_cuda() { DType::BF16 } else { DType::F32 };

        let base = path.as_ref();
        let config: ColQwen3Config =
            serde_json::from_str(&std::fs::read_to_string(base.join("config.json"))?)?;
        let preproc_cfg: PreprocessorConfig =
            serde_json::from_str(&std::fs::read_to_string(base.join("preprocessor_config.json"))?)?;
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

        // ColQwen3 BF16 weight prefixes: visual.*, language_model.*, custom_text_proj.*
        println!("Loading vision encoder...");
        let vision = VisionModel::new(vb.pp("visual"), &config.vision_config)?;

        println!("Loading text decoder (embedding mode, no lm_head)...");
        let decoder = TextDecoder::new(&config.text_config, vb.pp("language_model"))?;

        println!("Loading projection layer...");
        let custom_text_proj = linear(
            config.text_config.hidden_size,
            config.dims,
            vb.pp("custom_text_proj"),
        )?;

        println!("Initializing M-RoPE...");
        let mrope = MRoPE::new(&config.text_config, &device, dtype)?;

        let img_mean = Tensor::new(
            &[preproc_cfg.image_mean[0] as f32, preproc_cfg.image_mean[1] as f32, preproc_cfg.image_mean[2] as f32],
            &device,
        )?.reshape((3, 1, 1))?;
        let img_std = Tensor::new(
            &[preproc_cfg.image_std[0] as f32, preproc_cfg.image_std[1] as f32, preproc_cfg.image_std[2] as f32],
            &device,
        )?.reshape((3, 1, 1))?;

        let dims = config.dims;
        println!("ColQwen3 embedding model loaded! dims={}", dims);

        Ok(Self { vision, decoder, mrope, custom_text_proj, tokenizer, config, preproc_cfg, dims, device, dtype, img_mean, img_std })
    }

    /// Override the output embedding dimensions (Matryoshka truncation).
    /// Must be ≤ the projection layer output size (config.dims).
    pub fn set_dims(&mut self, dims: usize) {
        assert!(dims <= self.config.dims, "target dims ({}) > projection dims ({})", dims, self.config.dims);
        println!("ColQwen3 dims overridden: {} → {}", self.dims, dims);
        self.dims = dims;
    }

    /// Encode a batch of images into multi-vector embeddings.
    /// Returns one tensor per image, each of shape (num_tokens, dims).
    pub fn encode_images<P: AsRef<Path>>(&mut self, image_paths: &[P]) -> Result<Vec<Tensor>> {
        let (pixel_values, grid_thw) = self.preprocess_images(image_paths)?;

        // Vision encoder
        let (image_embeds, deepstack_features) = self.vision.forward(&pixel_values, &grid_thw)?;

        // Build visual prompt for each image
        let grid_thw_vec = grid_thw.to_vec2::<u32>()?;
        let merge = self.config.vision_config.spatial_merge_size as u32;

        let mut all_embeddings = Vec::new();
        let mut vis_offset = 0usize;

        for grid in &grid_thw_vec {
            let t = grid[0];
            let h = grid[1] / merge;
            let w = grid[2] / merge;
            let n_vis_tokens = (t * h * w) as usize;

            // Extract this image's vision embeddings
            let img_embeds = image_embeds.narrow(0, vis_offset, n_vis_tokens)?;

            // Build prompt tokens for this image
            let prompt = self.build_visual_prompt(n_vis_tokens);
            let input_ids = self.tokenizer.encode(prompt.as_str(), false)
                .map_err(|e| E::msg(format!("Tokenizer: {}", e)))?
                .get_ids().to_vec();

            // Merge text+vision embeddings
            let (input_embeds, vision_mask) = self.merge_embeddings(&input_ids, &img_embeds)?;

            // M-RoPE positions
            let single_grid = grid_thw.narrow(0, grid_thw_vec.iter().position(|g| g == grid).unwrap_or(0), 1)?;
            let (position_ids, _) = self.compute_mrope_positions(&input_ids, &single_grid)?;
            let (cos, sin) = self.mrope.forward(&position_ids)?;

            // Forward through decoder (returns all hidden states)
            let hidden = self.decoder.forward_hidden(
                input_embeds, &cos, &sin,
                Some(&deepstack_features), Some(&vision_mask),
            )?;

            // Project + L2 normalize
            let proj = self.project_and_normalize(&hidden)?;

            all_embeddings.push(proj.squeeze(0)?); // (seq_len, dims)
            vis_offset += n_vis_tokens;
        }

        Ok(all_embeddings)
    }

    /// Encode images from raw bytes (JPEG/PNG) into multi-vector embeddings.
    /// Avoids disk I/O — images are decoded directly from memory.
    pub fn encode_images_from_bytes(&mut self, images: &[&[u8]]) -> Result<Vec<Tensor>> {
        let (pixel_values, grid_thw) = self.preprocess_images_from_bytes(images)?;

        let (image_embeds, deepstack_features) = self.vision.forward(&pixel_values, &grid_thw)?;

        let grid_thw_vec = grid_thw.to_vec2::<u32>()?;
        let merge = self.config.vision_config.spatial_merge_size as u32;

        let mut all_embeddings = Vec::new();
        let mut vis_offset = 0usize;

        for grid in &grid_thw_vec {
            let t = grid[0];
            let h = grid[1] / merge;
            let w = grid[2] / merge;
            let n_vis_tokens = (t * h * w) as usize;

            let img_embeds = image_embeds.narrow(0, vis_offset, n_vis_tokens)?;

            let prompt = self.build_visual_prompt(n_vis_tokens);
            let input_ids = self.tokenizer.encode(prompt.as_str(), false)
                .map_err(|e| E::msg(format!("Tokenizer: {}", e)))?
                .get_ids().to_vec();

            let (input_embeds, vision_mask) = self.merge_embeddings(&input_ids, &img_embeds)?;

            let single_grid = grid_thw.narrow(0, grid_thw_vec.iter().position(|g| g == grid).unwrap_or(0), 1)?;
            let (position_ids, _) = self.compute_mrope_positions(&input_ids, &single_grid)?;
            let (cos, sin) = self.mrope.forward(&position_ids)?;

            let hidden = self.decoder.forward_hidden(
                input_embeds, &cos, &sin,
                Some(&deepstack_features), Some(&vision_mask),
            )?;

            let proj = self.project_and_normalize(&hidden)?;
            all_embeddings.push(proj.squeeze(0)?);
            vis_offset += n_vis_tokens;
        }

        Ok(all_embeddings)
    }

    /// Encode a batch of text queries into multi-vector embeddings.
    /// Returns one tensor per query, each of shape (num_tokens, dims).
    pub fn encode_queries(&mut self, queries: &[&str]) -> Result<Vec<Tensor>> {
        let mut all_embeddings = Vec::new();

        for query in queries {
            // Build query with prefix + augmentation tokens
            let processed = format!(
                "{}{}{}",
                QUERY_PREFIX,
                query,
                QUERY_AUGMENTATION_TOKEN.repeat(QUERY_AUGMENTATION_COUNT),
            );

            let input_ids = self.tokenizer.encode(processed.as_str(), false)
                .map_err(|e| E::msg(format!("Tokenizer: {}", e)))?
                .get_ids().to_vec();

            let seq_len = input_ids.len();

            // Token embeddings (no vision)
            let ids_tensor = Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let input_embeds = self.decoder.embed(&ids_tensor)?;

            // M-RoPE: text-only → all 3 dims get same sequential positions
            let positions: Vec<i64> = (0..seq_len as i64).collect();
            let pos_tensor = Tensor::new(positions.as_slice(), &self.device)?;
            let position_ids = Tensor::stack(&[pos_tensor.clone(), pos_tensor.clone(), pos_tensor], 0)?;
            let (cos, sin) = self.mrope.forward(&position_ids)?;

            // Forward (no vision features)
            let hidden = self.decoder.forward_hidden(
                input_embeds, &cos, &sin, None, None,
            )?;

            // Project + L2 normalize
            let proj = self.project_and_normalize(&hidden)?;

            // All tokens are valid (no masking needed for queries)
            all_embeddings.push(proj.squeeze(0)?); // (seq_len, dims)
        }

        Ok(all_embeddings)
    }

    /// ColBERT-style MaxSim scoring between query and passage embeddings.
    /// Returns (num_queries, num_passages) score matrix.
    ///
    /// Vectorized: pads all passages to max_seq_len, stacks into a single tensor,
    /// then scores all passages in one batched matmul per query chunk.
    pub fn score(
        qs: &[Tensor],
        ps: &[Tensor],
        batch_size: usize,
    ) -> Result<Tensor> {
        if qs.is_empty() || ps.is_empty() {
            anyhow::bail!("Empty query or passage embeddings");
        }

        let device = qs[0].device();
        let dtype = qs[0].dtype();
        let dims = qs[0].dim(D::Minus1)?;

        // Pre-allocate (N_pages, max_sp, dims) zeros and slice_assign each passage.
        // Avoids per-passage cat (which copies the full max_sp×dims block).
        let max_sp = ps.iter().map(|p| p.dim(0).unwrap_or(0)).max().unwrap_or(0);
        let mut ps_stacked = Tensor::zeros((ps.len(), max_sp, dims), dtype, device)?;
        for (i, p) in ps.iter().enumerate() {
            let sp = p.dim(0)?;
            ps_stacked = ps_stacked.slice_assign(
                &[i..i + 1, 0..sp, 0..dims],
                &p.unsqueeze(0)?,
            )?;
        }
        // (N_pages, dims, max_sp) for matmul
        let ps_t = ps_stacked.transpose(1, 2)?.contiguous()?;

        let mut scores_list: Vec<Tensor> = Vec::new();

        for i in (0..qs.len()).step_by(batch_size) {
            let end_q = (i + batch_size).min(qs.len());
            let mut q_scores_all = Vec::new();

            for qi in i..end_q {
                let q = &qs[qi]; // (sq, dims)
                let sq = q.dim(0)?;

                // Process passages in chunks to limit VRAM
                let mut chunk_scores = Vec::new();
                for j in (0..ps.len()).step_by(batch_size) {
                    let end_p = (j + batch_size).min(ps.len());
                    let ps_chunk = ps_t.narrow(0, j, end_p - j)?; // (chunk, dims, max_sp)
                    let chunk_size = end_p - j;

                    // q: (sq, dims) → broadcast (chunk, sq, dims)
                    let q_expanded = q.unsqueeze(0)?.expand(&[chunk_size, sq, dims])?;

                    // (chunk, sq, dims) @ (chunk, dims, max_sp) → (chunk, sq, max_sp)
                    let dots = q_expanded.matmul(&ps_chunk)?;

                    // max over max_sp dim → (chunk, sq), sum over sq → (chunk,)
                    let max_sim = dots.max(D::Minus1)?;
                    let scores = max_sim.sum(D::Minus1)?; // (chunk,)
                    chunk_scores.push(scores);
                }
                q_scores_all.push(Tensor::cat(&chunk_scores, 0)?); // (N_pages,)
            }
            scores_list.push(Tensor::stack(&q_scores_all, 0)?);
        }

        let scores = Tensor::cat(&scores_list, 0)?.to_dtype(DType::F32)?;
        Ok(scores)
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Apply custom_text_proj + L2 normalization.
    fn project_and_normalize(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let proj = self.custom_text_proj.forward(hidden)?;

        // Matryoshka truncation: take first `dims` dimensions if less than projection output
        let proj = if self.dims < self.config.dims {
            proj.narrow(D::Minus1, 0, self.dims)?
        } else {
            proj
        };

        // L2 normalize
        let norm = proj.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let norm = (norm + 1e-12)?;
        proj.broadcast_div(&norm)
    }

    fn build_visual_prompt(&self, n_vis_tokens: usize) -> String {
        // Matches OpsColQwen3Processor.visual_prompt_prefix exactly — no `\n`
        // between <|im_end|> and <|im_start|>.
        let mut prompt = String::from("<|im_start|>user\n<|vision_start|>");
        for _ in 0..n_vis_tokens {
            prompt.push_str("<|image_pad|>");
        }
        prompt.push_str("<|vision_end|>Describe the image.<|im_end|><|im_start|>assistant\n<|endoftext|>");
        prompt
    }

    fn compute_mrope_positions(
        &self,
        input_ids: &[u32],
        grid_thw: &Tensor,
    ) -> Result<(Tensor, i64)> {
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
        let position_ids = Tensor::stack(&[t_tensor, h_tensor, w_tensor], 0)?;
        Ok((position_ids, text_pos))
    }

    /// Build (input_embeds, vision_mask) by:
    ///   1) ONE embed call over the whole id sequence (image_pad rows will be overwritten)
    ///   2) slice_assign of vision embeddings over the image_pad positions.
    /// Numerically equivalent to embedding text-runs and concatenating with vision blocks.
    fn merge_embeddings(
        &self,
        input_ids: &[u32],
        image_embeds: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let image_token = self.config.image_token_id;
        let num_vis_tokens = image_embeds.dim(0)?;
        let hidden_size = image_embeds.dim(1)?;
        let n = input_ids.len();

        // Single embed pass — image_pad rows will be replaced below.
        let ids_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let mut combined = self.decoder.embed(&ids_tensor)?; // (1, N, H)

        // Detect contiguous runs of image_pad and the per-position mask.
        let mut mask_vals: Vec<f32> = Vec::with_capacity(n);
        let mut runs: Vec<(usize, usize, usize)> = Vec::new(); // (text_pos, vis_offset, len)
        let mut vis_off = 0usize;
        let mut run_start: Option<(usize, usize)> = None;

        for (i, &id) in input_ids.iter().enumerate() {
            if id == image_token && vis_off < num_vis_tokens {
                mask_vals.push(1.0f32);
                if run_start.is_none() {
                    run_start = Some((i, vis_off));
                }
                vis_off += 1;
                if i + 1 == n || input_ids[i + 1] != image_token || vis_off == num_vis_tokens {
                    let (rs, vs) = run_start.take().unwrap();
                    runs.push((rs, vs, i - rs + 1));
                }
            } else {
                mask_vals.push(0.0f32);
            }
        }

        // Overwrite image_pad rows with vision embeddings.
        for (text_pos, vis_offset, len) in runs {
            let block = image_embeds.narrow(0, vis_offset, len)?.unsqueeze(0)?;
            combined = combined.slice_assign(
                &[0..1, text_pos..text_pos + len, 0..hidden_size],
                &block,
            )?;
        }

        let mask = Tensor::new(mask_vals, &self.device)?;
        Ok((combined, mask))
    }

    /// Preprocess a single RGB image into pixel patches + grid_thw.
    fn preprocess_one_image(
        &self,
        img: image::RgbImage,
    ) -> Result<(Tensor, Tensor)> {
        let merge_size = self.preproc_cfg.merge_size;
        let patch_size = self.preproc_cfg.patch_size;
        let temporal_patch_size = self.preproc_cfg.temporal_patch_size;
        let factor = patch_size * merge_size;
        let min_pixels = self.preproc_cfg.size.shortest_edge;
        let max_pixels = self.preproc_cfg.size.longest_edge;

        let (w, h) = (img.width(), img.height());
        let (rh, rw) = crate::utils::image_utils::smart_resize(
            h as usize, w as usize, factor, min_pixels, max_pixels,
        )?;
        let img = image::imageops::resize(&img, rw as u32, rh as u32, image::imageops::FilterType::CatmullRom);

        let raw: Vec<u8> = img.into_raw();
        let raw_tensor = Tensor::from_vec(raw, (rh, rw, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .to_device(&self.device)?;
        let raw_f32 = (raw_tensor.to_dtype(DType::F32)? * (1.0 / 255.0))?;
        let tensor = raw_f32.broadcast_sub(&self.img_mean)?.broadcast_div(&self.img_std)?
            .unsqueeze(0)?.to_dtype(self.dtype)?;

        let tensor = Tensor::cat(&[&tensor, &tensor], 0)?;

        let grid_t = 2usize / temporal_patch_size;
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

        let grid = Tensor::from_vec(
            vec![grid_t as u32, grid_h as u32, grid_w as u32],
            (1, 3),
            &self.device,
        )?;
        Ok((tensor, grid))
    }

    fn preprocess_images<P: AsRef<Path>>(
        &self,
        image_paths: &[P],
    ) -> Result<(Tensor, Tensor)> {
        let mut all_pixels = Vec::new();
        let mut all_grid_thw = Vec::new();

        for path in image_paths {
            let img = image::open(path.as_ref())?.to_rgb8();
            let (pixels, grid) = self.preprocess_one_image(img)?;
            all_pixels.push(pixels);
            all_grid_thw.push(grid);
        }

        let pixel_values = Tensor::cat(&all_pixels, 0)?;
        let grid_thw = Tensor::cat(&all_grid_thw, 0)?;
        Ok((pixel_values, grid_thw))
    }

    fn preprocess_images_from_bytes(
        &self,
        images_bytes: &[&[u8]],
    ) -> Result<(Tensor, Tensor)> {
        let mut all_pixels = Vec::new();
        let mut all_grid_thw = Vec::new();

        for bytes in images_bytes {
            let img = image::load_from_memory(bytes)
                .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?
                .to_rgb8();
            let (pixels, grid) = self.preprocess_one_image(img)?;
            all_pixels.push(pixels);
            all_grid_thw.push(grid);
        }

        let pixel_values = Tensor::cat(&all_pixels, 0)?;
        let grid_thw = Tensor::cat(&all_grid_thw, 0)?;
        Ok((pixel_values, grid_thw))
    }
}
