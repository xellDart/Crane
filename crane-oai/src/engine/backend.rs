//! Model backend abstraction for the inference engine.
//!
//! The [`ModelBackend`] trait decouples the engine from specific model implementations,
//! allowing any compatible LLM to be served through the OpenAI-compatible API.
//!
//! # Capability Levels
//!
//! | Capability        | Required | Effect when absent                            |
//! |-------------------|----------|-----------------------------------------------|
//! | `forward_step`    | Yes      | —                                             |
//! | KV cache swap     | No       | `max_concurrent` capped to 1                  |
//! | Batch decode      | No       | Sequences decoded sequentially per step        |

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

// ─────────────────────────────────────────────────────────────
//  Trait
// ─────────────────────────────────────────────────────────────

/// Core abstraction over different model backends.
///
/// All models must support single-sequence forward passes and KV cache clearing.
/// Optionally, models can support KV cache extraction/restoration (for concurrent
/// sequence serving) and batched decoding (for GPU-efficient parallel generation).
pub trait ModelBackend: Send + 'static {
    /// Run a forward pass for a single sequence.
    ///
    /// * `input_ids` — token IDs to process
    /// * `start_pos` — KV cache position (0 for a fresh sequence)
    ///
    /// Returns logits tensor, typically `[1, seq_len, vocab_size]`.
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor>;

    /// Clear all KV caches.
    fn clear_kv_cache(&mut self);

    /// Number of transformer layers (for KV cache vector sizing).
    fn num_layers(&self) -> usize;

    /// Device the model is running on.
    fn device(&self) -> &Device;

    /// Data type of model weights.
    fn dtype(&self) -> DType;

    /// Reference to the underlying tokenizer.
    fn tokenizer(&self) -> &tokenizers::Tokenizer;

    /// The model's end-of-sequence token ID(s).
    fn eos_token_id(&self) -> Vec<u32>;

    /// Warm up the model with a small forward pass.
    fn warmup(&mut self);

    // ── KV cache swap (for concurrent sequence serving) ───────

    /// Whether this backend supports extracting and restoring KV caches.
    fn supports_kv_swap(&self) -> bool {
        false
    }

    /// Extract per-layer KV caches from the model.
    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        vec![]
    }

    /// Restore per-layer KV caches into the model.
    fn set_kv_caches(&mut self, _caches: Vec<Option<(Tensor, Tensor)>>) {}

    /// Compute bytes held by the model's active KV caches without copying.
    /// Used for memory tracking without the overhead of `get_kv_caches()`.
    fn active_kv_cache_bytes(&self) -> u64 {
        0
    }


    // ── Batch decode (GPU-efficient concurrent serving) ───────

    /// Whether this backend supports batched decoding.
    fn supports_batch_decode(&self) -> bool {
        false
    }

    /// Pad and load per-sequence KV caches for batched decoding.
    fn setup_batch_decode(
        &mut self,
        _seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        _extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    /// Run one batched decode step.
    fn step_batch_decode(
        &mut self,
        _input_ids: &Tensor,
        _positions: &[usize],
        _attention_mask: Option<&Tensor>,
        _batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    /// Extract per-sequence KV caches from batched state.
    fn extract_batch_kv(
        &mut self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    /// Build attention mask for batched decoding.
    fn build_batch_decode_mask(
        &self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    // ── VLM support (for vision-language models) ──────────────────

    /// Whether this backend is a vision-language model.
    fn is_vlm(&self) -> bool {
        false
    }

    /// Run VLM prefill: preprocess images, run vision encoder, build prompt,
    /// compute M-RoPE, merge embeddings, forward.
    /// Returns `(logits, input_ids, next_gen_pos, prefill_len)`.
    fn prefill_vlm(
        &mut self,
        _images: &[Vec<u8>],
        _prompt: &str,
    ) -> Result<(Tensor, Vec<u32>, i64, usize)> {
        anyhow::bail!("VLM prefill not supported by this backend")
    }

    /// Set VLM decode state (gen_pos, prefill_len) after swap-in.
    fn set_vlm_decode_state(&mut self, _gen_pos: i64, _prefill_len: usize) {}

    /// Get current VLM decode state. Returns None for non-VLM backends.
    fn get_vlm_decode_state(&self) -> Option<(i64, usize)> {
        None
    }

    /// Set per-sequence VLM generation positions for the next batch decode step.
    /// Each entry is (gen_pos, prefill_len). Only used by VLM backends with batch decode.
    fn set_batch_vlm_positions(&mut self, _positions: &[(i64, usize)]) {}
}

// ─────────────────────────────────────────────────────────────
//  Hunyuan Dense Backend
// ─────────────────────────────────────────────────────────────

pub struct HunyuanBackend {
    pub model: crane_core::models::hunyuan_dense::Model,
}

impl HunyuanBackend {
    pub fn new(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: crane_core::models::hunyuan_dense::ModelFormat,
    ) -> Result<Self> {
        let model =
            crane_core::models::hunyuan_dense::Model::new_with_format(model_path, device, dtype, format)?;
        Ok(Self { model })
    }
}

impl ModelBackend for HunyuanBackend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.model.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        vec![120020]
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    // ── KV swap ──

    fn supports_kv_swap(&self) -> bool {
        true
    }

    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.model.get_kv_caches()
    }

    fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.model.set_kv_caches(caches);
    }

    fn active_kv_cache_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    // ── Batch decode ──

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.model.setup_batch_decode(seq_kv_caches, extra_room)
    }

    fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        self.model
            .step_batch_decode_with_input_ids(input_ids, positions, attention_mask, batch_kv_info)
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        self.model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        crane_core::models::hunyuan_dense::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen 2.5 Backend
// ─────────────────────────────────────────────────────────────

pub struct Qwen25Backend {
    pub model: crane_core::models::qwen25::Model,
    dtype: DType,
}

impl Qwen25Backend {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model = crane_core::models::qwen25::Model::new(model_path, device, dtype)?;
        Ok(Self {
            model,
            dtype: *dtype,
        })
    }
}

impl ModelBackend for Qwen25Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        0 // KV swap not supported; vector is unused
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        self.model
            .tokenizer
            .tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| self.model.tokenizer.tokenizer.token_to_id("<|im_end|>"))
            .map(|id| vec![id])
            .unwrap_or_else(|| vec![151643])
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen 3 Backend
// ─────────────────────────────────────────────────────────────

pub struct Qwen3Backend {
    pub model: crane_core::models::qwen3::Model,
    _dtype: DType,
}

impl Qwen3Backend {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model = crane_core::models::qwen3::Model::new(model_path, device, dtype)?;
        Ok(Self {
            model,
            _dtype: *dtype,
        })
    }
}

impl ModelBackend for Qwen3Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.model.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        // Qwen3 chat models stop at <|im_end|> (151645).
        // Also include <|endoftext|> (151643) as a fallback.
        let tok = &self.model.tokenizer.tokenizer;
        let mut ids = Vec::new();
        if let Some(id) = tok.token_to_id("<|im_end|>") { ids.push(id); }
        if let Some(id) = tok.token_to_id("<|endoftext|>") { ids.push(id); }
        if ids.is_empty() { ids.push(151645); ids.push(151643); }
        ids
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    // ── KV swap ──

    fn supports_kv_swap(&self) -> bool {
        true
    }

    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.model.get_kv_caches()
    }

    fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.model.set_kv_caches(caches);
    }

    fn active_kv_cache_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    // ── Batch decode ──

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.model.setup_batch_decode(seq_kv_caches, extra_room)
    }

    fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        self.model
            .step_batch_decode_with_input_ids(input_ids, positions, attention_mask, batch_kv_info)
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        self.model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        crane_core::models::qwen3::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen3-VL Backend (Vision-Language Model)
// ─────────────────────────────────────────────────────────────

pub struct Qwen3VLBackend {
    pub model: crane_core::models::qwen3_vl::Qwen3VL,
    /// Current M-RoPE generation position (tracked per active sequence).
    current_gen_pos: Option<i64>,
    /// Prefill length of the current active sequence.
    current_prefill_len: Option<usize>,
    /// Per-sequence M-RoPE generation positions for batch decode.
    batch_gen_positions: Vec<i64>,
}

impl Qwen3VLBackend {
    pub fn new(model: crane_core::models::qwen3_vl::Qwen3VL) -> Self {
        Self {
            model,
            current_gen_pos: None,
            current_prefill_len: None,
            batch_gen_positions: Vec::new(),
        }
    }
}

impl ModelBackend for Qwen3VLBackend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        // VLM decode: use start_pos as KV cache offset (engine computes this correctly),
        // but use tracked gen_pos for M-RoPE 3D positions.
        let gen_pos = self.current_gen_pos
            .ok_or_else(|| anyhow::anyhow!("VLM forward_step called without decode state"))?;

        assert_eq!(input_ids.len(), 1, "VLM decode expects single token");
        let logits = self.model.decode_step(input_ids[0], gen_pos, start_pos)?;
        self.current_gen_pos = Some(gen_pos + 1);
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        self.model.decoder_mut().clear_kv_cache();
        self.current_gen_pos = None;
        self.current_prefill_len = None;
    }

    fn num_layers(&self) -> usize {
        self.model.decoder().num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.model.model_dtype()
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        self.model.tokenizer_ref()
    }

    fn eos_token_id(&self) -> Vec<u32> {
        let tok = self.model.tokenizer_ref();
        let mut ids = Vec::new();
        if let Some(id) = tok.token_to_id("<|im_end|>") { ids.push(id); }
        if let Some(id) = tok.token_to_id("<|endoftext|>") { ids.push(id); }
        if ids.is_empty() { ids.push(151645); }
        ids
    }

    fn warmup(&mut self) {
        // VLM warmup: just clear caches (vision encoder warms up on first use).
        self.model.decoder_mut().clear_kv_cache();
    }

    // ── KV swap ──

    fn supports_kv_swap(&self) -> bool {
        true
    }

    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.model.decoder().get_kv_caches()
    }

    fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.model.decoder_mut().set_kv_caches(caches);
    }

    fn active_kv_cache_bytes(&self) -> u64 {
        self.model.decoder().active_kv_cache_bytes()
    }

    // ── VLM ──

    fn is_vlm(&self) -> bool {
        true
    }

    fn prefill_vlm(
        &mut self,
        images: &[Vec<u8>],
        prompt: &str,
    ) -> Result<(Tensor, Vec<u32>, i64, usize)> {
        let result = self.model.prefill_from_bytes(images, prompt)?;
        // Store decode state for subsequent forward_step calls.
        self.current_gen_pos = Some(result.2);
        self.current_prefill_len = Some(result.3);
        Ok(result)
    }

    fn set_vlm_decode_state(&mut self, gen_pos: i64, prefill_len: usize) {
        self.current_gen_pos = Some(gen_pos);
        self.current_prefill_len = Some(prefill_len);
    }

    fn get_vlm_decode_state(&self) -> Option<(i64, usize)> {
        match (self.current_gen_pos, self.current_prefill_len) {
            (Some(gp), Some(pl)) => Some((gp, pl)),
            _ => None,
        }
    }

    fn set_batch_vlm_positions(&mut self, positions: &[(i64, usize)]) {
        self.batch_gen_positions = positions.iter().map(|(gp, _)| *gp).collect();
    }

    // ── Batch decode ──

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.model.decoder_mut().setup_batch_decode(seq_kv_caches, extra_room)
    }

    fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        _positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        // Build M-RoPE 3D positions from batch_gen_positions.
        // During decode all 3 dims (temporal, height, width) use the same gen_pos.
        // Tensor shape is (3, N) row-major: [all_temporal, all_height, all_width].
        let n = self.batch_gen_positions.len();
        let device = &self.model.device;
        let pos_data: Vec<i64> = (0..3)
            .flat_map(|_| self.batch_gen_positions.iter().copied())
            .collect();
        let pos_3d = Tensor::from_vec(pos_data, (3, n), device)?;
        let (cos, sin) = self.model.mrope().forward(&pos_3d)?;
        // cos/sin are [N, half_dim], need [N, 1, half_dim] for attention
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;

        let logits = self.model.decoder_mut().step_batch_decode(
            input_ids,
            &cos,
            &sin,
            attention_mask,
            batch_kv_info,
        )?;

        // Auto-increment gen positions for next round.
        for gp in &mut self.batch_gen_positions {
            *gp += 1;
        }

        Ok(logits)
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        self.model.decoder_mut().extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        crane_core::models::qwen3::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}
