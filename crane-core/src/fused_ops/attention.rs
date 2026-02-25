//! Unified scaled dot-product attention with Flash Attention dispatch.
//!
//! When the `flash-attn` feature is enabled and the inputs are on CUDA in
//! BF16/F16 with no custom additive mask, this module dispatches to
//! FlashAttention-2 (via `candle-flash-attn`). Otherwise it falls back to
//! the standard matmul-based SDPA implementation.
//!
//! Flash Attention provides O(N) memory and 2-4x speedup for prefill
//! (seq_len > 512) compared to standard O(N^2) SDPA.

use candle_core::{Result, Tensor, D};
#[cfg(feature = "flash-attn")]
use candle_core::{DType, Device};

/// Check whether flash attention can be used for the given query tensor and mask.
///
/// Requirements:
/// - CUDA device
/// - BF16 or F16 dtype
/// - No custom additive mask (causal masking is handled internally by flash-attn)
#[cfg(feature = "flash-attn")]
fn can_use_flash_attn(q: &Tensor, mask: Option<&Tensor>) -> bool {
    let on_cuda = matches!(q.device(), Device::Cuda(_));
    let supported_dtype = matches!(q.dtype(), DType::BF16 | DType::F16);
    let no_mask = mask.is_none();
    on_cuda && supported_dtype && no_mask
}

/// Flash Attention forward pass.
///
/// Transposes from Crane's `(B, H, S, D)` layout to flash-attn's `(B, S, H, D)`,
/// calls the kernel, and transposes back.
///
/// GQA is handled natively by flash-attn (num_heads_q % num_heads_kv == 0).
#[cfg(feature = "flash-attn")]
fn flash_attn_forward(
    q: &Tensor, // [B, H, S_q, D]
    k: &Tensor, // [B, H_kv, S_kv, D]
    v: &Tensor, // [B, H_kv, S_kv, D]
    causal: bool,
) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    // flash_attn expects (B, S, H, D); we have (B, H, S, D)
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;

    let out = candle_flash_attn::flash_attn(&q, &k, &v, softmax_scale, causal)?;

    // Back to (B, H, S, D)
    out.transpose(1, 2)
}

/// Flash Attention for unbatched 3D tensors (e.g. vision encoder).
///
/// Input shape: `(S, H, D)` — unsqueezes batch dim, runs flash attn, squeezes back.
/// Always non-causal for vision attention.
#[cfg(feature = "flash-attn")]
fn flash_attn_3d(
    q: &Tensor, // [S, H, D]
    k: &Tensor, // [S, H_kv, D]
    v: &Tensor, // [S, H_kv, D]
) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    // flash_attn expects (B, S, H, D); add batch dim
    let q = q.unsqueeze(0)?;
    let k = k.unsqueeze(0)?;
    let v = v.unsqueeze(0)?;

    let out = candle_flash_attn::flash_attn(&q, &k, &v, softmax_scale, false)?;

    out.squeeze(0)
}

/// Standard manual SDPA implementation (fallback).
///
/// Handles GQA head expansion internally. Supports optional additive mask.
///
/// Input layout: `(B, H, S_q, D)` for q, `(B, H_kv, S_kv, D)` for k/v.
fn manual_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (head_dim as f64).sqrt();

    let num_heads = q.dim(1)?;
    let num_kv_heads = k.dim(1)?;
    let n_rep = num_heads / num_kv_heads;

    let k = if n_rep > 1 {
        let (b, kv_h, s, d) = k.dims4()?;
        k.unsqueeze(2)?
            .expand((b, kv_h, n_rep, s, d))?
            .reshape((b, kv_h * n_rep, s, d))?
    } else {
        k.clone()
    };
    let v = if n_rep > 1 {
        let (b, kv_h, s, d) = v.dims4()?;
        v.unsqueeze(2)?
            .expand((b, kv_h, n_rep, s, d))?
            .reshape((b, kv_h * n_rep, s, d))?
    } else {
        v.clone()
    };

    let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
    let attn = match mask {
        Some(m) => attn.broadcast_add(m)?,
        None => attn,
    };
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    attn.matmul(&v)
}

/// Scaled dot-product attention with automatic Flash Attention dispatch.
///
/// Inputs (4D, Crane layout):
///   - `q`: `[B, num_heads, seq_q, head_dim]`
///   - `k`: `[B, num_kv_heads, seq_kv, head_dim]` (already includes KV cache)
///   - `v`: `[B, num_kv_heads, seq_kv, head_dim]`
///   - `mask`: Optional additive mask `[B, 1, 1, seq_kv]` (for padding in batch decode)
///   - `causal`: Whether to apply causal mask (true for prefill, false for decode)
///
/// Returns: `[B, num_heads, seq_q, head_dim]`
///
/// When `flash-attn` feature is enabled and conditions are met (CUDA, BF16/F16,
/// no custom mask), dispatches to FlashAttention-2. Otherwise uses manual SDPA.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    #[allow(unused_variables)] causal: bool,
) -> Result<Tensor> {
    #[cfg(feature = "flash-attn")]
    {
        if can_use_flash_attn(q, mask) {
            return flash_attn_forward(q, k, v, causal);
        }
    }

    manual_sdpa(q, k, v, mask)
}

/// Scaled dot-product attention for unbatched 3D tensors (vision encoder).
///
/// Inputs (3D):
///   - `q`: `[S, num_heads, head_dim]`
///   - `k`: `[S, num_kv_heads, head_dim]`
///   - `v`: `[S, num_kv_heads, head_dim]`
///
/// Returns: `[S, num_heads, head_dim]`
///
/// Always non-causal. Uses Flash Attention when available.
pub fn scaled_dot_product_attention_3d(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "flash-attn")]
    {
        let on_cuda = matches!(q.device(), Device::Cuda(_));
        let supported_dtype = matches!(q.dtype(), DType::BF16 | DType::F16);
        if on_cuda && supported_dtype {
            return flash_attn_3d(q, k, v);
        }
    }

    // Fallback: transpose to (H, S, D) for matmul-based SDPA
    let head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (head_dim as f64).sqrt();

    let q = q.transpose(0, 1)?.contiguous()?;
    let k = k.transpose(0, 1)?.contiguous()?;
    let v = v.transpose(0, 1)?.contiguous()?;

    let attn = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    let out = attn.matmul(&v)?;

    // Back to (S, H, D)
    out.transpose(0, 1)
}
