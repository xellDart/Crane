//! Standalone validation of the FP8 W8A8 linear primitive vs candle bf16 matmul.
//! Checks correctness (cosine) + speed on the real MLP GEMM shapes.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn cos(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?;
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
    let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb + 1e-9))
}

fn bench(dev: &Device, name: &str, t: usize, in_dim: usize, out_dim: usize) -> Result<()> {
    // Random activations [t, in] and weight [out, in], bf16.
    let x = (Tensor::randn(0f32, 0.1, (t, in_dim), dev)?).to_dtype(DType::BF16)?;
    let wf = (Tensor::randn(0f32, 0.1, (out_dim, in_dim), dev)?).to_dtype(DType::BF16)?;

    // Reference: candle bf16 linear y = x @ w^T
    let y_ref = x.matmul(&wf.t()?.contiguous()?)?;

    // FP8 path
    let (w_fp8, w_scale) = crane_core::fused_ops::fp8::quantize_weight_e4m3(&wf)?;
    let y_fp8 = crane_core::fused_ops::fp8::fp8_linear(&x, &w_fp8, w_scale)?;

    let c = cos(&y_ref, &y_fp8)?;

    // timing
    let iters = 50;
    dev.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = x.matmul(&wf.t()?.contiguous()?)?;
    }
    dev.synchronize()?;
    let bf_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    dev.synchronize()?;
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = crane_core::fused_ops::fp8::fp8_linear(&x, &w_fp8, w_scale)?;
    }
    dev.synchronize()?;
    let fp8_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

    println!(
        "{:<22} T={} in={} out={} | bf16 {:.3} ms | fp8 {:.3} ms | speedup {:.2}x | cos {:.5}",
        name, t, in_dim, out_dim, bf_ms, fp8_ms, bf_ms / fp8_ms, c
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("=== FP8 linear vs candle bf16 (incl. activation quant) ===");
    let t = 1481;
    bench(&dev, "Vultron mlp gate_up", t, 4096, 24576)?;
    bench(&dev, "Vultron mlp down", t, 12288, 4096)?;
    bench(&dev, "ops mlp gate_up", t, 2560, 19456)?;
    bench(&dev, "ops mlp down", t, 9728, 2560)?;
    Ok(())
}
