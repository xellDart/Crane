//! CUDA implementations of fused ops using custom PTX kernels.
//!
//! The PTX is compiled from `kernels/fused_ops.cu` at build time via
//! bindgen_cuda and embedded as a const string.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use candle_core::{DType, Device, Layout, Result, Shape, Tensor, WithDType};

// PTX compiled from kernels/fused_ops.cu — embedded at build time.
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "crane_fused_ops";

/// Load a function from the crane fused-ops PTX module.
///
/// Returns candle's `CudaFunc` opaque wrapper (not re-exported in candle 0.9.x,
/// so we avoid naming the type explicitly — callers rely on type inference).
macro_rules! load_func {
    ($dev:expr, $fn_name:expr) => {
        $dev.get_or_load_custom_func($fn_name, MODULE_NAME, ptx::FUSED_OPS)
    };
}

// =====================================================================
// 1. Fused SiLU(gate) * up
// =====================================================================

/// Fused SiLU activation + element-wise multiply.
///
/// Takes a `gate_up` tensor of shape `[..., 2*intermediate_size]`
/// (gate and up projections concatenated along the last dim) and returns
/// `silu(gate) * up` of shape `[..., intermediate_size]`.
///
/// Replaces 3 candle ops: `narrow(gate)` + `silu(gate)` + `gate * up`.
pub struct FusedSiluMul {
    pub intermediate_size: usize,
}

impl candle_core::CustomOp1 for FusedSiluMul {
    fn name(&self) -> &'static str {
        "fused-silu-mul"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        // CPU fallback — just do it the slow way.
        use candle_core::CpuStorage as C;

        fn inner<T: WithDType>(
            src: &[T],
            layout: &Layout,
            intermediate_size: usize,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle_core::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let dims = layout.shape().dims();
            let last = *dims.last().unwrap();
            if last != 2 * intermediate_size {
                candle_core::bail!(
                    "last dim {last} != 2*intermediate_size {}",
                    2 * intermediate_size
                );
            }
            let n_rows = src.len() / last;
            let mut dst = vec![T::zero(); n_rows * intermediate_size];
            for row in 0..n_rows {
                let gate = &src[row * last..row * last + intermediate_size];
                let up = &src[row * last + intermediate_size..row * last + last];
                let out = &mut dst[row * intermediate_size..(row + 1) * intermediate_size];
                for i in 0..intermediate_size {
                    let g: f64 = gate[i].to_f64();
                    let u: f64 = up[i].to_f64();
                    let silu_g = g / (1.0 + (-g).exp());
                    out[i] = T::from_f64(silu_g * u);
                }
            }
            let mut out_dims = dims.to_vec();
            *out_dims.last_mut().unwrap() = intermediate_size;
            let storage = T::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(&out_dims)))
        }

        match storage {
            C::BF16(s) => inner(s, layout, self.intermediate_size),
            C::F16(s) => inner(s, layout, self.intermediate_size),
            C::F32(s) => inner(s, layout, self.intermediate_size),
            C::F64(s) => inner(s, layout, self.intermediate_size),
            _ => candle_core::bail!("unsupported dtype for fused_silu_mul"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = storage.device();
        let dims = layout.shape().dims();
        let last = *dims.last().unwrap();
        let intermediate_size = self.intermediate_size;

        if last != 2 * intermediate_size {
            candle_core::bail!(
                "fused_silu_mul: last dim {last} != 2*intermediate_size {}",
                2 * intermediate_size
            );
        }

        let (o1, o2) = match layout.contiguous_offsets() {
            None => candle_core::bail!("fused_silu_mul: input must be contiguous"),
            Some(offsets) => offsets,
        };

        let n_rows = (o2 - o1) / last;
        let out_el = n_rows * intermediate_size;

        // Choose kernel name and launch
        let fn_name = match storage.dtype() {
            DType::BF16 => "fused_silu_mul_bf16",
            DType::F16 => "fused_silu_mul_f16",
            DType::F32 => "fused_silu_mul_f32",
            dt => candle_core::bail!("fused_silu_mul: unsupported dtype {dt:?}"),
        };
        let func = load_func!(dev, fn_name)?;

        let block_size = 1024u32.min(intermediate_size as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let slice = match &storage.slice {
            CudaStorageSlice::BF16(s) => {
                let s = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<half::bf16>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&s);
                builder.arg(&dst);
                let isize_i32 = intermediate_size as i32;
                builder.arg(&isize_i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            }
            CudaStorageSlice::F16(s) => {
                let s = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<half::f16>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&s);
                builder.arg(&dst);
                let isize_i32 = intermediate_size as i32;
                builder.arg(&isize_i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            }
            CudaStorageSlice::F32(s) => {
                let s = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<f32>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&s);
                builder.arg(&dst);
                let isize_i32 = intermediate_size as i32;
                builder.arg(&isize_i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            }
            _ => candle_core::bail!("fused_silu_mul: unsupported storage type"),
        };

        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = intermediate_size;
        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, Shape::from_dims(&out_dims)))
    }
}

/// Convenience function: fused SiLU(gate) * up.
///
/// `gate_up` must have shape `[..., 2*intermediate_size]` and be contiguous.
pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    gate_up.apply_op1_no_bwd(&FusedSiluMul { intermediate_size })
}

// =====================================================================
// 2. Fused residual_add + RMSNorm
// =====================================================================

/// Fused residual addition + RMSNorm in one kernel pass.
///
/// Computes:
///   `sum = residual + hidden`
///   `normalized = rmsnorm(sum, weight, eps)`
///
/// Returns `(sum, normalized)`. Both outputs are written in a single
/// kernel pass, halving memory reads compared to separate add + rmsnorm.
///
/// On CUDA with BF16: dispatches to `fused_add_rmsnorm_out_bf16` kernel.
/// Otherwise: falls back to separate candle ops.
pub fn fused_add_rmsnorm(
    residual: &Tensor,
    hidden: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        if residual.dtype() == DType::BF16
            && residual.device().is_cuda()
            && residual.is_contiguous()
        {
            return fused_add_rmsnorm_cuda(residual, hidden, weight, eps as f32);
        }
    }

    // CPU / non-BF16 fallback: separate add + rmsnorm
    let sum = (residual + hidden)?;
    let norm = candle_nn::RmsNorm::new(weight.clone(), eps);
    let normalized = candle_core::Module::forward(&norm, &sum)?;
    Ok((sum, normalized))
}

#[cfg(feature = "cuda")]
fn fused_add_rmsnorm_cuda(
    residual: &Tensor,
    hidden: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let hidden = hidden.contiguous()?;
    let weight = weight.contiguous()?;

    let dims = residual.dims();
    let ncols = dims[dims.len() - 1];
    let nrows: usize = dims[..dims.len() - 1].iter().product();

    let dev = match residual.device() {
        Device::Cuda(d) => d,
        _ => unreachable!(),
    };

    let func = load_func!(dev, "fused_add_rmsnorm_out_bf16")?;
    let block_size = 1024u32.min(ncols as u32);
    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Pre-allocate output tensors
    let sum_tensor = Tensor::zeros(residual.shape(), DType::BF16, residual.device())?;
    let norm_tensor = Tensor::zeros(residual.shape(), DType::BF16, residual.device())?;

    // Launch fused kernel — writes into sum_tensor and norm_tensor's
    // backing memory. Same write-through-shared-storage pattern as slice_set.
    {
        let (res_guard, res_layout) = residual.storage_and_layout();
        let (hid_guard, hid_layout) = hidden.storage_and_layout();
        let (wt_guard, wt_layout) = weight.storage_and_layout();
        let (sum_guard, _) = sum_tensor.storage_and_layout();
        let (norm_guard, _) = norm_tensor.storage_and_layout();

        let res_cuda = match &*res_guard {
            candle_core::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        let hid_cuda = match &*hid_guard {
            candle_core::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        let wt_cuda = match &*wt_guard {
            candle_core::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        let sum_cuda = match &*sum_guard {
            candle_core::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        let norm_cuda = match &*norm_guard {
            candle_core::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };

        match (
            &res_cuda.slice,
            &hid_cuda.slice,
            &wt_cuda.slice,
            &sum_cuda.slice,
            &norm_cuda.slice,
        ) {
            (
                CudaStorageSlice::BF16(res_s),
                CudaStorageSlice::BF16(hid_s),
                CudaStorageSlice::BF16(wt_s),
                CudaStorageSlice::BF16(sum_s),
                CudaStorageSlice::BF16(norm_s),
            ) => {
                let res_s = res_s.slice(res_layout.start_offset()..);
                let hid_s = hid_s.slice(hid_layout.start_offset()..);
                let wt_s = wt_s.slice(wt_layout.start_offset()..);

                let mut builder = func.builder();
                builder.arg(&res_s); // residual (read)
                builder.arg(&hid_s); // hidden (read)
                builder.arg(sum_s); // sum output (write)
                builder.arg(norm_s); // norm output (write)
                builder.arg(&wt_s); // weight (read)
                let ncols_i32 = ncols as i32;
                builder.arg(&ncols_i32);
                builder.arg(&eps);
                unsafe { builder.launch(cfg) }.w()?;
            }
            _ => candle_core::bail!("fused_add_rmsnorm: all inputs must be BF16"),
        }
    }

    Ok((sum_tensor, norm_tensor))
}

// =====================================================================
// 3. GPU Argmax — greedy decode without DtoH logits transfer
// =====================================================================

/// Perform argmax on the GPU, returning only the index (4 bytes DtoH
/// instead of the full vocab_size * 2 bytes).
///
/// `logits` shape: `[1, 1, vocab_size]` or `[1, vocab_size]` or `[vocab_size]`
/// Returns the token index as u32.
#[cfg(feature = "cuda")]
pub fn gpu_argmax(logits: &Tensor) -> Result<u32> {
    let device = logits.device();
    let dev = match device {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("gpu_argmax requires CUDA device"),
    };

    let logits = logits.contiguous()?.flatten_all()?;
    let vocab_size = logits.elem_count();

    // Get the underlying storage
    let (storage, layout) = logits.storage_and_layout();
    let cuda_storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("gpu_argmax: expected CUDA storage"),
    };

    let (o1, _o2) = match layout.contiguous_offsets() {
        Some(o) => o,
        None => candle_core::bail!("gpu_argmax: logits must be contiguous"),
    };

    // Phase 1: per-block reduction
    let num_blocks = 256u32.min((vocab_size as u32 + 1023) / 1024);
    let block_size = 256u32;

    let func1 = load_func!(dev, "gpu_argmax_bf16_phase1")?;
    let func2 = load_func!(dev, "gpu_argmax_phase2")?;

    // Allocate temporary buffers for block results
    let block_max_vals: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32> =
        unsafe { dev.alloc::<f32>(num_blocks as usize)? };
    let block_max_idxs: candle_core::cuda_backend::cudarc::driver::CudaSlice<i32> =
        unsafe { dev.alloc::<i32>(num_blocks as usize)? };
    let output_token: candle_core::cuda_backend::cudarc::driver::CudaSlice<i32> =
        unsafe { dev.alloc::<i32>(1)? };

    // Phase 1 launch
    let cfg1 = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    match &cuda_storage.slice {
        CudaStorageSlice::BF16(s) => {
            let s = s.slice(o1..);
            let mut builder = func1.builder();
            builder.arg(&s);
            builder.arg(&block_max_vals);
            builder.arg(&block_max_idxs);
            let vs = vocab_size as i32;
            builder.arg(&vs);
            unsafe { builder.launch(cfg1) }.w()?;
        }
        _ => candle_core::bail!("gpu_argmax currently only supports BF16"),
    }

    // Phase 2: reduce block results
    let cfg2 = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (num_blocks.min(256), 1, 1),
        shared_mem_bytes: 0,
    };

    {
        let mut builder = func2.builder();
        builder.arg(&block_max_vals);
        builder.arg(&block_max_idxs);
        builder.arg(&output_token);
        let nb = num_blocks as i32;
        builder.arg(&nb);
        unsafe { builder.launch(cfg2) }.w()?;
    }

    // DtoH: only 4 bytes!
    let result = dev.clone_dtoh(&output_token)?;
    Ok(result[0] as u32)
}

// =====================================================================
// 4. GPU TopK — returns indices of the top-k largest values
// =====================================================================

#[cfg(feature = "cuda")]
thread_local! {
    static TOPK_TMP: std::cell::RefCell<
        std::collections::HashMap<(candle_core::cuda_backend::DeviceId, usize), TopkTmpBufs>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[cfg(feature = "cuda")]
struct TopkTmpBufs {
    vals: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    idx: candle_core::cuda_backend::cudarc::driver::CudaSlice<u32>,
    cap_elems: usize,
}

/// GPU top-k indices for 1D f32 tensors (k ≤ 64).
///
/// Two-stage block reduction using custom CUDA kernels compiled from
/// `crane-core/kernels/fused_ops.cu`.
///
/// Returns a `[k]` U32 tensor of the indices of the k largest values,
/// sorted in descending order of value.
#[cfg(feature = "cuda")]
pub fn topk_indices(logits: &Tensor, k: usize) -> Result<Tensor> {
    if !logits.is_contiguous() {
        candle_core::bail!("topk_indices requires contiguous input");
    }
    if logits.rank() != 1 {
        candle_core::bail!("topk_indices expects a 1D tensor");
    }
    if k == 0 || k > 64 {
        candle_core::bail!("topk_indices expects 0 < k <= 64");
    }
    let n = logits.dims1()?;
    if k > n {
        candle_core::bail!("topk_indices expects k <= n");
    }
    logits.apply_op1_no_bwd(&TopKIndicesOp { k })
}

#[cfg(feature = "cuda")]
struct TopKIndicesOp {
    k: usize,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for TopKIndicesOp {
    fn name(&self) -> &'static str {
        "topk_indices"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("topk_indices requires contiguous layout");
        }
        let k = self.k;
        let n = layout.shape().elem_count();
        let start = layout.start_offset();
        let end = start + n;

        let mut pairs: Vec<(f32, u32)> = match storage {
            candle_core::CpuStorage::F32(vs) => vs[start..end]
                .iter()
                .enumerate()
                .map(|(i, &v)| (v, i as u32))
                .collect(),
            _ => candle_core::bail!("topk_indices only supports f32"),
        };

        let kth = k.saturating_sub(1);
        pairs.select_nth_unstable_by(kth, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Greater)
        });
        pairs.truncate(k);
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Greater));

        let out: Vec<u32> = pairs.into_iter().map(|(_, i)| i).collect();
        Ok((candle_core::CpuStorage::U32(out), Shape::from_dims(&[k])))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &candle_core::Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;

        if !layout.is_contiguous() {
            candle_core::bail!("topk_indices requires contiguous layout");
        }
        let k = self.k;
        let k_u32 = k as u32;
        let n = layout.shape().elem_count();
        let n_u32 = n as u32;
        let dev = &storage.device;

        let x = storage.as_cuda_slice::<f32>()?;
        let (o1, o2) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("topk: need contiguous offsets".into()))?;
        let x = x.slice(o1..o2);

        let block_dim1 = 128u32;
        let block_dim2 = 128u32;
        let items_per_block = (block_dim1 as usize) * 8;
        let grid = ((n + items_per_block - 1) / items_per_block).clamp(1, 1024);
        let grid_dim = grid as u32;
        let shared1 =
            block_dim1 as usize * k * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());
        let shared2 =
            block_dim2 as usize * k * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());

        let cap_elems = grid * k;
        let dev_id = dev.id();
        let (tmp_vals, tmp_idx) = TOPK_TMP.with(|cell| -> Result<_> {
            let mut map = cell.borrow_mut();
            match map.get_mut(&(dev_id, k)) {
                Some(bufs) if bufs.cap_elems >= cap_elems => {
                    Ok((bufs.vals.clone(), bufs.idx.clone()))
                }
                _ => {
                    let vals = unsafe { dev.alloc::<f32>(cap_elems)? };
                    let idx = unsafe { dev.alloc::<u32>(cap_elems)? };
                    map.insert(
                        (dev_id, k),
                        TopkTmpBufs {
                            vals: vals.clone(),
                            idx: idx.clone(),
                            cap_elems,
                        },
                    );
                    Ok((vals, idx))
                }
            }
        })?;

        let out_idx = unsafe { dev.alloc::<u32>(k)? };

        // Stage 1
        let f1 = load_func!(dev, "topk_stage1_f32")?;
        let items_per_block_u32 = items_per_block as u32;
        {
            let mut builder = f1.builder();
            builder.arg(&x);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder.arg(&items_per_block_u32);
            builder.arg(&tmp_vals);
            builder.arg(&tmp_idx);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (block_dim1, 1, 1),
                    shared_mem_bytes: shared1 as u32,
                })
            }
            .w()?;
        }

        // Stage 2
        let m = grid_dim * k_u32;
        let f2 = load_func!(dev, "topk_stage2_f32")?;
        {
            let mut builder = f2.builder();
            builder.arg(&tmp_vals);
            builder.arg(&tmp_idx);
            builder.arg(&m);
            builder.arg(&k_u32);
            builder.arg(&out_idx);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_dim2, 1, 1),
                    shared_mem_bytes: shared2 as u32,
                })
            }
            .w()?;
        }

        let dst = CudaStorage::wrap_cuda_slice(out_idx, dev.clone());
        Ok((dst, Shape::from_dims(&[k])))
    }
}

// =====================================================================
// 5. CUDA tensor memory utilities
// =====================================================================

/// Copy a u32 slice from host to the target device, returning a new 1-D U32 tensor.
///
/// The returned tensor has shape `[src.len()]` on the same device as `device`.
/// This is a plain HtoD allocation — no kernel launch.
#[cfg(feature = "cuda")]
pub fn copy_from_slice_u32(src: &[u32], device: &Device) -> Result<Tensor> {
    Tensor::new(src, device)
}

/// Clone a contiguous f32 tensor — returns a new contiguous copy on the same device.
///
/// For CUDA tensors this is a DtoD copy (no host round-trip).
#[cfg(feature = "cuda")]
pub fn copy_from_tensor_f32(src_tensor: &Tensor) -> Result<Tensor> {
    if src_tensor.dtype() != DType::F32 {
        candle_core::bail!(
            "copy_from_tensor_f32: expected f32 tensor, got {:?}",
            src_tensor.dtype()
        );
    }
    src_tensor.contiguous()
}

// =====================================================================
// Depthwise causal conv1d + SiLU
// =====================================================================

/// Depthwise causal conv1d (kernel_size = `weight.dim(1)`) followed by SiLU.
/// `x`: (B, C, T) ; `weight`: (C, KS), same dtype (BF16/F32). Returns (B, C, T).
/// One CUDA kernel instead of a per-timestep window `stack` of T slices.
pub fn causal_conv1d_silu(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let (b, c, t) = x.dims3()?;
    let ks = weight.dim(1)?;
    let total = (b * c * t) as i64;

    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("causal_conv1d_silu: CUDA only"),
    };
    let fn_name = match x.dtype() {
        DType::BF16 => "causal_conv1d_silu_bf16",
        DType::F32 => "causal_conv1d_silu_f32",
        dt => candle_core::bail!("causal_conv1d_silu: unsupported dtype {dt:?}"),
    };
    let func = load_func!(&dev, fn_name)?;

    let threads = 256u32;
    let blocks = (((total as u64 + threads as u64 - 1) / threads as u64).min(65_535)) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let out = Tensor::zeros((b, c, t), x.dtype(), x.device())?;
    {
        let (x_g, x_l) = x.storage_and_layout();
        let (w_g, w_l) = weight.storage_and_layout();
        let (o_g, _) = out.storage_and_layout();
        let x_cuda = match &*x_g {
            candle_core::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        let w_cuda = match &*w_g {
            candle_core::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        let o_cuda = match &*o_g {
            candle_core::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        let (c_i32, t_i32, ks_i32) = (c as i32, t as i32, ks as i32);

        match (&x_cuda.slice, &w_cuda.slice, &o_cuda.slice) {
            (
                CudaStorageSlice::BF16(xs),
                CudaStorageSlice::BF16(ws),
                CudaStorageSlice::BF16(os),
            ) => {
                let xs = xs.slice(x_l.start_offset()..);
                let ws = ws.slice(w_l.start_offset()..);
                let mut builder = func.builder();
                builder.arg(&xs);
                builder.arg(&ws);
                builder.arg(os);
                builder.arg(&total);
                builder.arg(&c_i32);
                builder.arg(&t_i32);
                builder.arg(&ks_i32);
                unsafe { builder.launch(cfg) }.w()?;
            }
            (CudaStorageSlice::F32(xs), CudaStorageSlice::F32(ws), CudaStorageSlice::F32(os)) => {
                let xs = xs.slice(x_l.start_offset()..);
                let ws = ws.slice(w_l.start_offset()..);
                let mut builder = func.builder();
                builder.arg(&xs);
                builder.arg(&ws);
                builder.arg(os);
                builder.arg(&total);
                builder.arg(&c_i32);
                builder.arg(&t_i32);
                builder.arg(&ks_i32);
                unsafe { builder.launch(cfg) }.w()?;
            }
            _ => candle_core::bail!("causal_conv1d_silu: x/weight/out dtype mismatch"),
        }
    }
    Ok(out)
}

// =====================================================================
// Chunked delta-rule intra-chunk inverse (forward substitution)
// =====================================================================

/// Per-group forward substitution + identity for the chunked gated delta rule.
/// `a`: (G, C, C) f32, strictly-lower. Returns (G, C, C) f32 = inverse factor + I.
/// One CUDA block per group does the C-step substitution in shared memory,
/// replacing a 63-step loop of full-tensor slice_assigns.
pub fn chunk_delta_invert(a: &Tensor, g: usize, c: usize) -> Result<Tensor> {
    let a = a.contiguous()?;
    if a.dtype() != DType::F32 {
        candle_core::bail!("chunk_delta_invert: f32 only");
    }
    let dev = match a.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("chunk_delta_invert: CUDA only"),
    };
    let func = load_func!(&dev, "chunk_delta_invert_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (g as u32, 1, 1),
        block_dim: (c as u32, 1, 1),
        shared_mem_bytes: (c * c * 4) as u32,
    };
    let out = Tensor::zeros((g, c, c), DType::F32, a.device())?;
    {
        let (a_g, a_l) = a.storage_and_layout();
        let (o_g, _) = out.storage_and_layout();
        let a_cuda = match &*a_g {
            candle_core::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        let o_cuda = match &*o_g {
            candle_core::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        match (&a_cuda.slice, &o_cuda.slice) {
            (CudaStorageSlice::F32(a_s), CudaStorageSlice::F32(o_s)) => {
                let a_s = a_s.slice(a_l.start_offset()..);
                let c_i32 = c as i32;
                let mut builder = func.builder();
                builder.arg(&a_s);
                builder.arg(o_s);
                builder.arg(&c_i32);
                unsafe { builder.launch(cfg) }.w()?;
            }
            _ => candle_core::bail!("chunk_delta_invert: f32 only"),
        }
    }
    Ok(out)
}

// =====================================================================
// Fused gated RMSNorm: rmsnorm(x)*weight * silu(gate)
// =====================================================================

/// `out = rmsnorm(x, weight, eps) * silu(gate)`, one kernel pass.
/// `x` (rows, D) is upcast to f32; `gate`/`weight` are BF16; output BF16.
pub fn fused_rmsnorm_gated(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let gate = gate.to_dtype(DType::BF16)?.contiguous()?;
    let weight = weight.to_dtype(DType::BF16)?.contiguous()?;
    let dims = x.dims();
    let d = dims[dims.len() - 1];
    let rows: usize = dims[..dims.len() - 1].iter().product();

    let dev = match x.device() {
        Device::Cuda(dv) => dv.clone(),
        _ => candle_core::bail!("fused_rmsnorm_gated: CUDA only"),
    };
    let func = load_func!(&dev, "fused_rmsnorm_gated")?;
    let block = 128u32.min(d as u32).max(32);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let out = Tensor::zeros(x.shape(), DType::BF16, x.device())?;
    {
        let (x_g, x_l) = x.storage_and_layout();
        let (g_g, g_l) = gate.storage_and_layout();
        let (w_g, w_l) = weight.storage_and_layout();
        let (o_g, _) = out.storage_and_layout();
        let xc = match &*x_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let gc = match &*g_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let wc = match &*w_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let oc = match &*o_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        match (&xc.slice, &gc.slice, &wc.slice, &oc.slice) {
            (
                CudaStorageSlice::F32(xs),
                CudaStorageSlice::BF16(gs),
                CudaStorageSlice::BF16(ws),
                CudaStorageSlice::BF16(os),
            ) => {
                let xs = xs.slice(x_l.start_offset()..);
                let gs = gs.slice(g_l.start_offset()..);
                let ws = ws.slice(w_l.start_offset()..);
                let (d_i32, eps_f) = (d as i32, eps);
                let mut builder = func.builder();
                builder.arg(&xs);
                builder.arg(&gs);
                builder.arg(&ws);
                builder.arg(os);
                builder.arg(&d_i32);
                builder.arg(&eps_f);
                unsafe { builder.launch(cfg) }.w()?;
            }
            _ => candle_core::bail!("fused_rmsnorm_gated: dtype mismatch"),
        }
    }
    Ok(out)
}

// =====================================================================
// FP8 W8A8 quantization: bf16 -> E4M3 with scalar reciprocal scale
// =====================================================================

/// Quantize a bf16 tensor to F8E4M3: `out = e4m3(x * inv)`. `inv = 448/absmax`.
/// Returns a contiguous F8E4M3 tensor of the same shape.
pub fn quantize_e4m3(x: &Tensor, inv: f32) -> Result<Tensor> {
    let x = x.to_dtype(DType::BF16)?.contiguous()?;
    let n = x.elem_count() as i64;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("quantize_e4m3: CUDA only"),
    };
    let func = load_func!(&dev, "quant_e4m3_bf16")?;
    let threads = 256u32;
    let blocks = (((n as u64 + threads as u64 - 1) / threads as u64).min(65_535)) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let out = Tensor::zeros(x.shape(), DType::F8E4M3, x.device())?;
    {
        let (x_g, x_l) = x.storage_and_layout();
        let (o_g, _) = out.storage_and_layout();
        let xc = match &*x_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let oc = match &*o_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        match (&xc.slice, &oc.slice) {
            (CudaStorageSlice::BF16(xs), CudaStorageSlice::F8E4M3(os)) => {
                let xs = xs.slice(x_l.start_offset()..);
                let mut builder = func.builder();
                builder.arg(&xs);
                builder.arg(os);
                builder.arg(&inv);
                builder.arg(&n);
                unsafe { builder.launch(cfg) }.w()?;
            }
            _ => candle_core::bail!("quantize_e4m3: expected bf16 in, f8e4m3 out"),
        }
    }
    Ok(out)
}

/// Per-tensor absmax of a tensor as f32 (host scalar). Reduces in the input dtype
/// (avoids a full F32 upcast copy of the activations) and upcasts only the scalar.
pub fn absmax_f32(x: &Tensor) -> Result<f32> {
    let m = x.abs()?.flatten_all()?.max(0)?;
    m.to_dtype(DType::F32)?.to_scalar::<f32>()
}

/// Fully on-device FP8 activation quant (no host sync): computes per-tensor absmax
/// of bf16 `x`, quantizes to E4M3, and writes the activation scale into `a_scale`
/// (a `[1]` f32 tensor read by cuBLASLt). `amax` is a `[1]` f32 scratch tensor.
/// Returns a contiguous F8E4M3 tensor of the same shape as `x`. All on one stream.
pub fn quantize_activation_dev(x: &Tensor, amax: &Tensor, a_scale: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::BF16)?.contiguous()?;
    let n = x.elem_count() as i64;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("quantize_activation_dev: CUDA only"),
    };
    let out = Tensor::zeros(x.shape(), DType::F8E4M3, x.device())?;

    let f_zero = load_func!(&dev, "set_zero_f32")?;
    let f_amax = load_func!(&dev, "absmax_bf16")?;
    let f_quant = load_func!(&dev, "quant_e4m3_dev")?;

    let threads = 256u32;
    let blocks = (((n as u64 + threads as u64 - 1) / threads as u64).min(65_535)) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg1 = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Storage guards scoped so they release before `out` is returned.
    {
        let (x_g, x_l) = x.storage_and_layout();
        let (o_g, _) = out.storage_and_layout();
        let (am_g, _) = amax.storage_and_layout();
        let (as_g, _) = a_scale.storage_and_layout();
        let xc = match &*x_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let oc = match &*o_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let amc = match &*am_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let asc = match &*as_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let xs = match &xc.slice { CudaStorageSlice::BF16(s) => s.slice(x_l.start_offset()..), _ => candle_core::bail!("qad: x bf16") };
        let os = match &oc.slice { CudaStorageSlice::F8E4M3(s) => s, _ => candle_core::bail!("qad: out f8") };
        let am = match &amc.slice { CudaStorageSlice::F32(s) => s, _ => candle_core::bail!("qad: amax f32") };
        let asf = match &asc.slice { CudaStorageSlice::F32(s) => s, _ => candle_core::bail!("qad: a_scale f32") };

        // 1. zero amax
        { let mut b = f_zero.builder(); b.arg(am); unsafe { b.launch(cfg1) }.w()?; }
        // 2. absmax
        { let mut b = f_amax.builder(); b.arg(&xs); b.arg(am); b.arg(&n); unsafe { b.launch(cfg) }.w()?; }
        // 3. quantize + emit scale
        { let mut b = f_quant.builder(); b.arg(&xs); b.arg(os); b.arg(am); b.arg(asf); b.arg(&n); unsafe { b.launch(cfg) }.w()?; }
    }
    Ok(out)
}

// =====================================================================
// GDN scan glue: chunk decay mask + negated strict-lower masked product
// =====================================================================

/// `decay_mask[g,i,j] = (i>=j) ? exp(g_cs[g,i] - g_cs[g,j]) : 0`, one kernel pass.
/// `g_cs`: (G, C) f32 within-chunk cumulative log-decay. Returns (G, C, C) f32.
/// Fuses candle's broadcast_sub -> mask -> exp -> mask chain (drops the (G,C,C)
/// `diff` intermediate).
pub fn gdn_decay_mask(g_cs: &Tensor, g: usize, c: usize) -> Result<Tensor> {
    let g_cs = g_cs.contiguous()?;
    if g_cs.dtype() != DType::F32 {
        candle_core::bail!("gdn_decay_mask: f32 only");
    }
    let dev = match g_cs.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("gdn_decay_mask: CUDA only"),
    };
    let func = load_func!(&dev, "gdn_decay_mask_f32")?;
    let total = (g * c * c) as i64;
    let threads = 256u32;
    let blocks = (((total as u64 + threads as u64 - 1) / threads as u64).min(65_535)) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let out = Tensor::zeros((g, c, c), DType::F32, g_cs.device())?;
    {
        let (x_g, x_l) = g_cs.storage_and_layout();
        let (o_g, _) = out.storage_and_layout();
        let xc = match &*x_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let oc = match &*o_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        match (&xc.slice, &oc.slice) {
            (CudaStorageSlice::F32(xs), CudaStorageSlice::F32(os)) => {
                let xs = xs.slice(x_l.start_offset()..);
                let c_i32 = c as i32;
                let mut builder = func.builder();
                builder.arg(&xs);
                builder.arg(os);
                builder.arg(&total);
                builder.arg(&c_i32);
                unsafe { builder.launch(cfg) }.w()?;
            }
            _ => candle_core::bail!("gdn_decay_mask: f32 only"),
        }
    }
    Ok(out)
}

/// `out[g,i,j] = (i>j) ? -(kk[g,i,j] * decay[g,i,j]) : 0`, one kernel pass.
/// `kk`, `decay`: (G, C, C) f32. Fuses candle's broadcast_mul -> neg -> broadcast_mul.
pub fn gdn_neg_lower_mul(kk: &Tensor, decay: &Tensor) -> Result<Tensor> {
    let kk = kk.contiguous()?;
    let decay = decay.contiguous()?;
    if kk.dtype() != DType::F32 || decay.dtype() != DType::F32 {
        candle_core::bail!("gdn_neg_lower_mul: f32 only");
    }
    let (g, c, c2) = kk.dims3()?;
    if c != c2 {
        candle_core::bail!("gdn_neg_lower_mul: kk must be (G,C,C)");
    }
    let dev = match kk.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("gdn_neg_lower_mul: CUDA only"),
    };
    let func = load_func!(&dev, "gdn_neg_lower_mul_f32")?;
    let total = (g * c * c) as i64;
    let threads = 256u32;
    let blocks = (((total as u64 + threads as u64 - 1) / threads as u64).min(65_535)) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let out = Tensor::zeros((g, c, c), DType::F32, kk.device())?;
    {
        let (k_g, k_l) = kk.storage_and_layout();
        let (d_g, d_l) = decay.storage_and_layout();
        let (o_g, _) = out.storage_and_layout();
        let kc = match &*k_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let dc = match &*d_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        let oc = match &*o_g { candle_core::Storage::Cuda(s) => s, _ => unreachable!() };
        match (&kc.slice, &dc.slice, &oc.slice) {
            (CudaStorageSlice::F32(ks), CudaStorageSlice::F32(ds), CudaStorageSlice::F32(os)) => {
                let ks = ks.slice(k_l.start_offset()..);
                let ds = ds.slice(d_l.start_offset()..);
                let c_i32 = c as i32;
                let mut builder = func.builder();
                builder.arg(&ks);
                builder.arg(&ds);
                builder.arg(os);
                builder.arg(&total);
                builder.arg(&c_i32);
                unsafe { builder.launch(cfg) }.w()?;
            }
            _ => candle_core::bail!("gdn_neg_lower_mul: f32 only"),
        }
    }
    Ok(out)
}

// =====================================================================
// Fused chunked delta-rule cross-chunk recurrence
// =====================================================================

/// Runs the sequential nc-chunk state recurrence of the chunked gated delta
/// rule in one kernel launch (one thread per group×output-feature).
/// All inputs F32, contiguous. Shapes: attn (BH,Nc,C,C); qg/kcd/ks (BH,Nc,C,Hk);
/// v (BH,Nc,C,Hv); glast (BH,Nc). Returns core (BH,Nc,C,Hv).
#[allow(clippy::too_many_arguments)]
pub fn fused_chunk_recurrence(
    attn: &Tensor, qg: &Tensor, kcd: &Tensor, ks: &Tensor, v: &Tensor, glast: &Tensor,
    bh: usize, nc: usize, c: usize, hk: usize, hv: usize,
) -> Result<Tensor> {
    let attn = attn.contiguous()?; let qg = qg.contiguous()?; let kcd = kcd.contiguous()?;
    let ks = ks.contiguous()?; let v = v.contiguous()?; let glast = glast.contiguous()?;
    let dev = match attn.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("fused_chunk_recurrence: CUDA only"),
    };
    let func = load_func!(&dev, "fused_chunk_recurrence_f32")?;
    let w = 32usize; // column tile width (matches kernel)
    let cfg = LaunchConfig {
        grid_dim: ((bh * (hv / w)) as u32, 1, 1),
        block_dim: (256, 1, 1), // P=8 partitions x W=32 columns
        shared_mem_bytes: ((hk * w + c * w) * 4) as u32,
    };
    let out = Tensor::zeros((bh, nc, c, hv), DType::F32, attn.device())?;
    {
        let (a_g, a_l) = attn.storage_and_layout();
        let (qg_g, qg_l) = qg.storage_and_layout();
        let (kcd_g, kcd_l) = kcd.storage_and_layout();
        let (ks_g, ks_l) = ks.storage_and_layout();
        let (v_g, v_l) = v.storage_and_layout();
        let (gl_g, gl_l) = glast.storage_and_layout();
        let (o_g, _) = out.storage_and_layout();
        macro_rules! f32s {
            ($g:expr) => {
                match &*$g {
                    candle_core::Storage::Cuda(cs) => match &cs.slice {
                        CudaStorageSlice::F32(s) => s,
                        _ => candle_core::bail!("fused_chunk_recurrence: f32 only"),
                    },
                    _ => unreachable!(),
                }
            };
        }
        let a_s = f32s!(a_g).slice(a_l.start_offset()..);
        let qg_s = f32s!(qg_g).slice(qg_l.start_offset()..);
        let kcd_s = f32s!(kcd_g).slice(kcd_l.start_offset()..);
        let ks_s = f32s!(ks_g).slice(ks_l.start_offset()..);
        let v_s = f32s!(v_g).slice(v_l.start_offset()..);
        let gl_s = f32s!(gl_g).slice(gl_l.start_offset()..);
        let o_s = f32s!(o_g);
        let (bh_i, nc_i, c_i, hk_i, hv_i) = (bh as i32, nc as i32, c as i32, hk as i32, hv as i32);
        let mut builder = func.builder();
        builder.arg(&a_s);
        builder.arg(&qg_s);
        builder.arg(&kcd_s);
        builder.arg(&ks_s);
        builder.arg(&v_s);
        builder.arg(&gl_s);
        builder.arg(o_s);
        builder.arg(&bh_i);
        builder.arg(&nc_i);
        builder.arg(&c_i);
        builder.arg(&hk_i);
        builder.arg(&hv_i);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(out)
}
