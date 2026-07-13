//! FP8 (E4M3) W8A8 linear via cuBLASLt, on top of candle's CUDA backend.
//!
//! candle 0.9 has the F8E4M3 dtype but no FP8 matmul; cudarc exposes cuBLASLt
//! only for f32/f16/bf16. This module drives cuBLASLt FP8 directly through
//! `cudarc::cublaslt::{result, sys}` (the same calls proven ~2x faster than bf16
//! in the standalone spike on this 4090).
//!
//! Orientation (no output transpose): candle Linear is `y[t,out]=x[t,in]@w[out,in]^T`.
//! FP8 needs transa=T/transb=N, so A=weights (op=T), B=acts (op=N), D=[out,t]
//! column-major == `[t,out]` row-major — candle's native layout.
//!
//! Fully on-device, single-stream: activation absmax+quant+scale run as kernels;
//! the activation-scale buffer is cached and rewritten each call (stream ordering
//! makes this race-free without a host sync). Weight scale is a per-weight `[1]`
//! device tensor produced at load. Per-shape cuBLASLt plans are cached.

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::cuda_backend::cudarc::cublaslt::{result, sys};
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;
    use candle_core::{DType, Device, Result, Storage, Tensor};
    use std::collections::HashMap;
    use std::ffi::c_void;

    const E4M3_MAX: f32 = 448.0;
    const WS_SIZE: usize = 32 * 1024 * 1024;

    thread_local! {
        static LT: std::cell::RefCell<Option<LtCtx>> = const { std::cell::RefCell::new(None) };
    }

    #[derive(Clone, Copy)]
    struct Plan {
        desc: sys::cublasLtMatmulDesc_t,
        a_layout: sys::cublasLtMatrixLayout_t,
        b_layout: sys::cublasLtMatrixLayout_t,
        d_layout: sys::cublasLtMatrixLayout_t,
        algo: sys::cublasLtMatmulAlgo_t,
    }

    struct LtCtx {
        handle: sys::cublasLtHandle_t,
        ws: candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
        a_scale: Tensor, // [1] f32, activation scale (written on-device each call)
        amax: Tensor,    // [1] f32 scratch
        plans: HashMap<(usize, usize, usize), Plan>,
    }

    fn w<E: std::fmt::Debug>(e: E) -> candle_core::Error {
        candle_core::Error::Msg(format!("cublasLt/cuda fp8: {e:?}"))
    }
    fn dt_e4m3() -> sys::cudaDataType {
        sys::cudaDataType_t::CUDA_R_8F_E4M3
    }
    fn dt_bf16() -> sys::cudaDataType {
        sys::cudaDataType_t::CUDA_R_16BF
    }
    fn dt_f32() -> sys::cudaDataType {
        sys::cudaDataType_t::CUDA_R_32F
    }

    /// Quantize a bf16 weight `[out,in]` to (F8E4M3 tensor, `[1]` f32 scale tensor).
    /// One-time at load. `scale = absmax/448`; the GEMM multiplies it back in.
    pub fn quantize_weight_e4m3(weight: &Tensor) -> Result<(Tensor, Tensor)> {
        let amax = super::super::absmax_f32(weight)?;
        let scale = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
        let inv = if amax > 0.0 { E4M3_MAX / amax } else { 0.0 };
        let wq = super::super::quantize_e4m3(weight, inv)?;
        let scale_t = Tensor::new(&[scale], weight.device())?;
        Ok((wq, scale_t))
    }

    /// FP8 linear: `x[..,in] (bf16) @ w_fp8[out,in]^T -> [..,out] (bf16)`.
    pub fn fp8_linear(x: &Tensor, w_fp8: &Tensor, w_scale: &Tensor) -> Result<Tensor> {
        let in_dim = *x.dims().last().unwrap();
        let (out_dim, w_in) = w_fp8.dims2()?;
        if w_in != in_dim {
            candle_core::bail!("fp8_linear: in mismatch {in_dim} vs weight {w_in}");
        }
        let lead: Vec<usize> = x.dims()[..x.dims().len() - 1].to_vec();
        let t: usize = lead.iter().product();
        let x2 = x.reshape((t, in_dim))?.to_dtype(DType::BF16)?.contiguous()?;
        // FP8 cuBLASLt needs the token dim %16; zero-pad rows, slice back after.
        let tp = t.div_ceil(16) * 16;
        let x2p = if tp != t { x2.pad_with_zeros(0, 0, tp - t)? } else { x2 };

        let d = fp8_gemm(&x2p, w_fp8, w_scale, tp, out_dim, in_dim)?;
        let d = if tp != t { d.narrow(0, 0, t)? } else { d };
        let mut out_shape = lead;
        out_shape.push(out_dim);
        d.reshape(out_shape)
    }

    /// D[t,out] bf16 = (X·W^T) in FP8 tensor-core. `x2p` is bf16 [t,in] (t%16==0).
    fn fp8_gemm(
        x2p: &Tensor,
        w_fp8: &Tensor,
        w_scale: &Tensor,
        t: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Tensor> {
        let dev = match x2p.device() {
            Device::Cuda(d) => d.clone(),
            _ => candle_core::bail!("fp8_gemm: CUDA only"),
        };
        let stream = dev.cuda_stream();

        LT.with(|cell| -> Result<Tensor> {
            {
                let mut ctx = cell.borrow_mut();
                if ctx.is_none() {
                    let handle = result::create_handle().map_err(w)?;
                    let ws = stream.alloc_zeros::<u8>(WS_SIZE).map_err(w)?;
                    let a_scale = Tensor::zeros((1,), DType::F32, x2p.device())?;
                    let amax = Tensor::zeros((1,), DType::F32, x2p.device())?;
                    *ctx = Some(LtCtx { handle, ws, a_scale, amax, plans: HashMap::new() });
                }
            }
            // Clone the cached scale/scratch handles (Arc — share storage) so the
            // activation quant runs without holding the ctx borrow.
            let (a_scale_t, amax_t) = {
                let ctx = cell.borrow();
                let c = ctx.as_ref().unwrap();
                (c.a_scale.clone(), c.amax.clone())
            };
            let x_fp8 = super::super::quantize_activation_dev(x2p, &amax_t, &a_scale_t)?;
            let out = Tensor::zeros((t, out_dim), DType::BF16, x2p.device())?;

            let mut ctx = cell.borrow_mut();
            let ctx = ctx.as_mut().unwrap();
            let key = (t, out_dim, in_dim);
            if !ctx.plans.contains_key(&key) {
                let plan = build_plan(ctx.handle, t, out_dim, in_dim)?;
                ctx.plans.insert(key, plan);
            }
            let plan = *ctx.plans.get(&key).unwrap();

            unsafe {
                // All storage guards + device_ptr guards must live until matmul returns.
                let (wsg, wsl) = w_scale.storage_and_layout();
                let (asg, asl) = a_scale_t.storage_and_layout();
                let (ag, al) = w_fp8.storage_and_layout();
                let (bg, bl) = x_fp8.storage_and_layout();
                let (dg, _dl) = out.storage_and_layout();
                let ws_scale = match &*wsg {
                    Storage::Cuda(s) => match &s.slice {
                        CudaStorageSlice::F32(s) => s.slice(wsl.start_offset()..),
                        _ => candle_core::bail!("fp8: w_scale not f32"),
                    },
                    _ => unreachable!(),
                };
                let as_scale = match &*asg {
                    Storage::Cuda(s) => match &s.slice {
                        CudaStorageSlice::F32(s) => s.slice(asl.start_offset()..),
                        _ => candle_core::bail!("fp8: a_scale not f32"),
                    },
                    _ => unreachable!(),
                };
                let a_s = match &*ag {
                    Storage::Cuda(s) => match &s.slice {
                        CudaStorageSlice::F8E4M3(s) => s.slice(al.start_offset()..),
                        _ => candle_core::bail!("fp8: w not f8e4m3"),
                    },
                    _ => unreachable!(),
                };
                let b_s = match &*bg {
                    Storage::Cuda(s) => match &s.slice {
                        CudaStorageSlice::F8E4M3(s) => s.slice(bl.start_offset()..),
                        _ => candle_core::bail!("fp8: x not f8e4m3"),
                    },
                    _ => unreachable!(),
                };
                let d_s = match &*dg {
                    Storage::Cuda(s) => match &s.slice {
                        CudaStorageSlice::BF16(s) => s,
                        _ => candle_core::bail!("fp8_gemm: out not bf16"),
                    },
                    _ => unreachable!(),
                };
                let (w_sc_ptr, _gws) = ws_scale.device_ptr(&stream);
                let (a_sc_ptr, _gas) = as_scale.device_ptr(&stream);
                let w_sc = w_sc_ptr;
                let a_sc = a_sc_ptr;
                // Scale pointers: A = weights, B = activations.
                set(plan.desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_sc)?;
                set(plan.desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_sc)?;

                let (a_ptr, _ga) = a_s.device_ptr(&stream);
                let (b_ptr, _gb) = b_s.device_ptr(&stream);
                let (d_ptr, _gd) = d_s.device_ptr(&stream);
                let (ws_ptr, _gw) = ctx.ws.device_ptr(&stream);

                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                result::matmul(
                    ctx.handle,
                    plan.desc,
                    &alpha as *const f32 as *const c_void,
                    &beta as *const f32 as *const c_void,
                    a_ptr as *const c_void,
                    plan.a_layout,
                    b_ptr as *const c_void,
                    plan.b_layout,
                    d_ptr as *const c_void,
                    plan.d_layout,
                    d_ptr as *mut c_void,
                    plan.d_layout,
                    &plan.algo as *const _,
                    ws_ptr as *mut c_void,
                    WS_SIZE,
                    stream.cu_stream() as *mut _,
                )
                .map_err(|e| w(format!("matmul: {e:?}")))?;
            }
            Ok(out)
        })
    }

    fn build_plan(handle: sys::cublasLtHandle_t, t: usize, out_dim: usize, in_dim: usize) -> Result<Plan> {
        let k = in_dim as u64;
        unsafe {
            let a_layout =
                result::create_matrix_layout(dt_e4m3(), k, out_dim as u64, in_dim as i64).map_err(w)?;
            let b_layout =
                result::create_matrix_layout(dt_e4m3(), k, t as u64, in_dim as i64).map_err(w)?;
            let d_layout =
                result::create_matrix_layout(dt_bf16(), out_dim as u64, t as u64, out_dim as i64).map_err(w)?;

            let desc =
                result::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, dt_f32()).map_err(w)?;
            let op_t: i32 = 1;
            let op_n: i32 = 0;
            set(desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &op_t)?;
            set(desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &op_n)?;
            let fast: i8 = 1;
            set(desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast)?;

            let pref = result::create_matmul_pref().map_err(w)?;
            let ws_bytes: usize = WS_SIZE;
            result::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            )
            .map_err(w)?;
            let heur = result::get_matmul_algo_heuristic(handle, desc, a_layout, b_layout, d_layout, d_layout, pref)
                .map_err(|e| w(format!("heuristic: {e:?}")))?;
            result::destroy_matmul_pref(pref).ok();
            Ok(Plan { desc, a_layout, b_layout, d_layout, algo: heur.algo })
        }
    }

    unsafe fn set<T>(
        desc: sys::cublasLtMatmulDesc_t,
        attr: sys::cublasLtMatmulDescAttributes_t,
        v: &T,
    ) -> Result<()> {
        result::set_matmul_desc_attribute(desc, attr, v as *const T as *const c_void, std::mem::size_of::<T>())
            .map_err(w)
    }
}
