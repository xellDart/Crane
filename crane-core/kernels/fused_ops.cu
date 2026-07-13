/**
 * Fused CUDA kernels for Crane transformer inference.
 *
 * Targets: sm_80+ (Ampere & newer, bf16 support)
 *
 * Kernels:
 *   1. fused_rmsnorm_residual_bf16  — RMSNorm + residual save
 *   2. fused_silu_mul_bf16          — SiLU(gate) * up  (one pass)
 *   3. fused_add_rmsnorm_bf16       — residual_add + RMSNorm (one pass)
 *   4. gpu_argmax_bf16              — GPU-side argmax over vocab (greedy decode)
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// =====================================================================
// Helpers
// =====================================================================

static constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Fast SiLU: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =====================================================================
// 1. Fused RMSNorm — one block per row
//
//    dst[row, col] = (x[row, col] / rms) * weight[col]
//    where rms = sqrt(mean(x²) + eps)
//
//    Identical to candle's rmsnorm but with explicit bf16 I/O and
//    handles up to 16384 columns per warp-tree reduction.
// =====================================================================

extern "C" __global__ void fused_rmsnorm_bf16(
    const __nv_bfloat16 *__restrict__ x,      // [rows, cols]
    __nv_bfloat16       *__restrict__ dst,     // [rows, cols]
    const __nv_bfloat16 *__restrict__ weight,  // [cols]
    const int ncols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Phase 1: compute sum of squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(x[row * ncols + col]);
        sum_sq += v * v;
    }

    // Warp reduce
    sum_sq = warp_reduce_sum_f32(sum_sq);

    // Cross-warp reduce via shared memory
    __shared__ float s_partial[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum_f32(sum_sq);
        if (lane_id == 0) s_partial[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(s_partial[0] / (float)ncols + eps);

    // Phase 2: normalize and write output
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(x[row * ncols + col]);
        float w = __bfloat162float(weight[col]);
        dst[row * ncols + col] = __float2bfloat16(v * scale * w);
    }
}

// =====================================================================
// 2. Fused SiLU(gate) * up — one pass over 2 * intermediate_size
//
//    Input:  gate_up [rows, 2*intermediate_size]  (gate||up concatenated)
//    Output: dst     [rows, intermediate_size]
//    dst[i] = silu(gate_up[i]) * gate_up[i + intermediate_size]
//
//    Saves 2 kernel launches (separate silu + mul) and 1 intermediate
//    tensor allocation.
// =====================================================================

extern "C" __global__ void fused_silu_mul_bf16(
    const __nv_bfloat16 *__restrict__ gate_up,  // [rows, 2*intermediate_size]
    __nv_bfloat16       *__restrict__ dst,       // [rows, intermediate_size]
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __nv_bfloat16 *gate_row = gate_up + row * 2 * intermediate_size;
    const __nv_bfloat16 *up_row   = gate_row + intermediate_size;
    __nv_bfloat16       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = __bfloat162float(gate_row[i]);
        float u = __bfloat162float(up_row[i]);
        dst_row[i] = __float2bfloat16(fast_silu(g) * u);
    }
}

// f16 variant
extern "C" __global__ void fused_silu_mul_f16(
    const __half *__restrict__ gate_up,
    __half       *__restrict__ dst,
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __half *gate_row = gate_up + row * 2 * intermediate_size;
    const __half *up_row   = gate_row + intermediate_size;
    __half       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = __half2float(gate_row[i]);
        float u = __half2float(up_row[i]);
        dst_row[i] = __float2half(fast_silu(g) * u);
    }
}

// f32 variant
extern "C" __global__ void fused_silu_mul_f32(
    const float *__restrict__ gate_up,
    float       *__restrict__ dst,
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float *gate_row = gate_up + row * 2 * intermediate_size;
    const float *up_row   = gate_row + intermediate_size;
    float       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = gate_row[i];
        float u = up_row[i];
        dst_row[i] = fast_silu(g) * u;
    }
}

// =====================================================================
// 3. Fused residual_add + RMSNorm — one read of hidden, write norm + residual
//
//    residual[row] += hidden[row]            (in-place update)
//    dst[row] = rmsnorm(residual[row]) * weight
//
//    Eliminates the separate add kernel + RMSNorm kernel + extra read.
// =====================================================================

extern "C" __global__ void fused_add_rmsnorm_bf16(
    __nv_bfloat16       *__restrict__ residual,  // [rows, cols] — updated in-place
    const __nv_bfloat16 *__restrict__ hidden,    // [rows, cols] — value to add
    __nv_bfloat16       *__restrict__ dst,        // [rows, cols] — normalized output
    const __nv_bfloat16 *__restrict__ weight,     // [cols]
    const int ncols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int row_offset = row * ncols;

    // Phase 1: add residual, compute sum of squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float r = __bfloat162float(residual[row_offset + col]);
        float h = __bfloat162float(hidden[row_offset + col]);
        float v = r + h;
        // Write residual back (in-place update)
        residual[row_offset + col] = __float2bfloat16(v);
        sum_sq += v * v;
    }

    // Warp + cross-warp reduce
    sum_sq = warp_reduce_sum_f32(sum_sq);
    __shared__ float s_partial[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum_f32(sum_sq);
        if (lane_id == 0) s_partial[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(s_partial[0] / (float)ncols + eps);

    // Phase 2: normalize from updated residual
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(residual[row_offset + col]);
        float w = __bfloat162float(weight[col]);
        dst[row_offset + col] = __float2bfloat16(v * scale * w);
    }
}

// =====================================================================
// 3b. Fused residual_add + RMSNorm (out-of-place variant)
//
//     sum_out[row] = residual[row] + hidden[row]
//     norm_out[row] = rmsnorm(sum_out[row]) * weight
//
//     Non-mutating: reads residual, writes to separate sum_out buffer.
//     Eliminates separate add kernel + RMSNorm kernel + extra memory read.
// =====================================================================

extern "C" __global__ void fused_add_rmsnorm_out_bf16(
    const __nv_bfloat16 *__restrict__ residual,  // [rows, cols] — read only
    const __nv_bfloat16 *__restrict__ hidden,    // [rows, cols] — read only
    __nv_bfloat16       *__restrict__ sum_out,   // [rows, cols] — residual + hidden
    __nv_bfloat16       *__restrict__ norm_out,  // [rows, cols] — normalized output
    const __nv_bfloat16 *__restrict__ weight,    // [cols]
    const int ncols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int row_offset = row * ncols;

    // Phase 1: compute sum, write sum_out, accumulate sum of squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float r = __bfloat162float(residual[row_offset + col]);
        float h = __bfloat162float(hidden[row_offset + col]);
        float v = r + h;
        sum_out[row_offset + col] = __float2bfloat16(v);
        sum_sq += v * v;
    }

    // Warp + cross-warp reduce
    sum_sq = warp_reduce_sum_f32(sum_sq);
    __shared__ float s_partial[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum_f32(sum_sq);
        if (lane_id == 0) s_partial[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(s_partial[0] / (float)ncols + eps);

    // Phase 2: normalize from sum_out (same addresses, L1-hot)
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(sum_out[row_offset + col]);
        float w = __bfloat162float(weight[col]);
        norm_out[row_offset + col] = __float2bfloat16(v * scale * w);
    }
}

// =====================================================================
// 4. GPU Argmax — two-phase reduction for vocab-size vectors
//
//    Phase 1: Each block reduces a chunk of rows → per-block max + argmax
//    Phase 2: Single block reduces per-block results → final argmax
//
//    For greedy decode: replaces 303KB DtoH + CPU argmax with
//    a single scalar (4 bytes) DtoH.
// =====================================================================

extern "C" __global__ void gpu_argmax_bf16_phase1(
    const __nv_bfloat16 *__restrict__ logits,  // [vocab_size]
    float               *__restrict__ block_max_vals,
    int32_t             *__restrict__ block_max_idxs,
    const int vocab_size
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int bid = blockIdx.x;
    const int num_blocks = gridDim.x;

    // Each block handles a strided chunk
    int chunk = (vocab_size + num_blocks - 1) / num_blocks;
    int start = bid * chunk;
    int end   = min(start + chunk, vocab_size);

    float local_max = -INFINITY;
    int   local_idx = -1;

    for (int i = start + tid; i < end; i += block_size) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int   other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    // Cross-warp reduce
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    __shared__ float s_max_vals[32];
    __shared__ int   s_max_idxs[32];

    if (lane_id == 0) {
        s_max_vals[warp_id] = local_max;
        s_max_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < num_warps) {
        local_max = s_max_vals[lane_id];
        local_idx = s_max_idxs[lane_id];

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
            int   other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            block_max_vals[bid] = local_max;
            block_max_idxs[bid] = local_idx;
        }
    }
}

extern "C" __global__ void gpu_argmax_phase2(
    const float   *__restrict__ block_max_vals,
    const int32_t *__restrict__ block_max_idxs,
    int32_t       *__restrict__ output_token,
    const int num_blocks
) {
    const int tid = threadIdx.x;

    float best_val = -INFINITY;
    int   best_idx = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > best_val) {
            best_val = v;
            best_idx = block_max_idxs[i];
        }
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int   other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    // Cross-warp reduce
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    __shared__ float s_vals[32];
    __shared__ int   s_idxs[32];

    if (lane_id == 0) {
        s_vals[warp_id] = best_val;
        s_idxs[warp_id] = best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        best_val = (lane_id < num_warps) ? s_vals[lane_id] : -INFINITY;
        best_idx = (lane_id < num_warps) ? s_idxs[lane_id] : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
            int   other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            *output_token = best_idx;
        }
    }
}

// =====================================================================
// 5. Fused residual_add in-place: residual += hidden
//    Simple element-wise kernel to avoid candle's tensor add overhead.
// =====================================================================

extern "C" __global__ void fused_residual_add_bf16(
    __nv_bfloat16       *__restrict__ residual,  // [n] — updated in-place
    const __nv_bfloat16 *__restrict__ hidden,    // [n]
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = __bfloat162float(residual[idx]);
        float h = __bfloat162float(hidden[idx]);
        residual[idx] = __float2bfloat16(r + h);
    }
}

// =====================================================================
// 6. GPU TopK — two-stage block reduction (k ≤ 64)
//
//    Stage 1: Each block processes `items_per_block` elements from
//             the input, producing per-block top-k.
//    Stage 2: Single block merges all per-block results → final top-k
//             indices output.
// =====================================================================

static inline __device__ void topk_insert(
    float v,
    uint32_t i,
    float * vals,
    uint32_t * idx,
    const int k
) {
    if (v <= vals[k - 1]) return;
    int p = k - 1;
    while (p > 0 && v > vals[p - 1]) {
        vals[p] = vals[p - 1];
        idx[p] = idx[p - 1];
        --p;
    }
    vals[p] = v;
    idx[p] = i;
}

extern "C" __global__ void topk_stage1_f32(
    const float * x,
    const uint32_t n,
    const uint32_t k,
    const uint32_t items_per_block,
    float * out_vals,
    uint32_t * out_idx
) {
    const uint32_t start = blockIdx.x * items_per_block;
    const uint32_t end = min(n, start + items_per_block);
    if (start >= end) return;

    float vals[64];
    uint32_t idx[64];
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    for (uint32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = x[i];
        topk_insert(v, i, vals, idx, (int)k);
    }

    extern __shared__ uint8_t smem[];
    float * block_vals = (float *)smem;
    uint32_t * block_idx = (uint32_t *)(block_vals + (uint32_t)blockDim.x * k);

    const uint32_t base = (uint32_t)threadIdx.x * k;
    for (uint32_t j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float bvals[64];
        uint32_t bidx[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            bvals[j] = -INFINITY;
            bidx[j] = 0;
        }
        for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
            const uint32_t tb = t * k;
            for (uint32_t j = 0; j < k; ++j) {
                topk_insert(block_vals[tb + j], block_idx[tb + j], bvals, bidx, (int)k);
            }
        }
        const uint32_t out_base = blockIdx.x * k;
        for (uint32_t j = 0; j < k; ++j) {
            out_vals[out_base + j] = bvals[j];
            out_idx[out_base + j] = bidx[j];
        }
    }
}

extern "C" __global__ void topk_stage2_f32(
    const float * in_vals,
    const uint32_t * in_idx,
    const uint32_t m,
    const uint32_t k,
    uint32_t * out_idx
) {
    float vals[64];
    uint32_t idx[64];
    #pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    for (uint32_t i = threadIdx.x; i < m; i += blockDim.x) {
        float v = in_vals[i];
        uint32_t id = in_idx[i];
        topk_insert(v, id, vals, idx, (int)k);
    }

    extern __shared__ uint8_t smem2[];
    float * block_vals = (float *)smem2;
    uint32_t * block_idx = (uint32_t *)(block_vals + (uint32_t)blockDim.x * k);

    const uint32_t base = (uint32_t)threadIdx.x * k;
    for (uint32_t j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float bvals[64];
        uint32_t bidx[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            bvals[j] = -INFINITY;
            bidx[j] = 0;
        }
        for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
            const uint32_t tb = t * k;
            for (uint32_t j = 0; j < k; ++j) {
                topk_insert(block_vals[tb + j], block_idx[tb + j], bvals, bidx, (int)k);
            }
        }
        for (uint32_t j = 0; j < k; ++j) {
            out_idx[j] = bidx[j];
        }
    }
}


// =====================================================================
// Depthwise causal conv1d (kernel_size KS) + SiLU. Replaces the window-stack
// (a per-timestep Tensor::stack of T slices, ~194MB/layer) with one launch.
//   x, out: (B, C, T) contiguous ; w: (C, KS)
//   out[b,c,t] = silu( sum_{k=0..KS-1} x[b,c,t-(KS-1)+k] * w[c,k] )   (t<0 -> 0)
// One thread per output element (grid-stride).
// =====================================================================
extern "C" __global__ void causal_conv1d_silu_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w,
    __nv_bfloat16       *__restrict__ out,
    const long total, const int C, const int T, const int KS
) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        const int t = (int)(i % T);
        const int c = (int)((i / T) % C);
        const long base = i - t;                 // b*C*T + c*T
        const __nv_bfloat16 *wc = w + (long)c * KS;
        float acc = 0.f;
        for (int k = 0; k < KS; ++k) {
            int tt = t - (KS - 1) + k;
            if (tt >= 0) acc += __bfloat162float(x[base + tt]) * __bfloat162float(wc[k]);
        }
        out[i] = __float2bfloat16(fast_silu(acc));
    }
}

extern "C" __global__ void causal_conv1d_silu_f32(
    const float *__restrict__ x,
    const float *__restrict__ w,
    float       *__restrict__ out,
    const long total, const int C, const int T, const int KS
) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        const int t = (int)(i % T);
        const int c = (int)((i / T) % C);
        const long base = i - t;
        const float *wc = w + (long)c * KS;
        float acc = 0.f;
        for (int k = 0; k < KS; ++k) {
            int tt = t - (KS - 1) + k;
            if (tt >= 0) acc += x[base + tt] * wc[k];
        }
        out[i] = fast_silu(acc);
    }
}

// =====================================================================
// Chunked delta-rule intra-chunk inverse (forward substitution).
// Replaces a 63-step Python-style loop of full-tensor slice_assigns.
// Input a (G,C,C) strictly-lower; output (G,C,C) = substituted + identity.
// One block per group, C threads (one per column), shared C*C tile.
//   for i in 1..C: row_i[j] = a[i,j] + sum_{k<i} a_orig[i,k]*T[k,j]   (j<i)
//   then T += I
// =====================================================================
extern "C" __global__ void chunk_delta_invert_f32(
    const float *__restrict__ a, float *__restrict__ out, const int C
) {
    extern __shared__ float tile[];   // C*C
    __shared__ float row_orig[128];   // C <= 128
    const int g = blockIdx.x;
    const int j = threadIdx.x;        // column
    const float *ag = a + (long)g * C * C;
    float *og = out + (long)g * C * C;
    for (int r = 0; r < C; ++r) tile[r * C + j] = ag[r * C + j];
    __syncthreads();
    for (int i = 1; i < C; ++i) {
        if (j < i) row_orig[j] = tile[i * C + j];
        __syncthreads();
        if (j < i) {
            float acc = row_orig[j];
            for (int k = 0; k < i; ++k) acc += row_orig[k] * tile[k * C + j];
            tile[i * C + j] = acc;
        }
        __syncthreads();
    }
    for (int r = 0; r < C; ++r)
        og[r * C + j] = tile[r * C + j] + (r == j ? 1.0f : 0.0f);
}

// =====================================================================
// Fused gated RMSNorm: out = rmsnorm(x)*weight * silu(gate).
//   x: (rows, D) f32 ; gate,weight: bf16 ; out: (rows, D) bf16.
// One block per row; block-reduces sum(x^2) over D.
// =====================================================================
extern "C" __global__ void fused_rmsnorm_gated(
    const float *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ w,
    __nv_bfloat16 *__restrict__ out,
    const int D, const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const float *xr = x + (long)row * D;
    const __nv_bfloat16 *gr = gate + (long)row * D;
    __nv_bfloat16 *outr = out + (long)row * D;

    float ss = 0.f;
    for (int d = tid; d < D; d += bs) { float v = xr[d]; ss += v * v; }
    ss = warp_reduce_sum_f32(ss);
    __shared__ float sh[32];
    if ((tid & 31) == 0) sh[tid >> 5] = ss;
    __syncthreads();
    int nwarps = (bs + 31) / 32;
    if (tid < 32) {
        ss = (tid < nwarps) ? sh[tid] : 0.f;
        ss = warp_reduce_sum_f32(ss);
    }
    __shared__ float inv_sh;
    if (tid == 0) inv_sh = rsqrtf(ss / (float)D + eps);
    __syncthreads();
    const float inv = inv_sh;

    for (int d = tid; d < D; d += bs) {
        float xn = xr[d] * inv * __bfloat162float(w[d]);
        outr[d] = __float2bfloat16(xn * fast_silu(__bfloat162float(gr[d])));
    }
}

// =====================================================================
// GDN scan glue: chunk decay mask.
//   g_cs: (G,C) within-chunk cumulative log-decay.
//   out : (G,C,C), out[g,i,j] = (i>=j) ? exp(g_cs[g,i] - g_cs[g,j]) : 0.
// Replaces candle's broadcast_sub -> mask -> exp -> mask chain (4 launches +
// a (G,C,C) `diff` intermediate) with one grid-stride pass reading only (G,C).
// =====================================================================
extern "C" __global__ void gdn_decay_mask_f32(
    const float *__restrict__ g_cs, float *__restrict__ out,
    const long total, const int C
) {
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        const int j = (int)(idx % C);
        const int i = (int)((idx / C) % C);
        const long g = idx / ((long)C * C);
        const float *gg = g_cs + g * C;
        out[idx] = (i >= j) ? expf(gg[i] - gg[j]) : 0.0f;
    }
}

// =====================================================================
// GDN scan glue: negated strict-lower masked product.
//   kk, decay: (G,C,C).  out[g,i,j] = (i>j) ? -(kk*decay) : 0.
// Replaces candle's broadcast_mul -> neg -> broadcast_mul (3 launches).
// =====================================================================
extern "C" __global__ void gdn_neg_lower_mul_f32(
    const float *__restrict__ kk, const float *__restrict__ decay,
    float *__restrict__ out, const long total, const int C
) {
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        const int j = (int)(idx % C);
        const int i = (int)((idx / C) % C);
        out[idx] = (i > j) ? -(kk[idx] * decay[idx]) : 0.0f;
    }
}

// =====================================================================
// Fused chunked delta-rule cross-chunk recurrence — tiled, shared-memory state.
// One block per (group g, column tile of width W=32). The tile's state (Hk x W)
// and v_new (C x W) live in shared memory; blockDim=256 = P(8) partitions x W(32)
// columns. Output column e is independent (no cross-thread reduction on outputs);
// partitions split the a/d loops for parallelism. Grid = BH * (Hv/W).
//   attn_all (BH,Nc,C,C), qg_all/kcd_all/ks_all (BH,Nc,C,Hk),
//   v_all (BH,Nc,C,Hv), glast_exp (BH,Nc)  ->  core_all (BH,Nc,C,Hv)
//   Hk<=128, C<=64, Hv % 32 == 0.
// =====================================================================
extern "C" __global__ void fused_chunk_recurrence_f32(
    const float *__restrict__ attn_all,
    const float *__restrict__ qg_all,
    const float *__restrict__ kcd_all,
    const float *__restrict__ ks_all,
    const float *__restrict__ v_all,
    const float *__restrict__ glast_exp,
    float *__restrict__ core_all,
    const int BH, const int Nc, const int C, const int Hk, const int Hv
) {
    const int W = 32;
    const int P = blockDim.x / W;      // partitions
    const int e = threadIdx.x % W;     // local column
    const int p = threadIdx.x / W;     // partition
    const int tiles = Hv / W;
    const int g = blockIdx.x / tiles;
    const int col0 = (blockIdx.x % tiles) * W;
    const int ge = col0 + e;           // global column

    extern __shared__ float sh[];
    float *state = sh;                 // [Hk*W]
    float *vnew = sh + Hk * W;         // [C*W]

    for (int d = p; d < Hk; d += P) state[d * W + e] = 0.f;
    __syncthreads();

    for (int i = 0; i < Nc; ++i) {
        const long chunk = (long)g * Nc + i;
        const float *attn_c = attn_all + chunk * C * C;
        const float *qg_c = qg_all + chunk * C * Hk;
        const float *kcd_c = kcd_all + chunk * C * Hk;
        const float *ks_c = ks_all + chunk * C * Hk;
        const float *v_c = v_all + chunk * C * Hv;
        float *core_c = core_all + chunk * C * Hv;

        // v_new[a][e] = v[a][ge] - sum_d kcd[a][d]*state[d][e]   (a over partitions)
        for (int a = p; a < C; a += P) {
            const float *kcd_a = kcd_c + a * Hk;
            float acc = 0.f;
            for (int d = 0; d < Hk; ++d) acc += kcd_a[d] * state[d * W + e];
            vnew[a * W + e] = v_c[a * Hv + ge] - acc;
        }
        __syncthreads();

        // core[a][ge] = sum_d qg[a][d]*state[d][e] + sum_b attn[a][b]*vnew[b][e]
        for (int a = p; a < C; a += P) {
            const float *qg_a = qg_c + a * Hk;
            float ai = 0.f;
            for (int d = 0; d < Hk; ++d) ai += qg_a[d] * state[d * W + e];
            const float *attn_a = attn_c + a * C;
            float cc = ai;
            for (int b = 0; b < C; ++b) cc += attn_a[b] * vnew[b * W + e];
            core_c[a * Hv + ge] = cc;
        }
        __syncthreads(); // finish reading state before the update below overwrites it

        // state[d][e] = state[d][e]*g_last + sum_a ks[a][d]*vnew[a][e]  (d over partitions)
        const float gl = glast_exp[chunk];
        for (int d = p; d < Hk; d += P) {
            float kv = 0.f;
            for (int a = 0; a < C; ++a) kv += ks_c[a * Hk + d] * vnew[a * W + e];
            state[d * W + e] = state[d * W + e] * gl + kv;
        }
        __syncthreads();
    }
}

// =====================================================================
// FP8 W8A8 quantization: bf16 -> E4M3 with a scalar reciprocal scale.
//   y[i] = (__nv_fp8_e4m3)( bf16_to_f32(x[i]) * inv )   where inv = 448/absmax.
// Used to quantize both activations (per-forward) and weights (offline).
// =====================================================================
extern "C" __global__ void quant_e4m3_bf16(
    const __nv_bfloat16 *__restrict__ x, __nv_fp8_e4m3 *__restrict__ y,
    const float inv, const long n
) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x) {
        y[i] = (__nv_fp8_e4m3)(__bfloat162float(x[i]) * inv);
    }
}
