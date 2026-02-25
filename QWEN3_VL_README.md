# Qwen3-VL-2B on Crane

High-performance Rust inference for the **Qwen3-VL-2B** vision-language model, built on the [Crane](https://github.com/3tic-project/Crane) inference engine.

## Features

- Pure Rust/CUDA implementation (no Python runtime)
- OpenAI-compatible API server (`/v1/chat/completions`)
- Streaming token generation
- BF16 support (~50% less VRAM, same output quality)
- CUDA-optimized: fused QKV, fused SiLU-mul, pre-allocated KV cache, GQA SDPA, F32 lm_head

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Rust | 1.75+ | Install via [rustup.rs](https://rustup.rs/) |
| CUDA Toolkit | 11.8+ | Optional. For GPU acceleration |
| Disk space | ~8.5 GB | For model weights |
| GPU VRAM | 6 GB (BF16) / 10 GB (F32) | Or use CPU mode (slower) |

---

## Quick Start

### Automated setup

```bash
bash setup_qwen3_vl.sh
```

This clones the repo, builds with CUDA (if available), downloads model weights, and runs a smoke test.

Options:
```bash
bash setup_qwen3_vl.sh --cpu          # Force CPU-only build
bash setup_qwen3_vl.sh --skip-model   # Skip model download
```

### Manual setup

```bash
# 1. Clone
git clone https://github.com/xellDart/Crane.git
cd Crane
git checkout feat/qwen3-vl-implementation

# 2. Build (pick one)
cargo build --release --features cuda    # GPU (NVIDIA)
cargo build --release                     # CPU only

# 3. Download model weights
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-2B --local-dir checkpoints/qwen3_vl_2b

# 4. Test
./target/release/qwen3_vl_simple checkpoints/qwen3_vl_2b ./photo.jpg "Describe this image"
```

---

## Direct Inference (`qwen3_vl_simple`)

### Image mode

Process one or more images with a text prompt:

```bash
# Single image
./target/release/qwen3_vl_simple <model_path> <image.jpg> "Your prompt"

# Multiple images (comma-separated)
./target/release/qwen3_vl_simple <model_path> img1.jpg,img2.jpg,img3.jpg "Compare these images"

# BF16 mode (half VRAM, recommended for GPUs with < 10 GB)
./target/release/qwen3_vl_simple <model_path> <image.jpg> "Your prompt" --bf16
```

### Dataset mode

Test against a JSON dataset with expected outputs:

```bash
# Process entry N from a dataset file
./target/release/qwen3_vl_simple <model_path> --entry 0 --dataset train.json --bf16
```

The dataset format is an array of objects with `images` (list of paths) and `conversations` (list of `{from, value}` pairs):

```json
[
  {
    "images": ["path/to/image.jpg"],
    "conversations": [
      {"from": "human", "value": "Describe this image"},
      {"from": "gpt", "value": "Expected output..."}
    ]
  }
]
```

### Output

```
Loading model from: checkpoints/qwen3_vl_2b (bf16=true)
Loading vision encoder...
Loading text decoder...
Initializing M-RoPE...
Model loaded!
---
{generated output here, streamed token by token}
---
Generated 142 tokens in 1.83s (77.6 tok/s)
```

---

## Server Mode (`crane-oai`)

### Start the server

```bash
./target/release/crane-oai \
  --model-path checkpoints/qwen3_vl_2b \
  --port 8080
```

The model type is auto-detected from `config.json`. To be explicit:

```bash
./target/release/crane-oai \
  --model-path checkpoints/qwen3_vl_2b \
  --model-type qwen3_vl \
  --port 8080
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model-path` | *(required)* | Path to model directory |
| `--model-type` | `auto` | Model type: `auto`, `qwen3_vl`, `qwen3`, etc. |
| `--port` | `8080` | Server port |
| `--host` | `0.0.0.0` | Bind address |
| `--cpu` | *(off)* | Force CPU inference |
| `--model-name` | *(dir name)* | Custom model name in API responses |

### Health check

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

---

## API Reference

### Chat Completions (single response)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg"}
          },
          {
            "type": "text",
            "text": "What is in this image?"
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg"}
          },
          {
            "type": "text",
            "text": "Describe this image in detail"
          }
        ]
      }
    ],
    "max_tokens": 512,
    "stream": true
  }'
```

### Multiple images

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://example.com/front.jpg"}},
          {"type": "image_url", "image_url": {"url": "https://example.com/back.jpg"}},
          {"type": "text", "text": "Compare these two images"}
        ]
      }
    ],
    "max_tokens": 512
  }'
```

### Local file images

Use `file://` URLs for local images:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "file:///absolute/path/to/image.jpg"}},
          {"type": "text", "text": "Extract all text from this document"}
        ]
      }
    ],
    "max_tokens": 1024
  }'
```

### Python client (OpenAI SDK)

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# From URL
response = client.chat.completions.create(
    model="qwen3-vl",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
                {"type": "text", "text": "Describe this image"},
            ],
        }
    ],
    max_tokens=512,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen3-vl",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
                {"type": "text", "text": "What do you see?"},
            ],
        }
    ],
    max_tokens=512,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Performance

### BF16 vs F32

| Mode | VRAM usage (2B) | Throughput | Accuracy |
|---|---|---|---|
| F32 | ~10 GB | Baseline | Reference |
| BF16 (`--bf16`) | ~6 GB | Similar | Identical output |

BF16 is recommended for GPUs with less than 10 GB VRAM. Output quality is identical thanks to the F32 lm_head projection.

### Optimizations applied

| Optimization | Speedup | Description |
|---|---|---|
| Pre-allocated KV cache | ~2x decode | `slice_set` O(1) vs `Tensor::cat` O(n) per token |
| Fused QKV projection | ~15% prefill | 1 matmul instead of 3 for Q/K/V |
| Fused SiLU-mul (CUDA) | ~10% MLP | Single kernel for gate+activation+up |
| GQA SDPA decode | ~20% decode | Avoids KV head expansion via reshape trick |
| F32 lm_head | Precise EOS | Prevents BF16 rounding from missing stop tokens |

---

## Troubleshooting

### CUDA out of memory

```
Error: CUDA out of memory
```

Use BF16 mode to halve VRAM usage:
```bash
# Direct inference
./target/release/qwen3_vl_simple <model_path> <image.jpg> "prompt" --bf16

# Server (BF16 is auto-detected from model weights, or force CPU)
./target/release/crane-oai --model-path <path> --cpu
```

### No safetensors files found

```
Error: No safetensors files found in checkpoints/qwen3_vl_2b
```

Download the model:
```bash
huggingface-cli download Qwen/Qwen3-VL-2B --local-dir checkpoints/qwen3_vl_2b
```

### matmul is only supported for contiguous tensors

This is already fixed in the current implementation. Make sure you're on the latest commit of `feat/qwen3-vl-implementation`.

### Build fails: CUDA not found

If you don't have CUDA installed, build without the cuda feature:
```bash
cargo build --release --bin qwen3_vl_simple
cargo build --release --bin crane-oai
```

The model will run on CPU (slower but functional).

---

## Architecture

```
Qwen3-VL-2B
├── Vision Encoder (24 layers)
│   ├── Conv3D Patch Embedding (16x16 patches)
│   ├── 2D Rotary Position Embeddings
│   ├── Multi-Head Self-Attention (16 heads)
│   ├── Spatial Merge (2x2 → 4x token reduction)
│   └── DeepStack (multi-scale vision features)
├── Text Decoder (28 layers)
│   ├── M-RoPE (3D: temporal + height + width positions)
│   ├── GQA Attention (16 heads, 8 KV heads)
│   ├── QK-Norm (per-head RMSNorm)
│   └── SwiGLU MLP (fused gate+up on CUDA)
└── LM Head (F32 for precise token selection)
```

---

## License

Qwen3-VL-2B model weights are released under the [Apache 2.0 license](https://huggingface.co/Qwen/Qwen3-VL-2B).
Crane inference engine is released under its own license. See the main [README](README.md).
