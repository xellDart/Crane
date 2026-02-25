# Crane

> **C**andle-based **R**ust **A**ccelerated **N**eural **E**ngine — High-performance LLM inference in pure Rust.

![](data/aa.gif)

Crane runs LLM, VLM, TTS, ASR and OCR models at native speed on CPU, CUDA and Metal. Single binary, no Python runtime, no GGUF conversion required. OpenAI-compatible API out of the box.

---

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/xellDart/Crane/main/install.sh | bash
```

This auto-detects your hardware (CUDA, GPU compute capability, Flash Attention support), installs Rust if needed, and builds everything.

### Install options

```bash
# Build + download a model + start server
curl -fsSL ... | bash -s -- --model qwen3-vl-2b --start

# Build + start as background daemon with existing model
curl -fsSL ... | bash -s -- --model-path /path/to/model --model-type qwen3_vl --daemon --port 9090

# Force CPU-only build
curl -fsSL ... | bash -s -- --cpu

# Install to custom directory
curl -fsSL ... | bash -s -- --dir /opt/crane
```

| Flag | Description |
|------|-------------|
| `--model <name>` | Download model after build (`qwen3-vl-2b`, `qwen3-8b`, `hunyuan-7b`) |
| `--model-path <path>` | Use an existing local model directory |
| `--model-type <type>` | Model type (`qwen3`, `qwen3_vl`, `hunyuan`). Auto-detected with `--model` |
| `--start` | Start server in foreground after build |
| `--daemon` | Start server as background daemon after build |
| `--port <port>` | Server port (default: `8080`) |
| `--cpu` | Force CPU-only build (skip CUDA) |
| `--dir <path>` | Install directory (default: `./Crane`) |
| `--branch <name>` | Git branch (default: `main`) |

---

## Manual Build

```bash
git clone https://github.com/xellDart/Crane.git
cd Crane
```

### CPU only

```bash
cargo build --release
```

### CUDA

```bash
cargo build --release --features cuda
```

### CUDA + Flash Attention (SM_80+: Ampere, Ada Lovelace, Hopper)

```bash
cargo build --release --features flash-attn
```

> First build with Flash Attention takes ~10 min (CUTLASS compilation). Subsequent builds are cached.

### Using the build script

```bash
bash build.sh                    # Auto-detect everything
bash build.sh --cpu              # Force CPU
bash build.sh --no-flash-attn    # CUDA without Flash Attention
bash build.sh --bin crane-oai    # Build only the API server
bash build.sh --clean            # Clean + rebuild
```

---

## Download Models

```bash
pip install huggingface_hub

# Vision-Language
huggingface-cli download Qwen/Qwen3-VL-2B --local-dir checkpoints/qwen3_vl_2b

# Text
huggingface-cli download Qwen/Qwen3-8B --local-dir checkpoints/qwen3_8b

# Text (Hunyuan)
huggingface-cli download tencent/Hunyuan-A13B-Instruct --local-dir checkpoints/hunyuan

# TTS
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir checkpoints/qwen3_tts_base
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir checkpoints/qwen3_tts_custom
```

---

## Start the Server

Use `crane-ctl` to manage the server:

```bash
# Start as daemon
crane-ctl start --model-path checkpoints/qwen3_vl_2b --model-type qwen3_vl

# Start on a custom port
crane-ctl start --model-path checkpoints/qwen3_vl_2b --model-type qwen3_vl --port 9090

# Start in foreground (for debugging)
crane-ctl start --model-path checkpoints/qwen3_vl_2b --model-type qwen3_vl -f
```

### Manage the server

```bash
crane-ctl status     # PID, memory, health, model info
crane-ctl log        # Tail logs (Ctrl+C to exit)
crane-ctl health     # Quick health check
crane-ctl restart    # Restart with last config (remembers model + port)
crane-ctl stop       # Stop the server
```

### crane-ctl commands

| Command | Description |
|---------|-------------|
| `start` | Start the server as a daemon (or `-f` for foreground) |
| `stop` | Graceful shutdown (SIGTERM, then SIGKILL after 10s) |
| `restart` | Stop + start with last used config |
| `status` | Show PID, memory, model, port, health |
| `log` | Tail server logs (live if running, last 50 lines if stopped) |
| `health` | Quick `GET /health` check |

After the first `start`, `crane-ctl restart` remembers your `--model-path`, `--model-type` and `--port`.

### Manual start (without crane-ctl)

```bash
# Foreground
./target/release/crane-oai \
  --model-path checkpoints/qwen3_vl_2b \
  --model-type qwen3_vl \
  --port 8080

# Daemon
nohup ./target/release/crane-oai \
  --model-path checkpoints/qwen3_vl_2b \
  --model-type qwen3_vl \
  --port 8080 > crane-oai.log 2>&1 &
```

---

## Supported Models

| Model | Type | `--model-type` | Sizes |
|-------|------|----------------|-------|
| Qwen3 | Text LLM | `qwen3` | 0.6B — 30B+ |
| Qwen 2.5 | Text LLM | `qwen25` | 0.5B — 72B |
| Hunyuan Dense | Text LLM | `hunyuan` | 7B+ |
| Qwen3-VL | Vision-Language | `qwen3_vl` | 2B, 4B |
| PaddleOCR-VL | OCR / Vision | `paddleocr_vl` | 0.9B, 1.5B |
| Qwen3-TTS | Text-to-Speech | `qwen3_tts` | 0.6B |
| Moonshine | ASR (Speech-to-Text) | — | — |
| Silero VAD | Voice Activity Detection | — | — |

---

## API Reference

The server is fully compatible with the OpenAI SDK and SGLang client.

### OpenAI endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming & non-streaming) |
| `POST` | `/v1/completions` | Text completions |
| `POST` | `/v1/audio/speech` | Text-to-speech (Qwen3-TTS) |
| `GET` | `/v1/models` | List models |
| `POST` | `/v1/tokenize` | Tokenize text |
| `POST` | `/v1/detokenize` | Detokenize tokens |

### SGLang endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate` | Native text generation |
| `GET` | `/model_info` | Model metadata |
| `GET` | `/server_info` | Server stats |
| `GET` | `/health_generate` | Deep health check |

### Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/stats` | Engine statistics |

---

## Usage Examples

### Text chat (curl)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "crane",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Vision-Language (curl)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "crane",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        {"type": "text", "text": "Describe this image"}
      ]
    }]
  }'
```

### Text-to-Speech (curl)

```bash
curl http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "crane",
    "input": "Hello, this is Crane speaking.",
    "voice": "Chelsie",
    "response_format": "wav"
  }' -o output.wav
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="crane",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Rust (crane library)

```rust
use crane::llm::LLM;

fn main() -> anyhow::Result<()> {
    let mut llm = LLM::from_local("checkpoints/qwen3_8b")?;
    let response = llm.chat("Tell me about Rust programming.")?;
    println!("{}", response);
    Ok(())
}
```

---

## Examples

Run examples from the `example/` directory:

```bash
# Text chat
cargo run --bin chat_simple --release -- -m checkpoints/qwen3_8b
cargo run --bin chat_streaming --release -- -m checkpoints/qwen3_8b

# Vision-Language
cargo run --bin qwen3_vl_simple --release -- -m checkpoints/qwen3_vl_2b
cargo run --bin vision_simple --release
cargo run --bin ocr_simple --release

# TTS
cargo run --bin tts_simple --release -- checkpoints/qwen3_tts_base
cargo run --bin tts_custom_voice --release -- checkpoints/qwen3_tts_custom
cargo run --bin tts_voice_clone --release -- checkpoints/qwen3_tts_base

# ASR
cargo run --bin asr_simple --release

# Hunyuan
cargo run --bin hunyuan_simple --release -- -m checkpoints/hunyuan
```

For CUDA, add `--features cuda` or `--features flash-attn`.

---

## Project Structure

```
Crane/
├── crane-core/              # Core: model architectures, CUDA kernels, tokenizer
│   ├── src/models/          # Qwen3, Qwen2.5, HunyuanDense, Qwen3-VL, PaddleOCR-VL, Qwen3-TTS, ...
│   ├── src/fused_ops/       # Fused CUDA kernels (RMSNorm, Flash Attention dispatch)
│   └── kernels/             # Raw CUDA kernel sources (.cu)
├── crane/                   # High-level SDK (Chat, Vision, Audio, Multimodal)
├── crane-oai/               # OpenAI & SGLang compatible API server
│   └── src/
│       ├── engine/          # Inference engine, model factory, continuous batching
│       └── handlers/        # HTTP handlers (OpenAI, SGLang)
├── example/                 # Example binaries
├── crane-ctl                # Server management CLI (start/stop/log/status)
├── build.sh                 # Auto-detect build script
├── install.sh               # Curl-installable setup script
└── Cargo.toml               # Workspace
```

---

## Performance

### Optimizations

- Flash Attention v2 (SM_80+) via `candle-flash-attn`
- Fused `add + RMSNorm` CUDA kernel (in-place residual + normalization)
- Pre-allocated KV cache with smart growth
- GQA 4D matmul for grouped-query attention
- Fused RoPE with cache pre-growth
- GGUF quantization support
- Batched decode with continuous batching
- Smart sampling fallback for large vocabularies

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CRANE_FORCE_GPU_TOPK` | `0` | Force GPU top-k sampling for large vocabularies |
| `CRANE_TOPP_FALLBACK_TOPK` | `64` | Top-k size when top_p is active on GPU |
| `CRANE_TOPK_SAMPLE_ON_CPU` | `0` | Force CPU sampling after GPU top-k |
| `CRANE_SAMPLE_TRACE` | `0` | Enable sampling timing logs |

### Benchmarks

| Model | Metal (M1) F16 | CPU (M1) F32 | PyTorch F32 |
|-------|-----------------|--------------|-------------|
| Qwen2.5-0.5B | **35 t/s** | 14 t/s | 6.9 t/s |

---

## Build Features

| Feature | Cargo flag | Description |
|---------|-----------|-------------|
| CUDA | `--features cuda` | GPU acceleration via CUDA |
| Flash Attention | `--features flash-attn` | Flash Attention v2 (includes CUDA, requires SM_80+) |
| cuDNN | `--features cudnn` | cuDNN acceleration |
| MKL | `--features mkl` | Intel MKL for CPU |
| ONNX | `--features onnx` | ONNX runtime support |

---

## Contributing

PRs welcome. To add a new model:

1. Implement the architecture in `crane-core/src/models/<your_model>/`
2. Register it in `crane-core/src/models/mod.rs`
3. Add a `ModelType` variant in `crane-oai/src/engine/model_factory.rs` if it should be serveable
4. Reference `crane-core/src/models/siglip2.rs` for a minimal model example

---

## Updates

| Date | Change |
|------|--------|
| 2026.02.25 | Flash Attention v2, fused `add_rmsnorm` CUDA kernel, curl-installable `install.sh` |
| 2026.02.23 | Qwen3-TTS support (voice cloning, OpenAI `/v1/audio/speech` endpoint) |
| 2026.02.18 | Qwen3 & Hunyuan Dense optimization (KV cache, GQA, fused RoPE, GGUF, batched decode) |
| 2026.01.30 | PaddleOCR-VL 1.5 support |
| 2025.03.21 | Qwen 2.5 with transformers-like Rust interface |
| 2025.03.19 | Project initialized |

---

## Citation

```bibtex
@misc{Crane,
  author       = {lucasjinreal},
  title        = {{Crane: Candle-based Rust Accelerated Neural Engine}},
  howpublished = {\url{https://github.com/lucasjinreal/Crane}},
  year         = {2025}
}
```

## License

See [LICENSE](LICENSE) for details.
