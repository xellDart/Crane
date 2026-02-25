# Crane

High-performance inference engine in Rust. Single binary, OpenAI-compatible API, CUDA + Flash Attention.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/xellDart/Crane/main/install.sh | bash
```

Auto-detects CUDA, GPU, Flash Attention. Installs Rust if needed.

```bash
# Build + download model + start
curl -fsSL ... | bash -s -- --model qwen3-vl-2b --daemon

# With existing model
curl -fsSL ... | bash -s -- --model-path /path/to/model --model-type qwen3_vl --daemon
```

## Build

```bash
git clone https://github.com/xellDart/Crane.git && cd Crane

# Auto-detect
bash build.sh

# Or manual
cargo build --release --features flash-attn   # CUDA + Flash Attention (SM_80+)
cargo build --release --features cuda          # CUDA only
cargo build --release                          # CPU only
```

## Server

```bash
crane-ctl start --model-path checkpoints/qwen3_vl_2b --model-type qwen3_vl
crane-ctl start --model-path checkpoints/qwen3_vl_2b --model-type qwen3_vl --port 9090
crane-ctl status
crane-ctl log
crane-ctl restart
crane-ctl stop
```

| Command | |
|---------|---|
| `start` | Start daemon (`-f` for foreground) |
| `stop` | Stop server |
| `restart` | Restart (remembers last config) |
| `status` | PID, memory, health |
| `log` | Tail logs |
| `health` | `GET /health` |

## API

OpenAI-compatible. Works with any OpenAI SDK.

```bash
# Text
curl localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"crane","messages":[{"role":"user","content":"Hello"}]}'

# Vision
curl localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"crane",
    "messages":[{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"https://example.com/img.jpg"}},
      {"type":"text","text":"Describe this image"}
    ]}]
  }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="x")
r = client.chat.completions.create(model="crane", messages=[{"role":"user","content":"Hi"}])
print(r.choices[0].message.content)
```

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat (streaming & non-streaming) |
| `POST /v1/completions` | Text completions |
| `POST /v1/audio/speech` | TTS |
| `GET /v1/models` | List models |
| `GET /health` | Health check |
| `POST /generate` | SGLang generate |

## Models

| Model | `--model-type` |
|-------|----------------|
| Qwen3 (0.6B—30B+) | `qwen3` |
| Qwen 2.5 (0.5B—72B) | `qwen25` |
| Hunyuan Dense | `hunyuan` |
| Qwen3-VL (2B, 4B) | `qwen3_vl` |
| PaddleOCR-VL | `paddleocr_vl` |
| Qwen3-TTS | `qwen3_tts` |

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-2B --local-dir checkpoints/qwen3_vl_2b
```

## Optimizations

- Flash Attention v2 (SM_80+)
- Fused `add + RMSNorm` CUDA kernel
- Pre-allocated KV cache
- GQA 4D matmul
- Fused RoPE
- Continuous batching
- GGUF quantization

## License

See [LICENSE](LICENSE).
