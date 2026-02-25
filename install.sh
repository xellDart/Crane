#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# Crane — One-Line Installer
#
# Install and build Crane with a single command:
#
#   curl -fsSL https://raw.githubusercontent.com/xellDart/Crane/main/install.sh | bash
#
# Or with options:
#   curl -fsSL ... | bash -s -- --model qwen3-vl-2b --start
#   curl -fsSL ... | bash -s -- --cpu
#   curl -fsSL ... | bash -s -- --dir /opt/crane
#
# What it does:
#   1. Installs Rust (if not found)
#   2. Installs huggingface-cli (if model download requested)
#   3. Clones Crane (or updates if already present)
#   4. Auto-detects CUDA, GPU, Flash Attention
#   5. Builds with optimal features
#   6. Downloads model weights (optional)
#   7. Prints the command to start the server
# ─────────────────────────────────────────────────────────────────────

REPO_URL="https://github.com/xellDart/Crane.git"
BRANCH="main"
INSTALL_DIR=""
MODEL=""
MODEL_PATH_ARG=""
MODEL_TYPE_ARG=""
FORCE_CPU=false
START_AFTER=false
DAEMON_MODE=false
PORT=8080

for arg in "$@"; do
  case "$arg" in
    --model=*)       MODEL="${arg#--model=}" ;;
    --model)         ;; # handled below
    --model-path=*)  MODEL_PATH_ARG="${arg#--model-path=}" ;;
    --model-path)    ;; # handled below
    --model-type=*)  MODEL_TYPE_ARG="${arg#--model-type=}" ;;
    --model-type)    ;; # handled below
    --dir=*)         INSTALL_DIR="${arg#--dir=}" ;;
    --dir)           ;; # handled below
    --branch=*)      BRANCH="${arg#--branch=}" ;;
    --port=*)        PORT="${arg#--port=}" ;;
    --port)          ;; # handled below
    --cpu)           FORCE_CPU=true ;;
    --start)         START_AFTER=true ;;
    --daemon)        START_AFTER=true; DAEMON_MODE=true ;;
    -h|--help)
      cat <<'HELP'
Crane Installer — build a high-performance inference server in one command.

Usage:
  curl -fsSL https://raw.githubusercontent.com/xellDart/Crane/main/install.sh | bash
  curl -fsSL ... | bash -s -- [OPTIONS]

Options:
  --model <name>        Download model after build. Supported:
                          qwen3-vl-2b    Qwen3-VL-2B (vision-language, ~4.5 GB)
                          qwen3-8b       Qwen3-8B (text, ~16 GB)
                          hunyuan-7b     HunyuanDense-7B (text, ~14 GB)
  --model-path <path>   Use an existing local model directory
  --model-type <type>   Model type (qwen3, qwen3_vl, hunyuan). Auto-detected if --model used
  --dir <path>          Install directory (default: ./Crane)
  --branch <name>       Git branch to checkout (default: main)
  --cpu                 Force CPU-only build (skip CUDA detection)
  --start               Start the server in foreground after build
  --daemon              Start the server as a background daemon after build
  --port <port>         Server port (default: 8080)
  -h, --help            Show this help

Examples:
  # Just build (auto-detect GPU)
  curl -fsSL ... | bash

  # Build + download Qwen3-VL-2B + start server
  curl -fsSL ... | bash -s -- --model qwen3-vl-2b --start

  # Build + start as daemon with existing model
  curl -fsSL ... | bash -s -- --model-path /path/to/model --model-type qwen3_vl --daemon --port 9090

  # Install to custom directory
  curl -fsSL ... | bash -s -- --dir /opt/crane --model qwen3-8b
HELP
      exit 0
      ;;
    *)
      if [ "${PREV_ARG:-}" = "--model" ]; then MODEL="$arg"; fi
      if [ "${PREV_ARG:-}" = "--model-path" ]; then MODEL_PATH_ARG="$arg"; fi
      if [ "${PREV_ARG:-}" = "--model-type" ]; then MODEL_TYPE_ARG="$arg"; fi
      if [ "${PREV_ARG:-}" = "--dir" ]; then INSTALL_DIR="$arg"; fi
      if [ "${PREV_ARG:-}" = "--port" ]; then PORT="$arg"; fi
      ;;
  esac
  PREV_ARG="$arg"
done

# ── Helpers ──────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${BLUE}${BOLD}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}${BOLD}[ OK ]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}${BOLD}[WARN]${RESET}  $*"; }
fail()  { echo -e "${RED}${BOLD}[FAIL]${RESET}  $*"; exit 1; }

elapsed() {
  local secs=$1
  printf "%dm %02ds" $((secs / 60)) $((secs % 60))
}

# ── Banner ───────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}"
cat << 'BANNER'
   ██████╗██████╗  █████╗ ███╗   ██╗███████╗
  ██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔════╝
  ██║     ██████╔╝███████║██╔██╗ ██║█████╗
  ██║     ██╔══██╗██╔══██║██║╚██╗██║██╔══╝
  ╚██████╗██║  ██║██║  ██║██║ ╚████║███████╗
   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝
BANNER
echo -e "${RESET}"
echo "  High-performance inference engine"
echo "  https://github.com/xellDart/Crane"
echo ""
echo "────────────────────────────────────────────────"
echo ""

# ═════════════════════════════════════════════════════════════════════
#  Step 1: Prerequisites
# ═════════════════════════════════════════════════════════════════════

info "Checking prerequisites..."

# ── Git ──
if ! command -v git &>/dev/null; then
  fail "git is required. Install it: https://git-scm.com/"
fi
ok "git: $(git --version)"

# ── Rust ──
if command -v cargo &>/dev/null; then
  ok "Rust: $(rustc --version 2>/dev/null)"
else
  info "Rust not found. Installing via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
  if command -v cargo &>/dev/null; then
    ok "Rust installed: $(rustc --version 2>/dev/null)"
  else
    fail "Rust installation failed. Install manually: https://rustup.rs/"
  fi
fi

# ── CUDA ──
HAS_CUDA=false
HAS_FLASH_ATTN=false
GPU_COMPUTE_CAP=""
GPU_NAME=""
CUDA_VERSION=""

if [ "$FORCE_CPU" = false ]; then
  if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA: $CUDA_VERSION"
    HAS_CUDA=true

    if command -v nvidia-smi &>/dev/null; then
      GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
      GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^[[:space:]]*//')

      if [ -n "$GPU_NAME" ]; then
        ok "GPU: $GPU_NAME (sm_${GPU_COMPUTE_CAP//./_})"
      fi

      if [ -n "$GPU_COMPUTE_CAP" ]; then
        MAJOR=$(echo "$GPU_COMPUTE_CAP" | cut -d. -f1)
        if [ "$MAJOR" -ge 8 ] 2>/dev/null; then
          HAS_FLASH_ATTN=true
          ok "Flash Attention: supported (SM_80+)"
        fi
      fi
    fi
  else
    warn "CUDA not found. Building for CPU only."
    warn "For GPU: install CUDA Toolkit and ensure nvcc is in PATH."
  fi
else
  info "CPU-only build requested (--cpu)"
fi

# ── huggingface-cli (if model download requested) ──
if [ -n "$MODEL" ]; then
  if ! command -v huggingface-cli &>/dev/null; then
    info "Installing huggingface-cli for model download..."
    pip install -q huggingface_hub 2>/dev/null || pip3 install -q huggingface_hub 2>/dev/null || {
      warn "Could not install huggingface-cli. You'll need to download models manually."
      MODEL=""
    }
  fi
  if command -v huggingface-cli &>/dev/null; then
    ok "huggingface-cli: ready"
  fi
fi

echo ""

# ═════════════════════════════════════════════════════════════════════
#  Step 2: Clone / Update Repository
# ═════════════════════════════════════════════════════════════════════

# Detect if we're already inside the Crane repo
INSIDE_REPO=false
if [ -f "Cargo.toml" ] && [ -d "crane-core" ]; then
  INSIDE_REPO=true
  CRANE_DIR="$(pwd)"
elif [ -f "../Cargo.toml" ] && [ -d "../crane-core" ]; then
  INSIDE_REPO=true
  CRANE_DIR="$(cd .. && pwd)"
fi

if [ "$INSIDE_REPO" = true ]; then
  info "Already inside Crane repository: $CRANE_DIR"
  cd "$CRANE_DIR"
  info "Pulling latest changes..."
  git pull --ff-only 2>/dev/null || warn "Could not pull (uncommitted changes?). Continuing with local version."
else
  [ -z "$INSTALL_DIR" ] && INSTALL_DIR="Crane"
  if [ -d "$INSTALL_DIR" ] && [ -d "$INSTALL_DIR/crane-core" ]; then
    info "Crane directory exists at $INSTALL_DIR, updating..."
    cd "$INSTALL_DIR"
    git pull --ff-only 2>/dev/null || warn "Could not pull. Continuing with local version."
  else
    info "Cloning Crane..."
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
  fi
  CRANE_DIR="$(pwd)"
fi

ok "Crane directory: $CRANE_DIR"
echo ""

# ═════════════════════════════════════════════════════════════════════
#  Step 3: Build
# ═════════════════════════════════════════════════════════════════════

FEATURES=()

if [ "$HAS_FLASH_ATTN" = true ]; then
  # flash-attn implies cuda in Cargo.toml
  FEATURES+=("flash-attn")
  FA_CACHE_DIR="/tmp/crane_flash_attn_cache"
  mkdir -p "$FA_CACHE_DIR"
  export CANDLE_FLASH_ATTN_BUILD_DIR="$FA_CACHE_DIR"
elif [ "$HAS_CUDA" = true ]; then
  FEATURES+=("cuda")
fi

FEATURES_FLAG=""
if [ ${#FEATURES[@]} -gt 0 ]; then
  FEATURES_STR=$(IFS=,; echo "${FEATURES[*]}")
  FEATURES_FLAG="--features $FEATURES_STR"
  ok "Build features: $FEATURES_STR"
else
  info "Build features: CPU-only"
fi

BUILD_START=$(date +%s)

info "Building Crane (release)..."
if [ "$HAS_FLASH_ATTN" = true ]; then
  info "First build with Flash Attention takes ~10 min (CUTLASS compilation)."
  info "Subsequent builds are much faster (cached)."
fi

# shellcheck disable=SC2086
cargo build --release $FEATURES_FLAG 2>&1

BUILD_END=$(date +%s)
BUILD_ELAPSED=$((BUILD_END - BUILD_START))

echo ""
ok "Build completed in $(elapsed $BUILD_ELAPSED)"

# Count built binaries
BUILT=0
for bin in crane-oai chat_simple qwen3_vl_simple hunyuan_simple; do
  [ -f "target/release/$bin" ] && BUILT=$((BUILT + 1))
done
ok "$BUILT binaries in target/release/"
echo ""

# ═════════════════════════════════════════════════════════════════════
#  Step 4: Download Model (optional)
# ═════════════════════════════════════════════════════════════════════

MODEL_PATH=""
MODEL_TYPE=""

# If --model-path was provided, use it directly
if [ -n "$MODEL_PATH_ARG" ]; then
  MODEL_PATH="$(cd "$(dirname "$MODEL_PATH_ARG")" 2>/dev/null && pwd)/$(basename "$MODEL_PATH_ARG")" 2>/dev/null || MODEL_PATH="$MODEL_PATH_ARG"
  MODEL_TYPE="${MODEL_TYPE_ARG:-auto}"
  if [ -d "$MODEL_PATH" ]; then
    ok "Using local model: $MODEL_PATH (type: $MODEL_TYPE)"
  else
    fail "Model path does not exist: $MODEL_PATH"
  fi
fi

if [ -n "$MODEL" ]; then
  case "$MODEL" in
    qwen3-vl-2b|qwen3_vl_2b)
      HF_ID="Qwen/Qwen3-VL-2B"
      MODEL_DIR="checkpoints/qwen3_vl_2b"
      MODEL_TYPE="qwen3_vl"
      ;;
    qwen3-8b|qwen3_8b)
      HF_ID="Qwen/Qwen3-8B"
      MODEL_DIR="checkpoints/qwen3_8b"
      MODEL_TYPE="qwen3"
      ;;
    hunyuan-7b|hunyuan_7b)
      HF_ID="tencent/Hunyuan-A13B-Instruct"
      MODEL_DIR="checkpoints/hunyuan_7b"
      MODEL_TYPE="hunyuan"
      ;;
    *)
      # Treat as HuggingFace model ID directly
      HF_ID="$MODEL"
      MODEL_DIR="checkpoints/$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
      MODEL_TYPE="auto"
      ;;
  esac

  MODEL_PATH="$CRANE_DIR/$MODEL_DIR"

  if [ -d "$MODEL_DIR" ] && ls "$MODEL_DIR"/*.safetensors &>/dev/null 2>&1; then
    ok "Model already downloaded: $MODEL_DIR"
  elif command -v huggingface-cli &>/dev/null; then
    info "Downloading model: $HF_ID"
    info "Destination: $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    huggingface-cli download "$HF_ID" --local-dir "$MODEL_DIR"
    ok "Model downloaded: $MODEL_DIR"
  else
    warn "huggingface-cli not available. Download manually:"
    echo ""
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download $HF_ID --local-dir $MODEL_DIR"
    echo ""
  fi
fi

echo ""

# ═════════════════════════════════════════════════════════════════════
#  Step 5: Start Server (optional)
# ═════════════════════════════════════════════════════════════════════

SERVER_STARTED=false
CRANE_BIN="$CRANE_DIR/target/release/crane-oai"
CRANE_LOG="$CRANE_DIR/crane-oai.log"
CRANE_PID_FILE="$CRANE_DIR/crane-oai.pid"

if [ "$START_AFTER" = true ] && [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ]; then
  if [ ! -f "$CRANE_BIN" ]; then
    fail "crane-oai binary not found at $CRANE_BIN"
  fi

  if [ "$DAEMON_MODE" = true ]; then
    # ── Daemon mode: run in background with nohup ──
    info "Starting Crane daemon on port $PORT..."

    # Kill existing daemon if running
    if [ -f "$CRANE_PID_FILE" ]; then
      OLD_PID=$(cat "$CRANE_PID_FILE" 2>/dev/null)
      if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        warn "Stopping existing Crane daemon (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 1
      fi
    fi

    nohup "$CRANE_BIN" \
      --model-path "$MODEL_PATH" \
      --model-type "$MODEL_TYPE" \
      --port "$PORT" \
      > "$CRANE_LOG" 2>&1 &

    DAEMON_PID=$!
    echo "$DAEMON_PID" > "$CRANE_PID_FILE"

    # Wait a moment and verify it started
    sleep 3
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
      ok "Crane daemon started (PID $DAEMON_PID)"
      ok "Logs:     tail -f $CRANE_LOG"
      ok "Stop:     kill \$(cat $CRANE_PID_FILE)"
      SERVER_STARTED=true
    else
      warn "Daemon may have failed to start. Check logs:"
      echo "  tail -20 $CRANE_LOG"
    fi
  else
    # ── Foreground mode: exec replaces this process ──
    info "Starting Crane server on port $PORT (foreground)..."
    echo ""
    exec "$CRANE_BIN" \
      --model-path "$MODEL_PATH" \
      --model-type "$MODEL_TYPE" \
      --port "$PORT"
  fi
fi

echo ""

# ═════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════"
echo "  Crane Installation Complete"
echo "════════════════════════════════════════════════"
echo ""
echo "  Location:    $CRANE_DIR"
echo "  Binary:      $CRANE_BIN"
echo "  Platform:    $(uname -s) $(uname -m)"
echo "  Rust:        $(rustc --version 2>/dev/null)"
if [ "$HAS_CUDA" = true ]; then
  echo "  CUDA:        $CUDA_VERSION"
  [ -n "$GPU_NAME" ] && echo "  GPU:         $GPU_NAME"
  echo "  Flash Attn:  $([ "$HAS_FLASH_ATTN" = true ] && echo "enabled" || echo "disabled")"
else
  echo "  CUDA:        not detected (CPU build)"
fi
echo "  Build time:  $(elapsed $BUILD_ELAPSED)"
if [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ]; then
  echo "  Model:       $MODEL_PATH"
  echo "  Model type:  $MODEL_TYPE"
fi
if [ "$SERVER_STARTED" = true ]; then
  echo "  Server:      http://localhost:$PORT (daemon, PID $(cat "$CRANE_PID_FILE"))"
fi
echo ""

# ── How to start the server ──
echo -e "${BOLD}  How to start the server:${RESET}"
echo ""
SHOW_MODEL_PATH="${MODEL_PATH:-/path/to/model}"
SHOW_MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"

echo "    # Foreground"
echo "    $CRANE_BIN \\"
echo "      --model-path $SHOW_MODEL_PATH \\"
echo "      --model-type $SHOW_MODEL_TYPE \\"
echo "      --port $PORT"
echo ""
echo "    # Daemon (background)"
echo "    nohup $CRANE_BIN \\"
echo "      --model-path $SHOW_MODEL_PATH \\"
echo "      --model-type $SHOW_MODEL_TYPE \\"
echo "      --port $PORT > $CRANE_LOG 2>&1 &"
echo ""
echo "    # Or use the installer:"
if [ -n "$MODEL_PATH" ]; then
  echo "    bash install.sh --model-path $MODEL_PATH --model-type $SHOW_MODEL_TYPE --daemon --port $PORT"
else
  echo "    bash install.sh --model qwen3-vl-2b --daemon --port $PORT"
fi
echo ""

# ── How to manage the daemon ──
echo -e "${BOLD}  Manage daemon:${RESET}"
echo ""
echo "    tail -f $CRANE_LOG        # View logs"
echo "    kill \$(cat $CRANE_PID_FILE)  # Stop server"
echo "    curl localhost:$PORT/health     # Health check"
echo ""

# ── How to test ──
echo -e "${BOLD}  Test the API:${RESET}"
echo ""
if [ "$SHOW_MODEL_TYPE" = "qwen3_vl" ]; then
  cat << CURL_VL
    curl http://localhost:$PORT/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
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
CURL_VL
else
  cat << CURL_TEXT
    curl http://localhost:$PORT/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"crane","messages":[{"role":"user","content":"Hello!"}]}'
CURL_TEXT
fi
echo ""

# ── Download models ──
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  echo -e "${BOLD}  Download a model:${RESET}"
  echo ""
  echo "    pip install huggingface_hub"
  echo "    huggingface-cli download Qwen/Qwen3-VL-2B --local-dir $CRANE_DIR/checkpoints/qwen3_vl_2b"
  echo ""
  echo "  Supported model types: qwen3, qwen3_vl, hunyuan"
  echo ""
fi

echo "  Re-run installer:  curl -fsSL https://raw.githubusercontent.com/xellDart/Crane/main/install.sh | bash"
echo "  Rebuild only:      cd $CRANE_DIR && bash build.sh"
echo ""
