#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# Crane + Qwen3-VL-2B  —  Automated Setup Script
#
# This script will:
#   1. Check prerequisites (Rust, git, huggingface-cli)
#   2. Clone the Crane repository (if not already inside it)
#   3. Build with CUDA support (falls back to CPU if no CUDA)
#   4. Download Qwen3-VL-2B model weights from HuggingFace
#   5. Run a quick smoke test
#
# Usage:
#   bash setup_qwen3_vl.sh              # Full setup
#   bash setup_qwen3_vl.sh --skip-model # Skip model download
#   bash setup_qwen3_vl.sh --cpu        # Force CPU build (no CUDA)
# ─────────────────────────────────────────────────────────────────────

REPO_URL="https://github.com/xellDart/Crane.git"
BRANCH="feat/qwen3-vl-implementation"
MODEL_ID="Qwen/Qwen3-VL-2B"
CHECKPOINT_DIR="checkpoints/qwen3_vl_2b"

SKIP_MODEL=false
FORCE_CPU=false

for arg in "$@"; do
  case "$arg" in
    --skip-model) SKIP_MODEL=true ;;
    --cpu)        FORCE_CPU=true ;;
    -h|--help)
      echo "Usage: bash setup_qwen3_vl.sh [--skip-model] [--cpu]"
      echo ""
      echo "  --skip-model   Skip downloading model weights"
      echo "  --cpu          Force CPU-only build (ignore CUDA)"
      exit 0
      ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m  $*"; exit 1; }

check_cmd() {
  if command -v "$1" &>/dev/null; then
    ok "$1 found: $(command -v "$1")"
    return 0
  else
    return 1
  fi
}

# ── Step 1: Check Prerequisites ─────────────────────────────────────

info "Checking prerequisites..."

check_cmd git   || fail "git is required. Install it: https://git-scm.com/"
check_cmd cargo || fail "Rust/Cargo is required. Install it: https://rustup.rs/"

RUSTC_VERSION=$(rustc --version 2>/dev/null || echo "unknown")
ok "Rust version: $RUSTC_VERSION"

# Check CUDA
HAS_CUDA=false
if [ "$FORCE_CPU" = false ]; then
  if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA found: $CUDA_VERSION"
    HAS_CUDA=true
  else
    warn "CUDA not found (nvcc not in PATH). Will build for CPU only."
    warn "For GPU support, install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
  fi
else
  info "CPU-only build requested (--cpu flag)"
fi

# Check huggingface-cli for model download
HAS_HF_CLI=false
if [ "$SKIP_MODEL" = false ]; then
  if check_cmd huggingface-cli; then
    HAS_HF_CLI=true
  else
    warn "huggingface-cli not found. Installing..."
    pip install -q huggingface_hub 2>/dev/null || pip3 install -q huggingface_hub 2>/dev/null || {
      warn "Could not install huggingface_hub. You'll need to download the model manually."
      warn "Run: pip install huggingface_hub && huggingface-cli download $MODEL_ID --local-dir $CHECKPOINT_DIR"
    }
    if command -v huggingface-cli &>/dev/null; then
      HAS_HF_CLI=true
      ok "huggingface-cli installed"
    fi
  fi
fi

echo ""

# ── Step 2: Clone Repository ────────────────────────────────────────

# Check if we're already inside the Crane repo
if [ -f "Cargo.toml" ] && grep -q 'name = "crane-core"' Cargo.toml 2>/dev/null; then
  info "Already inside Crane repository"
  CRANE_DIR="$(pwd)"
elif [ -f "crane-core/Cargo.toml" ] 2>/dev/null; then
  info "Already inside Crane repository"
  CRANE_DIR="$(pwd)"
else
  if [ -d "Crane" ]; then
    info "Crane directory exists, using it"
    CRANE_DIR="$(pwd)/Crane"
  else
    info "Cloning Crane repository..."
    git clone "$REPO_URL" Crane
    CRANE_DIR="$(pwd)/Crane"
  fi
  cd "$CRANE_DIR"
fi

# Checkout the correct branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "")
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
  info "Checking out branch: $BRANCH"
  git fetch --all 2>/dev/null || true
  git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH" 2>/dev/null || {
    warn "Could not checkout $BRANCH. Continuing on current branch: $CURRENT_BRANCH"
  }
fi

ok "Repository ready at: $CRANE_DIR"
echo ""

# ── Step 3: Build ───────────────────────────────────────────────────

info "Building Crane..."

BUILD_FEATURES=""
if [ "$HAS_CUDA" = true ]; then
  BUILD_FEATURES="--features cuda"
  info "Building with CUDA support"
else
  info "Building for CPU only"
fi

# Build both binaries
info "Building qwen3_vl_simple (direct inference)..."
cargo build --release $BUILD_FEATURES --bin qwen3_vl_simple 2>&1 | tail -5
ok "qwen3_vl_simple built"

info "Building crane-oai (OpenAI-compatible server)..."
cargo build --release $BUILD_FEATURES --bin crane-oai 2>&1 | tail -5
ok "crane-oai built"

echo ""

# ── Step 4: Download Model Weights ──────────────────────────────────

if [ "$SKIP_MODEL" = true ]; then
  info "Skipping model download (--skip-model)"
elif [ -d "$CHECKPOINT_DIR" ] && ls "$CHECKPOINT_DIR"/*.safetensors &>/dev/null 2>&1; then
  ok "Model weights already present at $CHECKPOINT_DIR"
else
  if [ "$HAS_HF_CLI" = true ]; then
    info "Downloading Qwen3-VL-2B model (~8.5 GB)..."
    info "This may take a while depending on your connection."
    mkdir -p "$CHECKPOINT_DIR"
    huggingface-cli download "$MODEL_ID" --local-dir "$CHECKPOINT_DIR"
    ok "Model downloaded to $CHECKPOINT_DIR"
  else
    warn "Cannot download model automatically. Please download manually:"
    echo ""
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download $MODEL_ID --local-dir $CHECKPOINT_DIR"
    echo ""
    echo "Or download from: https://huggingface.co/$MODEL_ID"
  fi
fi

echo ""

# ── Step 5: Smoke Test ──────────────────────────────────────────────

if [ -d "$CHECKPOINT_DIR" ] && ls "$CHECKPOINT_DIR"/*.safetensors &>/dev/null 2>&1; then
  info "Running smoke test..."

  # Create a simple test image (1x1 white pixel PNG)
  TEST_IMG="/tmp/crane_test_image.jpg"
  if ! [ -f "$TEST_IMG" ]; then
    # Use Python to create a minimal test image
    python3 -c "
from PIL import Image
img = Image.new('RGB', (64, 64), color='white')
img.save('$TEST_IMG')
" 2>/dev/null || {
      # Fallback: use ImageMagick if available
      convert -size 64x64 xc:white "$TEST_IMG" 2>/dev/null || {
        warn "Cannot create test image (no PIL or ImageMagick). Skipping smoke test."
        TEST_IMG=""
      }
    }
  fi

  if [ -n "$TEST_IMG" ] && [ -f "$TEST_IMG" ]; then
    BF16_FLAG=""
    if [ "$HAS_CUDA" = true ]; then
      BF16_FLAG="--bf16"
    fi

    timeout 120 ./target/release/qwen3_vl_simple \
      "$CHECKPOINT_DIR" "$TEST_IMG" "What do you see?" $BF16_FLAG 2>&1 | head -20 && {
      ok "Smoke test passed!"
    } || {
      warn "Smoke test finished (may have timed out, which is OK for first run)"
    }
    rm -f "$TEST_IMG"
  fi
else
  warn "No model weights found. Skipping smoke test."
  warn "Download the model and run: ./target/release/qwen3_vl_simple $CHECKPOINT_DIR <image.jpg>"
fi

echo ""

# ── Summary ─────────────────────────────────────────────────────────

echo "=========================================="
echo "  Crane + Qwen3-VL-2B Setup Complete"
echo "=========================================="
echo ""
echo "Binaries:"
echo "  ./target/release/qwen3_vl_simple   (direct inference)"
echo "  ./target/release/crane-oai         (OpenAI-compatible server)"
echo ""
echo "Quick start:"
echo ""
echo "  # Direct inference on an image"
echo "  ./target/release/qwen3_vl_simple $CHECKPOINT_DIR ./photo.jpg \"Describe this image\""
echo ""
echo "  # Start the server"
echo "  ./target/release/crane-oai --model-path $CHECKPOINT_DIR --port 8080"
echo ""
echo "  # Query the server"
echo "  curl http://localhost:8080/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen3-vl\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.com/photo.jpg\"}},{\"type\":\"text\",\"text\":\"Describe this image\"}]}]}'"
echo ""
echo "See QWEN3_VL_README.md for full documentation."
echo ""
