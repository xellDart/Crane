#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# Crane — Autonomous Build Script
#
# Auto-detects the environment and builds with optimal features:
#   - CUDA support (if nvcc found)
#   - Flash Attention (if CUDA + GPU compute capability >= 8.0)
#   - cuDNN (if --cudnn flag passed)
#   - MKL (if --mkl flag passed)
#
# Usage:
#   bash build.sh                    # Full auto-detect build
#   bash build.sh --cpu              # Force CPU-only (no CUDA)
#   bash build.sh --no-flash-attn    # CUDA but skip Flash Attention
#   bash build.sh --cudnn            # Enable cuDNN
#   bash build.sh --mkl              # Enable Intel MKL
#   bash build.sh --clean            # Clean build artifacts first
#   bash build.sh --bin crane-oai    # Build only a specific binary
# ─────────────────────────────────────────────────────────────────────

FORCE_CPU=false
NO_FLASH_ATTN=false
ENABLE_CUDNN=false
ENABLE_MKL=false
CLEAN_FIRST=false
SPECIFIC_BIN=""

for arg in "$@"; do
  case "$arg" in
    --cpu)            FORCE_CPU=true ;;
    --no-flash-attn)  NO_FLASH_ATTN=true ;;
    --cudnn)          ENABLE_CUDNN=true ;;
    --mkl)            ENABLE_MKL=true ;;
    --clean)          CLEAN_FIRST=true ;;
    --bin=*)          SPECIFIC_BIN="${arg#--bin=}" ;;
    --bin)            ;; # handled below with next arg
    -h|--help)
      echo "Usage: bash build.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --cpu              Force CPU-only build (ignore CUDA)"
      echo "  --no-flash-attn    Disable Flash Attention even if GPU supports it"
      echo "  --cudnn            Enable cuDNN support"
      echo "  --mkl              Enable Intel MKL support"
      echo "  --clean            Run cargo clean before building"
      echo "  --bin <name>       Build only a specific binary"
      echo "  -h, --help         Show this help"
      exit 0
      ;;
    *)
      # Handle --bin <name> (space-separated)
      if [ "${PREV_ARG:-}" = "--bin" ]; then
        SPECIFIC_BIN="$arg"
      fi
      ;;
  esac
  PREV_ARG="$arg"
done

# ── Helpers ──────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m  $*"; exit 1; }

elapsed() {
  local secs=$1
  printf "%dm %02ds" $((secs / 60)) $((secs % 60))
}

# ── Step 1: Prerequisites ────────────────────────────────────────────

echo ""
info "Crane Build System"
echo "────────────────────────────────────────────────"

# Find project root (script may be invoked from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/Cargo.toml" ] && [ -d "$SCRIPT_DIR/crane-core" ]; then
  CRANE_DIR="$SCRIPT_DIR"
else
  fail "Cannot find Crane workspace. Run this script from the Crane root directory."
fi
cd "$CRANE_DIR"
info "Project root: $CRANE_DIR"

# Rust
command -v cargo &>/dev/null || fail "Rust/Cargo not found. Install: https://rustup.rs/"
RUSTC_VERSION=$(rustc --version 2>/dev/null)
ok "Rust: $RUSTC_VERSION"

# ── Step 2: Detect CUDA & GPU ────────────────────────────────────────

HAS_CUDA=false
HAS_FLASH_ATTN=false
GPU_COMPUTE_CAP=""
GPU_NAME=""

if [ "$FORCE_CPU" = false ]; then
  if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA Toolkit: $CUDA_VERSION"
    HAS_CUDA=true

    # Detect GPU compute capability via nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
      GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
      GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^[[:space:]]*//')

      if [ -n "$GPU_NAME" ]; then
        ok "GPU: $GPU_NAME (sm_${GPU_COMPUTE_CAP//./_})"
      fi

      # Flash Attention requires SM_80+ (Ampere, Ada Lovelace, Hopper)
      if [ -n "$GPU_COMPUTE_CAP" ] && [ "$NO_FLASH_ATTN" = false ]; then
        MAJOR=$(echo "$GPU_COMPUTE_CAP" | cut -d. -f1)
        if [ "$MAJOR" -ge 8 ] 2>/dev/null; then
          HAS_FLASH_ATTN=true
          ok "Flash Attention: enabled (compute capability $GPU_COMPUTE_CAP >= 8.0)"
        else
          warn "Flash Attention: disabled (compute capability $GPU_COMPUTE_CAP < 8.0, needs Ampere+)"
        fi
      fi
    else
      warn "nvidia-smi not found. CUDA enabled but cannot detect GPU capability."
      warn "Flash Attention disabled (cannot verify SM_80+ support)."
    fi
  else
    warn "nvcc not found. Building for CPU only."
    warn "For GPU support, install CUDA Toolkit and ensure nvcc is in PATH."
  fi
else
  info "CPU-only build requested (--cpu)"
fi

if [ "$NO_FLASH_ATTN" = true ] && [ "$HAS_CUDA" = true ]; then
  info "Flash Attention: explicitly disabled (--no-flash-attn)"
fi

echo ""

# ── Step 3: Assemble Features ────────────────────────────────────────

FEATURES=()

if [ "$HAS_CUDA" = true ]; then
  FEATURES+=("cuda")
fi

if [ "$HAS_FLASH_ATTN" = true ]; then
  FEATURES+=("flash-attn")

  # Create flash-attn build cache directory (CUTLASS compilation artifacts)
  FA_CACHE_DIR="/tmp/crane_flash_attn_cache"
  mkdir -p "$FA_CACHE_DIR"
  export CANDLE_FLASH_ATTN_BUILD_DIR="$FA_CACHE_DIR"
  info "Flash Attention build cache: $FA_CACHE_DIR"
fi

if [ "$ENABLE_CUDNN" = true ]; then
  FEATURES+=("cudnn")
fi

if [ "$ENABLE_MKL" = true ]; then
  FEATURES+=("mkl")
fi

# Build the --features flag
FEATURES_FLAG=""
if [ ${#FEATURES[@]} -gt 0 ]; then
  FEATURES_STR=$(IFS=,; echo "${FEATURES[*]}")
  FEATURES_FLAG="--features $FEATURES_STR"
  ok "Build features: $FEATURES_STR"
else
  info "Build features: (none, CPU-only)"
fi

echo ""

# ── Step 4: Clean (optional) ─────────────────────────────────────────

if [ "$CLEAN_FIRST" = true ]; then
  info "Cleaning build artifacts..."
  cargo clean 2>/dev/null || true
  ok "Clean complete"
  echo ""
fi

# ── Step 5: Build ────────────────────────────────────────────────────

BUILD_START=$(date +%s)

if [ -n "$SPECIFIC_BIN" ]; then
  info "Building binary: $SPECIFIC_BIN"
  # shellcheck disable=SC2086
  cargo build --release $FEATURES_FLAG --bin "$SPECIFIC_BIN" 2>&1
else
  info "Building full workspace (release)..."
  if [ "$HAS_FLASH_ATTN" = true ]; then
    info "First build with Flash Attention takes ~10 min (CUTLASS compilation)."
    info "Subsequent builds use cached artifacts and are much faster."
  fi
  # shellcheck disable=SC2086
  cargo build --release $FEATURES_FLAG 2>&1
fi

BUILD_END=$(date +%s)
BUILD_ELAPSED=$((BUILD_END - BUILD_START))

echo ""
ok "Build completed in $(elapsed $BUILD_ELAPSED)"

# ── Step 6: Verify Binaries ──────────────────────────────────────────

echo ""
info "Built binaries:"
BINARIES=(
  "crane-oai"
  "chat_simple"
  "chat_streaming"
  "asr_simple"
  "vision_simple"
  "ocr_simple"
  "tts_simple"
  "tts_custom_voice"
  "tts_voice_clone"
  "hunyuan_simple"
  "llava_simple"
  "qwen3_vl_simple"
  "bm_resize"
)

FOUND_COUNT=0
for bin in "${BINARIES[@]}"; do
  if [ -f "target/release/$bin" ]; then
    SIZE=$(du -h "target/release/$bin" | cut -f1)
    echo "  ./target/release/$bin  ($SIZE)"
    FOUND_COUNT=$((FOUND_COUNT + 1))
  fi
done

if [ "$FOUND_COUNT" -eq 0 ]; then
  warn "No binaries found in target/release/"
else
  ok "$FOUND_COUNT binaries ready"
fi

# ── Step 7: Verify Warnings ──────────────────────────────────────────

echo ""
info "Checking for compiler warnings..."
# shellcheck disable=SC2086
WARNING_COUNT=$(cargo build --release $FEATURES_FLAG 2>&1 | grep -c "^warning\[" || true)
if [ "$WARNING_COUNT" -eq 0 ]; then
  ok "Zero warnings"
else
  warn "$WARNING_COUNT warning(s) detected"
fi

# ── Summary ──────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════"
echo "  Crane Build Complete"
echo "════════════════════════════════════════════════"
echo ""
echo "  Platform:    $(uname -s) $(uname -m)"
echo "  Rust:        $RUSTC_VERSION"
if [ "$HAS_CUDA" = true ]; then
  echo "  CUDA:        $CUDA_VERSION"
  [ -n "$GPU_NAME" ] && echo "  GPU:         $GPU_NAME (sm_${GPU_COMPUTE_CAP//./_})"
  echo "  Flash Attn:  $([ "$HAS_FLASH_ATTN" = true ] && echo "enabled" || echo "disabled")"
else
  echo "  CUDA:        disabled"
fi
echo "  Features:    $([ ${#FEATURES[@]} -gt 0 ] && echo "$FEATURES_STR" || echo "none")"
echo "  Build time:  $(elapsed $BUILD_ELAPSED)"
echo "  Binaries:    $FOUND_COUNT"
echo ""
