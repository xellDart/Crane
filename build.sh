#!/usr/bin/env bash
# build.sh — Compila Crane (ColQwen3 embeddings library + CLI).
#
# Defaults: clean + cuda + flash-attn (auto-detectados). Output → ./release/
#
# Uso:
#   ./build.sh                # default: clean, cuda, flash-attn
#   ./build.sh --no-clean     # incremental
#   ./build.sh --cpu          # CPU-only (sin CUDA, sin flash-attn)
#   ./build.sh --no-flash     # CUDA pero sin flash-attn (Pascal/Volta/Turing)

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$(pwd)"

DO_CLEAN=1
WANT_CUDA=1
WANT_FA=1

for arg in "$@"; do
  case "$arg" in
    --no-clean) DO_CLEAN=0 ;;
    --cpu)      WANT_CUDA=0; WANT_FA=0 ;;
    --no-flash) WANT_FA=0 ;;
    -h|--help)
      sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yel()   { printf "\033[33m%s\033[0m\n" "$*"; }

# ── 1. Rust toolchain ────────────────────────────────────────────────
bold "[1/5] Rust"
if ! command -v cargo >/dev/null 2>&1; then
  yel "  cargo no encontrado — instalando rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "${CARGO_HOME:-$HOME/.cargo}/env"
fi
green "  $(cargo --version)"

# ── 2. Detectar CUDA + flash-attn ────────────────────────────────────
bold "[2/5] CUDA / flash-attn"
CUDA_OK=0
FA_OK=0

if [[ "$WANT_CUDA" == "1" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    green "  nvcc: $(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | head -1)"
    CUDA_OK=1
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    green "  nvcc añadido al PATH desde /usr/local/cuda/bin"
    CUDA_OK=1
  else
    yel "  nvcc no encontrado → CPU-only"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    green "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  fi
fi

if [[ "$CUDA_OK" == "1" && "$WANT_FA" == "1" ]]; then
  CC=""
  command -v nvidia-smi >/dev/null 2>&1 \
    && CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')"
  if [[ -n "$CC" && "$CC" -ge 80 ]]; then
    green "  compute_cap=$CC ≥ 80 → flash-attn-2 ON"
    FA_OK=1
  else
    yel "  compute_cap=$CC < 80 → flash-attn OFF (necesita Ampere/Ada/Hopper)"
  fi
fi

FEATURES=""
[[ "$CUDA_OK" == "1" ]] && FEATURES="cuda"
[[ "$FA_OK"   == "1" ]] && FEATURES="cuda,flash-attn"

# ── 3. Clean ──────────────────────────────────────────────────────────
bold "[3/5] Clean"
if [[ "$DO_CLEAN" == "1" ]]; then
  cargo clean
  green "  target/ limpio"
else
  yel "  --no-clean → build incremental"
fi

# ── 4. Build (lib + CLI) ─────────────────────────────────────────────
bold "[4/5] Build (features: ${FEATURES:-<none>})"
CARGO_FLAGS=(--release)
[[ -n "$FEATURES" ]] && CARGO_FLAGS+=(--features "$FEATURES")

# crane-core (lib) — para que nebuia-embs lo consuma
cargo build "${CARGO_FLAGS[@]}" -p crane-core
# CLI
cargo build "${CARGO_FLAGS[@]}" -p crane-examples --bin embedding_simple

# ── 5. Empaquetar en release/ ────────────────────────────────────────
bold "[5/5] release/"
rm -rf release
mkdir -p release
install -m 755 target/release/embedding_simple release/

cat > release/README.txt <<EOF
crane (ColQwen3 embeddings)

Contenido:
  embedding_simple   CLI: ranking sobre directorio de imágenes

Uso:
  ./embedding_simple <model_path> <images_dir> <query> [--top-k N] [--bf16]

Ejemplo:
  ./embedding_simple ./colqwen3-4b ./data/images "tabla de accionistas" --top-k 4 --bf16

La librería crane-core se consume desde otros workspaces vía:
  crane-core = { path = "../Crane/crane-core" }
EOF

echo
green "  ✓ Build OK"
ls -la release/ | tail -n +2 | awk '{print "    " $NF}'
