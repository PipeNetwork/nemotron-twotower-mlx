#!/bin/bash
# Download one MLX model into a local folder.
# Usage:
#   scripts/download.sh ar   4bit          -> AR/context tower (stock mlx-lm)
#   scripts/download.sh diff 4bit ./tt-4bit -> full TwoTower diffusion (custom code)
set -euo pipefail

KIND="${1:-diff}"; QUANT="${2:-4bit}"; DEST="${3:-}"

if [[ "$KIND" == "ar" ]]; then
  REPO="pipenetwork/Nemotron-3-Nano-30B-A3B-context-mlx-${QUANT}"
  DEST="${DEST:-./ar-${QUANT}}"
elif [[ "$KIND" == "diff" ]]; then
  REPO="pipenetwork/Nemotron-Labs-TwoTower-30B-A3B-mlx-${QUANT}"
  DEST="${DEST:-./tt-${QUANT}}"
else
  echo "usage: $0 <ar|diff> <4bit|6bit|8bit|bf16> [dest]"; exit 2
fi

echo "Downloading $REPO -> $DEST"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$REPO" --local-dir "$DEST"
echo "Done: $DEST"
