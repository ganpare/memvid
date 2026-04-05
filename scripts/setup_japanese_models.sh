#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-sungo-ganpare/memvid-embedding-models}"
MODELS_DIR="${1:-${XDG_CACHE_HOME:-$HOME/.cache}/memvid/text-models}"
FORCE="${FORCE:-0}"

download_file() {
  local url="$1"
  local out="$2"

  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    echo "Skipping existing file: $out"
    return
  fi

  mkdir -p "$(dirname "$out")"
  echo "Downloading $url"
  curl -L "$url" -o "$out"
}

BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"

download_file "${BASE_URL}/multilingual-e5-large/multilingual-e5-large.onnx" "${MODELS_DIR}/multilingual-e5-large.onnx"
download_file "${BASE_URL}/multilingual-e5-large/model.onnx_data" "${MODELS_DIR}/model.onnx_data"
download_file "${BASE_URL}/multilingual-e5-large/multilingual-e5-large_tokenizer.json" "${MODELS_DIR}/multilingual-e5-large_tokenizer.json"
download_file "${BASE_URL}/ruri-pt-large/ruri-pt-large.onnx" "${MODELS_DIR}/ruri-pt-large.onnx"
download_file "${BASE_URL}/ruri-pt-large/vocab.txt" "${MODELS_DIR}/vocab.txt"
download_file "${BASE_URL}/ruri-pt-large/tokenizer_config.json" "${MODELS_DIR}/tokenizer_config.json"
download_file "${BASE_URL}/ruri-pt-large/special_tokens_map.json" "${MODELS_DIR}/special_tokens_map.json"

echo
echo "Japanese embedding models are ready in: ${MODELS_DIR}"
echo "Use TextEmbedConfig::multilingual_e5_large() or TextEmbedConfig::ruri_pt_large()."
