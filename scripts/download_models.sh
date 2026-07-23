#!/usr/bin/env bash
# Download all model weights required by tripoflux-mlx.
#
# Usage:
#   ./scripts/download_models.sh [--dir DIR] [--hf-token TOKEN]
#
# Options:
#   --dir DIR        Target directory for checkpoints (default: ./ckpts)
#   --hf-token TOKEN Hugging Face access token (default: $HF_TOKEN)
#
# Requirements:
#   - huggingface-cli (pip install huggingface-hub)
#   - wget or curl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT_DIR="${PROJECT_ROOT}/ckpts"
HF_TOKEN="${HF_TOKEN:-}"

usage() {
    grep '^#' "$0" | grep -v '#!' | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            CKPT_DIR="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

if [[ -n "${HF_TOKEN}" ]]; then
    export HF_TOKEN
    echo "Using HF_TOKEN from environment/CLI"
fi

mkdir -p "${CKPT_DIR}"

echo "=============================================="
echo " TripoFlux MLX - Model Downloader"
echo "=============================================="
echo "Checkpoint directory: ${CKPT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Helper: download a file from Hugging Face if it does not already exist.
# ---------------------------------------------------------------------------
download_hf_file() {
    local repo_id="$1"
    local filename="$2"
    local target_dir="$3"
    local target_path="${target_dir}/${filename}"

    mkdir -p "${target_dir}"

    if [[ -f "${target_path}" ]]; then
        echo "[skip] ${filename} already exists"
        return 0
    fi

    echo "[downloading] ${repo_id}/${filename} → ${target_path}"

    if command -v hf &> /dev/null; then
        hf download "${repo_id}" "${filename}" \
            --local-dir "${target_dir}"
    elif command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "${repo_id}" "${filename}" \
            --local-dir "${target_dir}" \
            --local-dir-use-symlinks False
    elif command -v wget &> /dev/null; then
        local url="https://huggingface.co/${repo_id}/resolve/main/${filename}"
        wget -O "${target_path}" "${url}"
    elif command -v curl &> /dev/null; then
        local url="https://huggingface.co/${repo_id}/resolve/main/${filename}"
        curl -L -o "${target_path}" "${url}"
    else
        echo "ERROR: neither hf, huggingface-cli, wget, nor curl is available" >&2
        return 1
    fi
}

# ---------------------------------------------------------------------------
# 1. TripoSplat weights
# ---------------------------------------------------------------------------
TRIPOSPLAT_REPO="VAST-AI/TripoSplat"
TRIPOSPLAT_DIR="${CKPT_DIR}/VAST-AI/TripoSplat"

echo ""
echo ">>> Downloading TripoSplat weights (${TRIPOSPLAT_REPO})"
download_hf_file "${TRIPOSPLAT_REPO}" "diffusion_models/triposplat_fp16.safetensors" "${TRIPOSPLAT_DIR}"
download_hf_file "${TRIPOSPLAT_REPO}" "clip_vision/dino_v3_vit_h.safetensors" "${TRIPOSPLAT_DIR}"
download_hf_file "${TRIPOSPLAT_REPO}" "vae/triposplat_vae_decoder_fp16.safetensors" "${TRIPOSPLAT_DIR}"
download_hf_file "${TRIPOSPLAT_REPO}" "vae/flux2-vae.safetensors" "${TRIPOSPLAT_DIR}"
download_hf_file "${TRIPOSPLAT_REPO}" "background_removal/birefnet.safetensors" "${TRIPOSPLAT_DIR}"

# ---------------------------------------------------------------------------
# 2. FLUX.2-klein-9B (single-file checkpoint used by mflux)
# ---------------------------------------------------------------------------
FLUX_REPO="black-forest-labs/FLUX.2-klein-9B"
FLUX_DIR="${CKPT_DIR}/black-forest-labs/FLUX.2-klein-9B"

echo ""
echo ">>> Downloading FLUX.2-klein-9B (${FLUX_REPO})"
echo "    Note: this is a gated model. You must accept the license at"
echo "    https://huggingface.co/${FLUX_REPO} before downloading."
echo "    (mflux will also auto-download on first use; pre-downloading here for offline use)"

download_hf_file "${FLUX_REPO}" "model_index.json" "${FLUX_DIR}" || true
download_hf_file "${FLUX_REPO}" "flux-2-klein-9b.safetensors" "${FLUX_DIR}" || true

# ---------------------------------------------------------------------------
# 3. BiRefNet standalone (optional fallback)
# ---------------------------------------------------------------------------
BIREFNET_REPO="ZhengPeng7/BiRefNet"
BIREFNET_DIR="${CKPT_DIR}/ZhengPeng7/BiRefNet"

echo ""
echo ">>> Downloading BiRefNet standalone (${BIREFNET_REPO})"
download_hf_file "${BIREFNET_REPO}" "model.safetensors" "${BIREFNET_DIR}" || true

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Download complete"
echo "=============================================="
echo ""
echo "Expected layout:"
echo "  ${TRIPOSPLAT_DIR}/"
echo "    diffusion_models/triposplat_fp16.safetensors"
echo "    clip_vision/dino_v3_vit_h.safetensors"
echo "    vae/triposplat_vae_decoder_fp16.safetensors"
echo "    vae/flux2-vae.safetensors"
echo "    background_removal/birefnet.safetensors"
echo "  ${FLUX_DIR}/"
echo "    flux-2-klein-9b.safetensors (single-file checkpoint for mflux)"
echo "  ${BIREFNET_DIR}/"
echo "    model.safetensors (optional standalone BiRefNet)"
echo ""
echo "To start the server:"
echo "  python -m tripoflux.server"
