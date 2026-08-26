#!/usr/bin/env bash
# Provision a RunPod GPU pod to run pdbench Config A.
# Assumes an NVIDIA CUDA + PyTorch base image (RunPod "vLLM" or "PyTorch" template).
#
#   bash scripts/setup_runpod.sh gpt-oss-20b
#   bash scripts/setup_runpod.sh gpt-oss-120b
#
# Then run, e.g.:
#   python run.py --config A --model gpt-oss-20b
set -euo pipefail

MODEL_KEY="${1:-gpt-oss-20b}"
case "$MODEL_KEY" in
  gpt-oss-20b)  HF="openai/gpt-oss-20b" ;;
  gpt-oss-120b) HF="openai/gpt-oss-120b" ;;
  *) echo "unknown model key: $MODEL_KEY"; exit 1 ;;
esac

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "== installing vLLM + harness deps =="
pip install -q --upgrade pip
# vLLM brings the CUDA torch build; pin a recent version known to serve gpt-oss.
pip install -q "vllm>=0.10.1"
pip install -q httpx matplotlib numpy "transformers>=4.44"

echo "== pre-downloading model weights ($HF) =="
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$HF", allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja"])
print("downloaded to", p)
PY

echo "== done. example run =="
echo "  python run.py --config A --model $MODEL_KEY --quick     # smoke"
echo "  python run.py --config A --model $MODEL_KEY             # full matrix + max-extent"
