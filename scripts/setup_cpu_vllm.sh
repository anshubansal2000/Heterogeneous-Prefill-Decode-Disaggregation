#!/usr/bin/env bash
# Build/install the vLLM CPU backend in an isolated venv (for Config E prefill).
# The CPU backend cannot coexist with the GPU vLLM in one env, so we keep it in
# /workspace/venv-cpu. Requires an AVX-512 x86 host (AMX on Xeon SPR helps a lot).
#
#   bash scripts/setup_cpu_vllm.sh
# Then serve on CPU:
#   /workspace/venv-cpu/bin/vllm serve Qwen/Qwen3-8B --device cpu --dtype bfloat16 ...
set -uo pipefail

echo "== CPU features =="
grep -o -m1 -E 'avx512[a-z_]*|amx[a-z_]*' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
echo "== cores / RAM =="; nproc; free -g | awk 'NR<=2'

python3 -m venv /workspace/venv-cpu --system-site-packages
source /workspace/venv-cpu/bin/activate
pip install -q --upgrade pip

# Try the prebuilt CPU wheel first (fast); fall back to source build.
echo "== attempting vLLM CPU install =="
if pip install -q vllm --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null && \
   python -c "import vllm" 2>/dev/null; then
  echo "CPU_WHEEL_OK"
else
  echo "wheel path failed; trying VLLM_TARGET_DEVICE=cpu source build (slow)"
  pip install -q cmake ninja setuptools-scm
  VLLM_TARGET_DEVICE=cpu pip install -q --no-build-isolation \
      "git+https://github.com/vllm-project/vllm.git" 2>&1 | tail -20
fi

python -c "import vllm; print('CPU_VLLM', vllm.__version__)" || { echo CPU_VLLM_FAILED; exit 1; }
pip install -q httpx numpy transformers huggingface_hub
echo CPU_SETUP_COMPLETE
