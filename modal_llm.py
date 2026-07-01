"""Self-hosted, OpenAI-compatible vLLM inference server on Modal.

Serves an OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, ...) that
ScrapeFlow's `llm-worker` calls via the OpenAI SDK with `base_url` pointed here.
Scale-to-zero keeps idle cost near zero on Modal's free tier.

Deploy:   modal deploy modal_llm.py
Dev:      modal serve modal_llm.py   (live reload; shuts down on Ctrl-C)
Logs:     modal app logs scrapeflow-llm
"""

import os
import subprocess

import modal

# --- Configuration ----------------------------------------------------------
# L4 (24GB) easily fits Qwen3-4B in FP16 (~8GB) with large KV-cache headroom.
GPU = "L4"
# The "-Instruct-2507" variant is a *non-thinking* instruct model: unlike the
# base Qwen3 models it never emits <think> reasoning blocks, which is exactly
# what we want for schema extraction (no reasoning needed, faster, cheaper).
# Apache-2.0 open weights — no HF gate / HF_TOKEN required.
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODELS_DIR = "/models"
PORT = 8000

# vLLM must be recent enough to support Qwen3 (2507 variants) AND OpenAI
# `response_format={"type": "json_schema", ...}` structured outputs (xgrammar) —
# the latter is what ScrapeFlow relies on for schema-constrained extraction.
#
# Base off the CUDA *-devel* image, not debian_slim: vLLM's FlashInfer backend
# JIT-compiles attention kernels at runtime and needs `nvcc` + the CUDA toolkit.
# debian_slim has only the CUDA runtime (bundled in the torch wheel), so engine
# init fails with "Could not find nvcc / /usr/local/cuda doesn't exist".
# `.entrypoint([])` clears the base image's entrypoint so Modal controls startup.
# HF_HUB_ENABLE_HF_TRANSFER speeds up the one-time weight download on cold start.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install("vllm==0.21.0", "huggingface_hub[hf_transfer]>=0.27.1")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Keep torch-inductor's cache under ~/.cache (default is a /tmp dir)
            # so it lands on the persisted cache volume mounted there below.
            "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor",
        }
    )
)

app = modal.App("scrapeflow-llm")

CACHE_DIR = "/root/.cache"

# Persist downloaded weights (~8GB) across cold starts.
model_volume = modal.Volume.from_name("model-cache", create_if_missing=True)
# Persist JIT/compile caches (FlashInfer kernels, vLLM torch.compile artifacts)
# across cold starts — these all default to ~/.cache, so mounting a volume there
# lets a cold container reuse them instead of recompiling (~10–30s) every time.
cache_volume = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS_DIR: model_volume, CACHE_DIR: cache_volume},
    # llm-secrets provides VLLM_API_KEY (the bearer token ScrapeFlow sends).
    # Qwen3 is open-weights, so no HF_TOKEN is needed to download it.
    secrets=[modal.Secret.from_name("llm-secrets")],
    timeout=600,
    min_containers=0,  # scale-to-zero when idle — do NOT set to 1 on free tier
    max_containers=1,  # cost guardrail: never fan out to a 2nd L4 (~$1/hr each)
    scaledown_window=300,  # stay warm 5 min after last request to hide cold starts
)
# Let one container field many simultaneous requests so vLLM's continuous
# batching handles them on a single GPU, instead of Modal cold-starting another
# L4 per concurrent request. 32 is generous headroom over ScrapeFlow's ~3
# concurrent workers (LLM_MAX_WORKERS).
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=600)
def serve():
    subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL,
            "--download-dir",
            MODELS_DIR,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--api-key",
            os.environ["VLLM_API_KEY"],
            "--gpu-memory-utilization",
            "0.90",
            # ScrapeFlow truncates content to 50k chars (~14k tokens); leave room
            # for the schema + up to 4k output tokens. Qwen3 supports far more,
            # but a tight window keeps KV-cache memory small on the L4.
            "--max-model-len",
            "32768",
        ]
    )
