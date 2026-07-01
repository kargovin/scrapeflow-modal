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

# use hf transfer to speed up the cold start when initially downloading the models
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm>=0.10.0", "huggingface_hub[hf_transfer]>=0.27.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("scrapeflow-llm")

# Persist downloaded weights (~8GB) across cold starts.
model_volume = modal.Volume.from_name("model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS_DIR: model_volume},
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
