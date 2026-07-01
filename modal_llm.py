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
# A10G (24GB) fits Llama-3.1-8B in FP16 (16GB) with KV-cache headroom.
GPU = "A10G"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODELS_DIR = "/models"
PORT = 8000

# vLLM >= 0.6 supports OpenAI `response_format={"type": "json_schema", ...}` via
# structured outputs (xgrammar backend) — this is what ScrapeFlow relies on for
# schema-constrained extraction, so pin a version known to support it.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.6.6", "huggingface_hub[hf_transfer]==0.27.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("scrapeflow-llm")

# Persist downloaded weights (~16GB) across cold starts.
model_volume = modal.Volume.from_name("model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS_DIR: model_volume},
    # llm-secrets provides VLLM_API_KEY (bearer token ScrapeFlow sends) and
    # HF_TOKEN (required to pull the gated Llama weights).
    secrets=[modal.Secret.from_name("llm-secrets")],
    timeout=600,
    min_containers=0,  # scale-to-zero when idle — do NOT set to 1 on free tier
    scaledown_window=300,  # stay warm 5 min after last request to hide cold starts
)
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
            "--max-model-len",
            "8192",
        ]
    )
