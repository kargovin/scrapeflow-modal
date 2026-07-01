# scrapeflow-modal

Self-hosted, OpenAI-compatible **vLLM inference server on [Modal](https://modal.com)**,
serving structured-extraction requests from [ScrapeFlow](https://github.com/kargovin/scrapeflow)'s
`llm-worker`.

ScrapeFlow's worker already speaks the OpenAI wire protocol, so no worker code
changes are needed — a job is configured with `provider=openai_compatible`, this
deployment's `base_url`, the shared API key, and the model name.

- **Model:** `Qwen/Qwen3-4B-Instruct-2507` — a *non-thinking* instruct model
  (no `<think>` reasoning blocks), ideal for schema extraction. Apache-2.0 open
  weights, so no `HF_TOKEN` needed.
- **GPU:** L4 (24GB) — Qwen3-4B FP16 (~8GB) fits with large KV-cache headroom
- **Scaling:** scale-to-zero (`min_containers=0`), 5-min warm window
- **API:** OpenAI-compatible (`/v1/chat/completions`, `/v1/models`), bearer-token auth,
  `response_format={"type": "json_schema", ...}` structured outputs

## Prerequisites

```bash
pip install -r requirements.txt
modal setup                       # browser auth (one time)
modal volume create model-cache   # weight cache across cold starts
modal secret create llm-secrets VLLM_API_KEY=<random-token>
# (No HF_TOKEN needed — Qwen3 is open weights.)
```

## Deploy

```bash
modal serve  modal_llm.py   # dev: live reload, tears down on Ctrl-C
modal deploy modal_llm.py   # persistent deployment
modal app logs scrapeflow-llm
```

Endpoint: `https://<workspace>--scrapeflow-llm-serve.modal.run` → use `<endpoint>/v1`
as the OpenAI `base_url`.

## Verify

```bash
curl https://<endpoint>/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
python smoke_test.py   # requires VLLM_ENDPOINT and VLLM_API_KEY env vars
```

## Cost note

Never set `min_containers=1` on the free tier — an always-on L4 burns ~$24/day
and exhausts the $30 monthly credit in about a day. See
`docs/guides/modal-llm-inference.md` in the ScrapeFlow repo for the full cost model.
