# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file Modal deployment (`modal_llm.py`) that runs a self-hosted,
OpenAI-compatible **vLLM** server on a GPU. Its only consumer is the sibling
repo **ScrapeFlow** (`../scrapeflow`, GitHub `kargovin/scrapeflow`), whose
`llm-worker` sends schema-constrained extraction requests over the OpenAI wire
protocol. This repo owns the *inference endpoint*; ScrapeFlow owns everything
that calls it.

## Commands

```bash
modal setup                    # one-time browser auth
modal serve  modal_llm.py      # dev: live reload, tears down on Ctrl-C
modal deploy modal_llm.py      # persistent deployment
modal app logs scrapeflow-llm  # tail server + vLLM logs
python smoke_test.py           # end-to-end check (needs VLLM_ENDPOINT + VLLM_API_KEY)
```

There is no build, lint, or unit-test suite — the deliverable is a running Modal
app. `smoke_test.py` is the integration check; it drives the endpoint exactly as
ScrapeFlow's worker does (json_schema `response_format`).

## Required Modal resources (must exist before deploy)

- Volume `model-cache` — caches the ~8GB weights across cold starts.
- Volume `vllm-cache` — persists FlashInfer/torch.compile JIT caches (mounted at
  `~/.cache`) so cold starts skip recompilation. Auto-created (`create_if_missing`).
- Secret `llm-secrets` with `VLLM_API_KEY` (bearer token ScrapeFlow sends).
  No `HF_TOKEN` — Qwen3 is open-weights.

## The contract with ScrapeFlow — do not break it

ScrapeFlow's worker (`../scrapeflow/llm-worker/worker/llm.py`,
`_call_openai_compatible`) calls this endpoint with:

- `AsyncOpenAI(base_url=<endpoint>/v1, api_key=<VLLM_API_KEY>)`
- `chat.completions.create(..., response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": <output_schema>}})`
- It then does `json.loads(response.choices[0].message.content)`.

Implications when changing this repo:
- The `--model` value **is** the `model` string ScrapeFlow sends. Changing the
  model requires coordinating the model name on both sides.
- The vLLM version must keep supporting `response_format: json_schema` structured
  outputs. If that call returns non-JSON content, ScrapeFlow jobs fail. Verify
  with `smoke_test.py` after any version bump.
- Auth is a single shared bearer token (`--api-key`); there is no per-user auth
  here — access control lives in ScrapeFlow.

## Cost / scaling constraints (Modal free tier, $30/mo)

- `min_containers=0` (scale-to-zero) is mandatory. **Never set it to 1** — an
  always-on L4 is ~$24/day and drains the monthly credit in about a day.
- `scaledown_window=300` trades ~5 min of idle warmth for hidden cold starts
  during active sessions. Cold start with cached weights is a few seconds
  (small model); first-ever download is ~1–2 min.
- GPU/model sizing: L4 (24GB) holds Qwen3-4B FP16 (~8GB) + KV cache with room to
  spare. Model choice is deliberate — extraction needs no reasoning, so a small
  *non-thinking* `-Instruct-2507` model is fastest/cheapest. Scaling up (e.g.
  `Qwen3-8B`, or the base hybrid models) means re-checking VRAM/GPU and, for
  hybrid Qwen3, suppressing `<think>` output. See the sizing table in
  ScrapeFlow's `docs/guides/modal-llm-inference.md`, the original spec for this repo.
