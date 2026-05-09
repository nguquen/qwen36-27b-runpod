# Qwen3.6-27B vLLM Docker — RunPod Serverless edition

Docker-based vLLM serving for [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) with [Lorbus AutoRound INT4 quant](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) and MTP speculative decoding.

This fork adds **RunPod Serverless Load Balancer** support: a small Caddy sidecar provides the `/ping` health probe RunPod requires and proxies the OpenAI-compatible API with SSE streaming preserved. The image still runs unchanged on bare-metal Docker / Compose.

Upstream: [tedivm/qwen36-27b-docker](https://github.com/tedivm/qwen36-27b-docker) (originally forked from [k0zakinio/qwen36-vllm-setup](https://github.com/k0zakinio/qwen36-vllm-setup)).

## What you get

| Metric (dual RTX 3090, TP=2, 200K context) | Value                                        |
| ------------------------------------------ | -------------------------------------------- |
| Sustained TPS on coding workloads          | **~118**                                     |
| Sustained TPS on prose                     | ~89                                          |
| Max context length                         | 200,000 tokens (172K KV pool headroom)       |
| Vision support                             | yes (MoonViT, via `image_url` content parts) |

### Verified benchmarks (2x RTX 3090, TP=2)

Measured with `bench_tps.py` against the Docker container:

| Workload               | Tokens | Time   | TPS        |
| ---------------------- | ------ | ------ | ---------- |
| Prose (800-word story) | 800    | 8.99s  | **88.98**  |
| Code (LRU cache impl)  | 1200   | 10.16s | **118.13** |

Dense model, not MoE — no tensor shuffling at token boundaries, clean TP=2 split. No NVLink required; PCIe TP is fine on this workload.

## Hardware

Tested on 2x RTX 3090 (48 GB VRAM total). Also works at lower context on a single 24 GB card. GPU count is auto-detected — pass `--gpus all` for multi-GPU or `--gpus '"device=0"'` for single-GPU.

Minimum disk: ~20 GB for weights + ~6 GB for caches.

## Quick start

### Option 1: Docker Compose (recommended)

```bash
# Download model (one-time, stores in ./models)
docker compose run --rm qwen36 download

# Start the server
docker compose up -d

# Stop
docker compose down
```

### Option 2: Docker CLI

```bash
# Build locally
docker build -t qwen36-vllm .

# Or pull from GHCR
docker pull ghcr.io/tedivm/qwen36-27b-docker:latest

# Download model (one-time, stores in /path/on/host/models)
docker run --rm --gpus all \
  -v /path/on/host/models:/data/models \
  qwen36-vllm download

# Start server (auto-detects 1 or 2 GPUs)
docker run -d --name qwen36 --gpus all -p 1234:1234 \
  -v /path/on/host/models:/data/models \
  qwen36-vllm

# Single-GPU override
docker run -d --name qwen36 --gpus '"device=0"' -p 1234:1234 \
  -v /path/on/host/models:/data/models \
  qwen36-vllm

# Upgrade (no redownload needed)
docker stop qwen36 && docker rm qwen36
docker pull ghcr.io/tedivm/qwen36-27b-docker:latest
docker run -d --name qwen36 --gpus all -p 1234:1234 \
  -v /path/on/host/models:/data/models \
  qwen36-vllm
```

## Environment variables

All configuration is via environment variables with sensible defaults:

| Variable                              | Default                             | Description                                                             |
| ------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------- |
| `MODEL_DIR`                           | `/data/models`                      | Model weights path (mount from host)                                    |
| `MODEL_REPO`                          | `Lorbus/Qwen3.6-27B-int4-AutoRound` | HuggingFace model repo                                                  |
| `PORT`                                | `1234`                              | API port                                                                |
| `SERVED_MODEL_NAME`                   | `qwen3.6-27b`                       | Model name for API                                                      |
| `MAX_MODEL_LEN`                       | `200000`                            | Max context length (auto-lowered to 32768 for single GPU)               |
| `MAX_NUM_SEQS`                        | `3`                                 | Concurrent sequences                                                    |
| `MAX_NUM_BATCHED_TOKENS`              | `4128`                              | Prefill chunk size; raise to 8192/16384 on 48 GB GPUs for lower TTFT    |
| `GPU_MEMORY_UTIL`                     | `0.92`                              | GPU memory fraction (auto-set to 0.95 for single GPU)                   |
| `TENSOR_PARALLEL`                     | _(auto)_                            | Tensor parallelism (auto-detected from GPU count)                       |
| `TEMPERATURE`                         | `0.6`                               | Generation temperature                                                  |
| `TOP_P`                               | `0.95`                              | Nucleus sampling threshold                                              |
| `TOP_K`                               | `20`                                | Top-k sampling                                                          |
| `MIN_P`                               | `0.0`                               | Min-p sampling threshold                                                |
| `PRESENCE_PENALTY`                    | `0`                                 | Presence penalty                                                        |
| `REPETITION_PENALTY`                  | `1.0`                               | Repetition penalty                                                      |
| `REASONING_PARSER`                    | `qwen3`                             | Reasoning parser (blank to disable)                                     |
| `HF_TOKEN`                            | _(empty)_                           | HuggingFace auth token for gated models                                 |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`  | _(empty)_                           | OTel traces endpoint (e.g. `grpc://otel-collector:4317`)                |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | _(empty)_                           | OTel metrics endpoint (e.g. `grpc://otel-collector:4317`)               |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`    | _(empty)_                           | OTel logs endpoint (e.g. `grpc://otel-collector:4317`)                  |
| `OTEL_EXPORTER_OTLP_INSECURE`         | _(empty)_                           | Set to `true` for plaintext gRPC (OTel SDK env, applies to all signals) |
| `OTEL_EXPORTER_OTLP_TRACES_INSECURE`  | _(empty)_                           | Traces-only plaintext override (OTel SDK env)                           |
| `OTEL_EXPORTER_OTLP_METRICS_INSECURE` | _(empty)_                           | Metrics-only plaintext override (OTel SDK env)                          |
| `OTEL_EXPORTER_OTLP_LOGS_INSECURE`    | _(empty)_                           | Logs-only plaintext override (OTel SDK env)                             |

## Usage

**Monitor** (inside the container):

```bash
docker exec -it qwen36 watch-vllm.py
docker exec -it qwen36 watch-vllm.py /data/models/.tmp/vllm.log 24
```

**Benchmark** (inside the container):

```bash
docker exec -it qwen36 python bench_tps.py
```

**OpenAI-compatible API**:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Authorization: Bearer any" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}]}'
```

## Server flags

| Flag                                                           | Why                                                                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--quantization auto_round`                                    | Matches the Lorbus weights                                                                                                                                                                                                                                                                                                                |
| `--kv-cache-dtype fp8`                                         | FP8 KV cache (~2x KV pool vs FP16, lossless in practice). TurboQuant (`turboquant_k8v4` etc.) is supported on Qwen3.6 only by vLLM's post-#39931 wheel which currently OOMs at cold start on 1×4090 24 GB; the image stays on stable vLLM + fp8 until a tagged release with #39931 ships and is verified. Override with `KV_CACHE_DTYPE`. |
| `--enable-prefix-caching`                                      | Not default for Qwen3.6 hybrid attention; opt in                                                                                                                                                                                                                                                                                          |
| `--enable-chunked-prefill`                                     | Recommended alongside spec-decode for throughput                                                                                                                                                                                                                                                                                          |
| `--speculative-config method=mtp, num_speculative_tokens=3`    | ~2x throughput on code; 3 is the sweet spot                                                                                                                                                                                                                                                                                               |
| `--max-num-seqs 3`                                             | Solo user + subagents; raise for more concurrency                                                                                                                                                                                                                                                                                         |
| `--max-num-batched-tokens 4128`                                | Prefill chunk size with `--enable-chunked-prefill`; matches vLLM's CUDA-graph capture endpoint (`4096 + 32`). Override with `MAX_NUM_BATCHED_TOKENS`. Raising trades activation memory for fewer chunks per long prompt — `8192` on 48 GB cards halves TTFT but adds ~600 MB peak; on 24 GB it tends to OOM at cold start.                |
| `--gpu-memory-utilization 0.92`                                | Leaves CUDA-graph margin                                                                                                                                                                                                                                                                                                                  |
| `--disable-custom-all-reduce`                                  | No NVLink — stock NCCL is faster                                                                                                                                                                                                                                                                                                          |
| `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice` | OpenAI-style tool calls                                                                                                                                                                                                                                                                                                                   |
| `--reasoning-parser qwen3`                                     | Enables extended thinking output                                                                                                                                                                                                                                                                                                          |

TP=2 beats TP=1 by ~1.5x on dual 3090s. Memory-bandwidth savings from splitting weights across two cards outweigh the PCIe NCCL all-reduce cost.

## Deploy on RunPod Serverless (Load Balancer)

The image bundles a Caddy sidecar that exposes `:8000` with:

- `GET /ping` — returns **200** only when vLLM has finished loading the model (probes upstream `/v1/models`, not `/health` — see below), **204** while it's still initializing (so the RunPod LB holds traffic instead of marking the worker failed).
- Everything else (`/v1/*`, `/metrics`, `/tokenize`, …) — reverse-proxied to vLLM on `127.0.0.1:1234` with response buffering disabled, so SSE token streams flush per chunk.

vLLM still binds to `1234` internally; nothing else changed for bare-metal users.

### Architecture

```
[RunPod LB] ──:8000──► [caddy] ──/ping──► 200 if upstream /v1/models=200, else 204
                          │
                          └──/* (incl. /v1/*) ──► [vLLM :1234 on 127.0.0.1]
```

### Step 1 — Pre-populate a Network Volume (one-time)

Cold-starting a 27B model on every fresh worker is brutal. Put the weights on a RunPod Network Volume so workers in the same DC mount them instantly.

1. RunPod console → **Storage → New Network Volume**: name `qwen36-weights`, ~25 GB, in your target datacenter.
2. Spin up a temporary GPU pod (any 24 GB+ card), attach the volume to `/data/models`, then run:
   ```bash
   docker run --rm --gpus all \
     -v /data/models:/data/models \
     -e MODEL_REPO=Lorbus/Qwen3.6-27B-int4-AutoRound \
     ghcr.io/nguquen/qwen36-27b-runpod:latest \
     download
   ```
   (Or build this repo locally and substitute the image tag.)
3. Destroy the temporary pod once `Download complete.` prints.

### Step 2 — Create the Serverless endpoint

RunPod console → **Serverless → New Endpoint → Import from GitHub**:

| Setting           | Value                                                 |
| ----------------- | ----------------------------------------------------- |
| Repository        | `nguquen/qwen36-27b-runpod`                           |
| Branch            | `main`                                                |
| Dockerfile path   | `Dockerfile` (root)                                   |
| **Endpoint Type** | **Load Balancer**                                     |
| **Internal port** | `8000`                                                |
| GPU               | `RTX 4090` (24 GB) or `RTX A6000` (48 GB)             |
| Network Volume    | `qwen36-weights` → mount at `/data/models`            |
| Workers           | min `0`, max `2–3`                                    |
| Idle timeout      | `60s` (cold starts are expensive — keep workers warm) |

**Environment variables:**

| Variable                 | Recommended                                | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------ | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MODEL_DOWNLOAD`         | `0`                                        | Volume is pre-populated. Set `1` to self-heal on a fresh volume (adds ~60–120s to cold start).                                                                                                                                                                                                                                                                                                                                                              |
| `MAX_NUM_SEQS`           | `3`                                        | Conservative; raise for higher throughput if memory allows.                                                                                                                                                                                                                                                                                                                                                                                                 |
| `MAX_NUM_BATCHED_TOKENS` | `4128` (default; safe on 24 GB)            | Prefill chunk size with `--enable-chunked-prefill`. `4128` (= `4096 + 32`, vLLM's CUDA-graph capture endpoint) is what fits on 1×4090 24 GB alongside the fp8 KV pool at MML 32768. On 1×A6000 48 GB raise to `8192` (~600 MB more activation, ~halved TTFT for long prompts); on TP=2 or 48 GB+ you can push `16384`. Raising past `4128` on 24 GB tends to OOM at cold start (vLLM reserves activation buffers sized to this value during `profile_run`). |
| `MAX_MODEL_LEN`          | `32768` (4090, auto) / `65536` (A6000 48G) | `docker-entrypoint.sh` auto-clamps to `32768` on single GPU when the upstream image default (`200000`) is detected. With fp8 KV (~85 B/token) and ~5 GB KV budget on a 24 GB card after weights/GDN/activations/cudagraph, 32K is the safe ceiling. Set explicitly to override.                                                                                                                                                                             |
| `GPU_MEMORY_UTIL`        | `0.95` (auto on single GPU)                | Auto-raised from upstream `0.92` → `0.95` on single GPU. Lower to `0.93` if you OOM at boot with vision or longer context.                                                                                                                                                                                                                                                                                                                                  |
| `CHAT_TEMPLATE_KWARGS`   | `{"preserve_thinking": true}`              | JSON dict passed as vLLM's `--default-chat-template-kwargs`. Default keeps prior `<think>` blocks in the prompt across multi-turn chats; per-request `chat_template_kwargs` still override per call.                                                                                                                                                                                                                                                        |
| `CHAT_TEMPLATE_PATH`     | `/etc/vllm/chat_template.jinja`            | Overrides the template bundled with the model. The image ships [froggeric's v9](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) which fixes the `System message must be at the beginning` assertion + several other bugs (`developer` role, `</thinking>` hallucinations, etc.). Set empty to fall back to the model's bundled template. See `chat-templates/README.md`.                                                                        |
| `KV_CACHE_DTYPE`         | `fp8`                                      | KV cache quantization. Default is `fp8` (~2x KV pool vs FP16, lossless in practice). TurboQuant on Qwen3.6 needs vLLM PR #39931 (merged 2026-05-05, post-v0.20.1); the merge-commit wheel was tried and OOMs at cold start on 1×4090 24 GB, so the image stays on stable `vllm==0.20.1` + fp8. Re-enable once a tagged release ships #39931 and is verified.                                                                                                |
| `HF_TOKEN`               | _(empty)_                                  | Only needed if you swap to a gated model repo.                                                                                                                                                                                                                                                                                                                                                                                                              |

### Step 3 — Test

```bash
EID=<your-endpoint-id>
KEY=$RUNPOD_API_KEY

# Health (no auth needed in some configs; LB injects bearer)
curl https://$EID.api.runpod.ai/ping -H "Authorization: Bearer $KEY"

# OpenAI-compatible chat
curl https://$EID.api.runpod.ai/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}]}'

# Streaming (SSE)
curl -N https://$EID.api.runpod.ai/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","stream":true,"messages":[{"role":"user","content":"count to 5"}]}'
```

### Cold-start expectations

On `RTX 4090` with weights on a Network Volume in the same DC: roughly **90–150 s** from worker start → first token. CUDA-graph compile + MTP head + spec-decode warmup dominate; the actual weight load is ~5–10 s with `hf_transfer`. Plan accordingly:

- Set `Workers Min = 1` for production traffic to avoid cold starts entirely.
- Bump `Idle Timeout` so warm workers survive request gaps.
- Caddy returns `204` from `/ping` during this window — the LB will queue requests instead of failing them.

### RunPod-specific tuning notes

- **24 GB 4090 is tight** with MTP spec-decode + 32K context + fp8 KV. If you OOM at boot, lower `MAX_MODEL_LEN` to `24576` or `16384`, drop `num_speculative_tokens` to `2` (edit `docker-entrypoint.sh`), or lower `GPU_MEMORY_UTIL` to `0.93`. TurboQuant would unlock ~64K context but currently OOMs (see KV cache caveat).
- **Vision (MoonViT) + long context simultaneously** can OOM on 24 GB. Reduce `MAX_MODEL_LEN` if vision is the primary use case.
- **Network Volume is region-pinned.** The endpoint workers must run in the same datacenter or the volume can't mount.
- **Image rebuilds on every push to `main`.** RunPod re-pulls and rebuilds from source each time — full vLLM/CUDA build is ~10–20 min. Push deliberately.

## Caveats

- **Mamba prefix caching is experimental** for Qwen3.6. vLLM auto-picks the `align` fallback mode for `Qwen3_5ForConditionalGeneration`. Regular-attention layers cache fine (~85% hit rate); Mamba/GDN linear-attention layers re-run prefill on every new request.
- **Spec-decode silently ignores** `min_p` and `logit_bias` per-request params.
- **Deprecation warnings about `Qwen2VLImageProcessorFast` / `use_fast`** are upstream-transformers noise; ignore.
- **CUDA graph mode downgrades** to `PIECEWISE` under spec-decode (FlashInfer limitation) — automatic and expected.
- **Readiness probe is `/v1/models`, not `/health`** — vLLM's `/health` endpoint returns `200` as soon as the engine subprocess is alive, which is **before** weights finish loading, before `profile_run` reserves activation buffers, and before CUDA-graph capture finishes. The Caddy `/ping` shim probes `/v1/models` instead because that route is only registered after `init_app_state` completes (i.e. the model is actually ready to serve). 4xx (FastAPI up but model registry empty), 5xx (engine errors), and connection-refused (Caddy starts before vLLM binds `:1234`) all map to `204` so the LB queues instead of routing traffic to a worker that would 500 the first request.
- **Chat template fixes** — the image ships a vendored copy of [froggeric's v9 chat template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) at `/etc/vllm/chat_template.jinja` (passed as `--chat-template`). It fixes the stock Qwen3.6 template's hard `System message must be at the beginning.` assertion (clients like LangChain/Codex CLI routinely send multiple system messages), accepts the `developer` role, handles `</thinking>` hallucinations, and avoids a no-user-query crash. Set `CHAT_TEMPLATE_PATH=` (empty) to fall back to the model-bundled template. See [`chat-templates/README.md`](chat-templates/README.md).
- **TurboQuant KV cache** — currently disabled. The image was briefly bumped to a per-commit vllm wheel (`0.20.2rc1.dev25+g4f2af1a7c`, post PR #39931 which fixes TurboQuant on hybrid attention+Mamba/GDN models like Qwen3.6) with `KV_CACHE_DTYPE=turboquant_k8v4` baked in, but that wheel OOMs at cold start on 1×4090 24 GB even with conservative settings. The image is back on stable `vllm==0.20.1` + fp8 KV; the `KV_CACHE_DTYPE` env knob and `--kv-cache-dtype` plumbing are kept so a future stable release shipping #39931 can re-enable TurboQuant by setting `KV_CACHE_DTYPE=turboquant_k8v4`. With fp8, the single-GPU `MAX_MODEL_LEN` clamp lands at 32K instead of 64K.

## Acknowledgments

- [tedivm/qwen36-27b-docker](https://github.com/tedivm/qwen36-27b-docker) — the Docker packaging this RunPod edition forks from.
- [k0zakinio/qwen36-vllm-setup](https://github.com/k0zakinio/qwen36-vllm-setup) — the original repo this was forked from; all of the vLLM flag tuning, performance benchmarks, and serve scripts originated there.
- [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) — vendored v9 chat template that fixes multi-system messages and several other Qwen3.6-template bugs.
- [Lorbus](https://huggingface.co/Lorbus) — AutoRound INT4 quant that preserves the MTP head in BF16 and keeps MoonViT in FP16.
- [Qwen team](https://github.com/QwenLM/Qwen3.6) — the base model and the MTP head.
- Medium article ["An Overnight Stack for Qwen3.6-27B"](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914?postPublishedType=repub) — original source of the AutoRound + MTP + TurboQuant stack.
- [Sandermage's Genesis patches](https://github.com/Sandermage/genesis-vllm-patches) — more aggressive approach with TurboQuant KV; useful reference for pushing further.
