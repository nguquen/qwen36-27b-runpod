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
| `MAX_MODEL_LEN`                       | `200000`                            | Max context length (auto-lowered to 48000 for single GPU)               |
| `MAX_NUM_SEQS`                        | `3`                                 | Concurrent sequences                                                    |
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

| Flag                                                           | Why                                                   |
| -------------------------------------------------------------- | ----------------------------------------------------- |
| `--quantization auto_round`                                    | Matches the Lorbus weights                            |
| `--kv-cache-dtype fp8`                                         | Halves KV memory vs FP16; 200K x 3 seqs fits on 48 GB |
| `--enable-prefix-caching`                                      | Not default for Qwen3.6 hybrid attention; opt in      |
| `--enable-chunked-prefill`                                     | Recommended alongside spec-decode for throughput      |
| `--speculative-config method=mtp, num_speculative_tokens=3`    | ~2x throughput on code; 3 is the sweet spot           |
| `--max-num-seqs 3`                                             | Solo user + subagents; raise for more concurrency     |
| `--max-num-batched-tokens 4128`                                | Matches vLLM's CUDA-graph compile range endpoint      |
| `--gpu-memory-utilization 0.92`                                | Leaves CUDA-graph margin                              |
| `--disable-custom-all-reduce`                                  | No NVLink — stock NCCL is faster                      |
| `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice` | OpenAI-style tool calls                               |
| `--reasoning-parser qwen3`                                     | Enables extended thinking output                      |

TP=2 beats TP=1 by ~1.5x on dual 3090s. Memory-bandwidth savings from splitting weights across two cards outweigh the PCIe NCCL all-reduce cost.

## Deploy on RunPod Serverless (Load Balancer)

The image bundles a Caddy sidecar that exposes `:8000` with:

- `GET /ping` — returns **200** when vLLM is ready, **204** while it's still initializing (so the RunPod LB holds traffic instead of marking the worker failed).
- Everything else (`/v1/*`, `/metrics`, `/tokenize`, …) — reverse-proxied to vLLM on `127.0.0.1:1234` with response buffering disabled, so SSE token streams flush per chunk.

vLLM still binds to `1234` internally; nothing else changed for bare-metal users.

### Architecture

```
[RunPod LB] ──:8000──► [caddy] ──/ping──► 200 if upstream /health=200, else 204
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

| Variable          | Recommended                                        | Notes                                                                                                                                           |
| ----------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `MODEL_DOWNLOAD`  | `0`                                                | Volume is pre-populated. Set `1` to self-heal on a fresh volume (adds ~60–120s to cold start).                                                  |
| `MAX_NUM_SEQS`    | `3`                                                | Conservative; raise for higher throughput if memory allows.                                                                                     |
| `MAX_MODEL_LEN`   | `32768` (4090, auto) / `65536`–`96000` (A6000 48G) | `runpod-entrypoint.sh` auto-clamps to `32768` on single GPU when the upstream image default (`200000`) is detected. Set explicitly to override. |
| `GPU_MEMORY_UTIL` | `0.95` (auto on single GPU)                        | Auto-raised from upstream `0.92` → `0.95` on single GPU. Lower to `0.93` if you OOM at boot with vision or longer context.                      |
| `HF_TOKEN`        | _(empty)_                                          | Only needed if you swap to a gated model repo.                                                                                                  |

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

- **24 GB 4090 is tight** with MTP spec-decode + 48K context. If you OOM at boot, drop `num_speculative_tokens` to `2` (edit `docker-entrypoint.sh`) or lower `GPU_MEMORY_UTIL` to `0.93`.
- **Vision (MoonViT) + long context simultaneously** can OOM on 24 GB. Reduce `MAX_MODEL_LEN` if vision is the primary use case.
- **Network Volume is region-pinned.** The endpoint workers must run in the same datacenter or the volume can't mount.
- **Image rebuilds on every push to `main`.** RunPod re-pulls and rebuilds from source each time — full vLLM/CUDA build is ~10–20 min. Push deliberately.

## Caveats

- **Mamba prefix caching is experimental** for Qwen3.6. vLLM auto-picks the `align` fallback mode for `Qwen3_5ForConditionalGeneration`. Regular-attention layers cache fine (~85% hit rate); Mamba/GDN linear-attention layers re-run prefill on every new request.
- **Spec-decode silently ignores** `min_p` and `logit_bias` per-request params.
- **Deprecation warnings about `Qwen2VLImageProcessorFast` / `use_fast`** are upstream-transformers noise; ignore.
- **CUDA graph mode downgrades** to `PIECEWISE` under spec-decode (FlashInfer limitation) — automatic and expected.

## Acknowledgments

- [k0zakinio/qwen36-vllm-setup](https://github.com/k0zakinio/qwen36-vllm-setup) — the original repo this was forked from; all of the vLLM flag tuning, performance benchmarks, and serve scripts originated there.
- [Lorbus](https://huggingface.co/Lorbus) — AutoRound INT4 quant that preserves the MTP head in BF16 and keeps MoonViT in FP16.
- [Qwen team](https://github.com/QwenLM/Qwen3.6) — the base model and the MTP head.
- Medium article ["An Overnight Stack for Qwen3.6-27B"](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914?postPublishedType=repub) — original source of the AutoRound + MTP + TurboQuant stack.
- [Sandermage's Genesis patches](https://github.com/Sandermage/genesis-vllm-patches) — more aggressive approach with TurboQuant KV; useful reference for pushing further.
