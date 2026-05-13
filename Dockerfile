ARG CUDA_VERSION=13.2.1-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        curl \
        git \
        build-essential \
        && ln -sf /usr/bin/python3 /usr/bin/python \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed pip wheel

# TRIAL BRANCH: vllm 0.21.0rc2.
# 0.21.0rc2 is the first tagged release that contains PR #39931 (merge commit
# 4f2af1a7c), which fixes TurboQuant startup on hybrid (attention + Mamba/GDN)
# models like Qwen3.6. It also bundles ~250 other commits over v0.20.1
# including #35520 (ModelRunner V2 hybrid support), #42070 (GDN nested
# torch.compile in cudagraph capture), #41617 (causal_conv1d IMA on long
# sequences), and #40961 (preserve max_seq_len in ubatch metadata during
# cudagraph capture) — all on the Mamba/GDN path Qwen3.6 uses.
#
# OPEN RISK: the surgical post-#39931 wheel (0.20.2rc1.dev25+g4f2af1a7c) cold-
# start OOM'd on 1x4090 24 GB even with fp8 KV and a 32K clamp. Whether the
# rc2 bundle materially changes peak profile_run memory is unknown without a
# real-hardware run. Roll back via KV_CACHE_DTYPE=fp8 if k8v4 OOMs.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        'vllm==0.21.0rc2' \
        auto-round \
        hf_transfer \
        huggingface_hub

RUN mkdir -p /data/models /data/logs

# --- Caddy: /ping shim + SSE-friendly reverse proxy for RunPod LB ---------
ARG CADDY_VERSION=2.8.4
RUN curl -fsSL "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VERSION}/caddy_${CADDY_VERSION}_linux_amd64.tar.gz" \
        | tar -xz -C /usr/local/bin caddy \
    && chmod +x /usr/local/bin/caddy \
    && mkdir -p /etc/caddy

COPY Caddyfile /etc/caddy/Caddyfile

# --- Vendored chat template (fixes multi-system, developer role, etc.) ----
RUN mkdir -p /etc/vllm
COPY chat-templates/chat_template-v9.jinja /etc/vllm/chat_template.jinja

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY runpod-entrypoint.sh /usr/local/bin/runpod-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh /usr/local/bin/runpod-entrypoint.sh

COPY scripts/ /usr/local/bin/scripts/
ENV PATH="/usr/local/bin/scripts:${PATH}"

ENV MODEL_DIR=/data/models
ENV LOG_DIR=/data/logs
ENV MODEL_REPO=Lorbus/Qwen3.6-27B-int4-AutoRound
ENV PORT=1234
ENV SERVED_MODEL_NAME=qwen3.6-27b
ENV MAX_MODEL_LEN=200000
ENV MAX_NUM_SEQS=3
ENV MAX_NUM_BATCHED_TOKENS=4128
ENV GPU_MEMORY_UTIL=0.92
ENV TEMPERATURE=0.6
ENV TOP_P=0.95
ENV TOP_K=20
ENV MIN_P=0.0
ENV PRESENCE_PENALTY=0
ENV REPETITION_PENALTY=1.0
ENV REASONING_PARSER=qwen3
ENV MODEL_DOWNLOAD=0
ENV PUBLIC_PORT=8000
ENV CHAT_TEMPLATE_KWARGS={\"preserve_thinking\":true}
ENV CHAT_TEMPLATE_PATH=/etc/vllm/chat_template.jinja

# TRIAL BRANCH: re-enable TurboQuant k8v4 KV cache (8-bit K, 4-bit V packed).
# k8v4 stores ~65 B/token vs fp8 ~85 B/token, so the single-GPU MAX_MODEL_LEN
# clamp in docker-entrypoint.sh lifts from 32768 -> 65536 when this is active.
# Override with `-e KV_CACHE_DTYPE=fp8` to fall back to stable behavior.
ENV KV_CACHE_DTYPE=turboquant_k8v4

# Caddy listens on PUBLIC_PORT (RunPod LB attaches here).
# vLLM listens on PORT (internal, 127.0.0.1 only — proxied by caddy).
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/runpod-entrypoint.sh"]
CMD ["serve"]
