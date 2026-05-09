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

# vLLM pinned to post-#39931 commit wheel (4f2af1a, 2026-05-05) to get
# TurboQuant hybrid-model support (Qwen3.6 = attention + Mamba/GDN).
# v0.20.1 was tagged the day BEFORE #39931 merged, so any tagged release
# < 0.21 lacks the hybrid path. Switch back to a stable pin once v0.20.2/0.21
# ships with the merge included.
ARG VLLM_COMMIT=4f2af1a7c03aae2b3227dd7e69d726104d44a711
ARG VLLM_VERSION=0.20.2rc1.dev25+g4f2af1a7c
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}/" \
        --pre "vllm==${VLLM_VERSION}" \
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
ENV KV_CACHE_DTYPE=turboquant_k8v4

# Caddy listens on PUBLIC_PORT (RunPod LB attaches here).
# vLLM listens on PORT (internal, 127.0.0.1 only — proxied by caddy).
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/runpod-entrypoint.sh"]
CMD ["serve"]
