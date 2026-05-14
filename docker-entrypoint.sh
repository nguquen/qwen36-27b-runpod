#!/usr/bin/env bash
set -euo pipefail

# --- Auto-detect GPU count & set TP if not overridden --------------------
if [[ -z "${TENSOR_PARALLEL:-}" ]]; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if (( GPU_COUNT == 0 )); then
        echo "ERROR: No GPUs detected. Pass --gpus to docker run." >&2
        exit 1
    fi
    TENSOR_PARALLEL=$GPU_COUNT
    echo "Detected ${GPU_COUNT} GPU(s), setting TENSOR_PARALLEL=${TENSOR_PARALLEL}"
fi
export TENSOR_PARALLEL

# --- Adjust defaults based on GPU count -----------------------------------
# The Dockerfile bakes the multi-GPU defaults (MAX_MODEL_LEN=200000,
# GPU_MEMORY_UTIL=0.92) into the image as ENV, which means a plain
# `:=` lazy default never fires (the var is set, just to the wrong
# value for a single 24 GB card — vLLM refuses to start with "KV cache
# is needed ... larger than available"). So when we detect TP=1 AND
# the values still match the baked image defaults, actively clamp them
# to safe single-GPU values. Operator overrides via `-e` / RunPod env
# vars (any other value) are left untouched.
#
# Single-GPU context ceiling depends on KV cache dtype. TurboQuant variants
# (PR #39931, in vllm 0.21.0rc2+) compress KV beyond fp8's 2x; caps below
# are derived from the actual per-slot byte size in vllm's TurboQuant
# config, anchored to the real-hardware k8v4 baseline.
#
# Calibration anchor: k8v4 at 57344 was validated on 1x4090 24 GB with vllm
# 0.21.0rc2 + MTP spec-decode (3 draft tokens) + MAX_NUM_SEQS=3 +
# GPU_MEMORY_UTIL=0.95. vllm's KV planner reported 59136 as the absolute
# ceiling; 57344 (7*8192) leaves ~3% margin for block-alloc variance.
#
# Other variants: math ceiling = 59136 * (k8v4_slot / variant_slot), then
# rounded DOWN to a multiple of 8192 with ~15% safety margin (vs k8v4's
# 3% — these are unvalidated on real hardware yet).
#
# Slot size from vllm config.py (TurboQuantConfig.key_packed_size +
# value_packed_size) for Qwen3.6 head_dim=256:
#   key   = ceil(head_dim * key_bits / 8) + 2   (FP8 keys: head_dim bytes)
#   value = ceil(head_dim * value_bits / 8) + 4 (scale + zero, fp16 each)
#
#     dtype                bits  slot  ratio  math_max  baked   PPL
#     ------------------   ----  ----  -----  --------  ------  -------
#     fp8                  --    --    --     --        32768   0%
#     turboquant_k8v4      8/4   388   1.00x   59136*   57344   +1.17%
#     turboquant_4bit_nc   4/4   262   1.48x   87602    73728   +2.71%
#     turboquant_k3v4_nc   3/4   230   1.69x   99762    81920   +10.63%
#     turboquant_3bit_nc   3/3   198   1.96x  115892    98304   +20.59%
#     (* k8v4 row is the real-hw planner ceiling, not extrapolated)
#
# NOTE: vllm 0.21.0rc2 docstring claims k3v4_nc compresses at ~3.5x, which
# would put it below 4bit_nc (3.8x) and violate the bit ordering. The slot
# math above gives 4.45x; the docstring number appears to be a typo (the
# `~` prefix already flags it as approximate). We trust the slot math.
#
# Operator can override per-endpoint via MAX_MODEL_LEN env var.
case "${KV_CACHE_DTYPE:-fp8}" in
    turboquant_k8v4)
        SINGLE_GPU_CTX_CAP=57344
        ;;
    turboquant_4bit_nc)
        SINGLE_GPU_CTX_CAP=73728
        ;;
    turboquant_k3v4_nc)
        SINGLE_GPU_CTX_CAP=81920
        ;;
    turboquant_3bit_nc)
        SINGLE_GPU_CTX_CAP=98304
        ;;
    *)
        SINGLE_GPU_CTX_CAP=32768
        ;;
esac
if (( TENSOR_PARALLEL == 1 )); then
    if [[ "${MAX_MODEL_LEN:-}" == "200000" ]]; then
        echo "Single-GPU mode (KV=${KV_CACHE_DTYPE:-fp8}): clamping MAX_MODEL_LEN 200000 -> ${SINGLE_GPU_CTX_CAP}"
        MAX_MODEL_LEN=$SINGLE_GPU_CTX_CAP
    fi
    if [[ "${GPU_MEMORY_UTIL:-}" == "0.92" ]]; then
        echo "Single-GPU mode: raising GPU_MEMORY_UTIL 0.92 -> 0.95"
        GPU_MEMORY_UTIL=0.95
    fi
    : "${MAX_MODEL_LEN:=48000}"
    : "${GPU_MEMORY_UTIL:=0.95}"
    echo "Single-GPU mode: MAX_MODEL_LEN=${MAX_MODEL_LEN}, GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL}"
fi
export MAX_MODEL_LEN
export GPU_MEMORY_UTIL

# --- Auto-scale MAX_NUM_BATCHED_TOKENS by GPU count ----------------------
# vLLM reserves activation buffers sized to --max-num-batched-tokens during
# the cold-start profile_run step. On 1x4090 24 GB the safe ceiling is 4128
# (matches vLLM's CUDA-graph capture endpoint 4096+32). With TP>=2 the model
# weights shard across GPUs, freeing per-GPU memory, so we can scale up:
#   TP=2  -> 8192   (~2x faster prefill on long prompts)
#   TP>=4 -> 16384  (best for long-prompt agents)
# Only auto-bump when the value still matches the baked sentinel "4128"
# (or is unset); operator overrides via -e / RunPod env vars are untouched.
if (( TENSOR_PARALLEL >= 2 )) && [[ "${MAX_NUM_BATCHED_TOKENS:-4128}" == "4128" ]]; then
    if (( TENSOR_PARALLEL == 2 )); then
        MAX_NUM_BATCHED_TOKENS=8192
    else
        MAX_NUM_BATCHED_TOKENS=16384
    fi
    echo "Multi-GPU mode (TP=${TENSOR_PARALLEL}): raising MAX_NUM_BATCHED_TOKENS 4128 -> ${MAX_NUM_BATCHED_TOKENS}"
fi
: "${MAX_NUM_BATCHED_TOKENS:=4128}"
export MAX_NUM_BATCHED_TOKENS

# --- Determine visible GPU indices ----------------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=""
    for i in $(seq 0 $((TENSOR_PARALLEL - 1))); do
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:+${CUDA_VISIBLE_DEVICES},}${i}"
    done
fi
export CUDA_VISIBLE_DEVICES

# --- Cachedir setup -------------------------------------------------------
export HF_HOME="${MODEL_DIR}/.hf_cache"
export TMPDIR="${MODEL_DIR}/.tmp"
export PIP_CACHE_DIR="${MODEL_DIR}/.pip_cache"
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR"

# --- Logs setup -----------------------------------------------------------
mkdir -p "$LOG_DIR"

# --- Commands --------------------------------------------------------------
CMD="${1:-serve}"

case "$CMD" in
    download)
        if [[ -f "${MODEL_DIR}/config.json" ]]; then
            echo "Model already present at ${MODEL_DIR} — skipping download."
            exit 0
        fi
        echo "Downloading ${MODEL_REPO} to ${MODEL_DIR} ..."
        mkdir -p "$MODEL_DIR"
        HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_REPO" \
            --local-dir "$MODEL_DIR"
        echo "Download complete."
        exit 0
        ;;

    serve)
        if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
            if [[ "${MODEL_DOWNLOAD:-0}" == "1" ]]; then
                echo "Model not found and MODEL_DOWNLOAD=1 — downloading now ..."
                HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_REPO" \
                    --local-dir "$MODEL_DIR"
                echo "Download complete."
            else
                echo "ERROR: Model not found at ${MODEL_DIR}." >&2
                echo "Run 'docker compose run --rm qwen36 download' or set MODEL_DOWNLOAD=1." >&2
                exit 1
            fi
        fi

        # --- Build generation config JSON from env vars ---------------------
        GEN_CONFIG="{\"temperature\": ${TEMPERATURE}, \"top_p\": ${TOP_P}, \"top_k\": ${TOP_K}, \"min_p\": ${MIN_P}, \"presence_penalty\": ${PRESENCE_PENALTY}, \"repetition_penalty\": ${REPETITION_PENALTY}}"

        # --- Optional reasoning parser --------------------------------------
        REASONING_PARSER_FLAG=""
        [[ -n "${REASONING_PARSER:-}" ]] && REASONING_PARSER_FLAG="--reasoning-parser ${REASONING_PARSER}"

        # --- Optional default chat-template kwargs (e.g. preserve_thinking) -
        CHAT_TEMPLATE_KWARGS_FLAG=()
        if [[ -n "${CHAT_TEMPLATE_KWARGS:-}" ]]; then
            CHAT_TEMPLATE_KWARGS_FLAG=(--default-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")
        fi

        # --- Optional override chat template (fixes multi-system, etc.) -----
        # Set CHAT_TEMPLATE_PATH=/path/to/template.jinja to override the
        # template bundled with the model. Empty string falls back to the
        # model's bundled template. The image bakes a default pointing at
        # the vendored froggeric v9 template under /etc/vllm.
        CHAT_TEMPLATE_FLAG=()
        if [[ -n "${CHAT_TEMPLATE_PATH:-}" && -f "${CHAT_TEMPLATE_PATH}" ]]; then
            CHAT_TEMPLATE_FLAG=(--chat-template "${CHAT_TEMPLATE_PATH}")
            echo "Using chat template: ${CHAT_TEMPLATE_PATH}"
        elif [[ -n "${CHAT_TEMPLATE_PATH:-}" ]]; then
            echo "WARNING: CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH} not found; using model's bundled template" >&2
        fi

        # --- Optional OTel endpoints ----------------------------------------
        OTEL_TRACES_FLAG=""
        [[ -n "${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-}" ]] && OTEL_TRACES_FLAG="--otlp-traces-endpoint ${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}"
        OTEL_METRICS_FLAG=""
        [[ -n "${OTEL_EXPORTER_OTLP_METRICS_ENDPOINT:-}" ]] && OTEL_METRICS_FLAG="--otlp-metrics-endpoint ${OTEL_EXPORTER_OTLP_METRICS_ENDPOINT}"
        OTEL_LOGS_FLAG=""
        [[ -n "${OTEL_EXPORTER_OTLP_LOGS_ENDPOINT:-}" ]] && OTEL_LOGS_FLAG="--otlp-logs-endpoint ${OTEL_EXPORTER_OTLP_LOGS_ENDPOINT}"

        echo "Starting vLLM server:"
        echo "  Model              : ${MODEL_DIR}"
        echo "  Served as          : ${SERVED_MODEL_NAME}"
        echo "  Port               : ${PORT}"
        echo "  Tensor parallelism : ${TENSOR_PARALLEL}"
        echo "  Max model len      : ${MAX_MODEL_LEN}"
        echo "  Max num seqs       : ${MAX_NUM_SEQS}"
        echo "  Max batched tokens : ${MAX_NUM_BATCHED_TOKENS}"
        echo "  GPU memory util    : ${GPU_MEMORY_UTIL}"
        echo "  CUDA devices       : ${CUDA_VISIBLE_DEVICES}"
        echo "  Generation config  : ${GEN_CONFIG}"
        echo ""

        vllm serve "$MODEL_DIR" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --override-generation-config "$GEN_CONFIG" \
            --port "$PORT" \
            --dtype float16 \
            --quantization auto_round \
            --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --tensor-parallel-size "$TENSOR_PARALLEL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --disable-custom-all-reduce \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_coder \
            --trust-remote-code \
            ${REASONING_PARSER_FLAG} \
            "${CHAT_TEMPLATE_KWARGS_FLAG[@]}" \
            "${CHAT_TEMPLATE_FLAG[@]}" \
            ${OTEL_TRACES_FLAG} \
            ${OTEL_METRICS_FLAG} \
            ${OTEL_LOGS_FLAG} \
            --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
            2>&1 | tee -a "${LOG_DIR}/vllm.log"
        ;;

    *)
        echo "Unknown command: ${CMD}" >&2
        echo "Usage: docker-entrypoint.sh {download|serve}" >&2
        exit 1
        ;;
esac
