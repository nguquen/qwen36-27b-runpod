# Vendored chat templates

This directory holds chat templates we vendor for the image, with full
attribution and a documented re-sync procedure.

## `chat_template-v9.jinja`

A drop-in replacement for the chat template bundled with
`Lorbus/Qwen3.6-27B-int4-AutoRound`, sourced from
[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates).

| Field           | Value                                                                                               |
| --------------- | --------------------------------------------------------------------------------------------------- |
| Source URL      | https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/blob/main/qwen3.6/chat_template-v9.jinja |
| Pinned commit   | `1dc4a4a74f8a39adc1cc2c64f015fb3385c31f3c`                                                          |
| Upstream sha256 | `edd26617fb7816af8a47fc25da7a3da57ef6f8075cc90c60e07591fe0c0b0c1c` (body only, before our header)   |
| License         | Apache-2.0                                                                                          |
| Author          | froggeric (template fixes); Alibaba Cloud Qwen team (original Qwen template)                        |

The local file is byte-identical to the upstream blob with **only** a
leading Jinja `{#- ... -#}` comment block prepended for attribution.
Comments in Jinja render to nothing, so the rendered prompt is exactly
what froggeric's template produces.

### What v9 fixes vs the stock Qwen3.6 template

- **Multi-system messages no longer raise.** The stock template hard-asserts
  `System message must be at the beginning.` whenever a `system` role appears
  past index 0; v9 hoists the leading system/developer message to the top
  and emits any later system messages inline at their natural index.
- **`developer` role accepted** and treated as `system` (mirrors the
  OpenAI Responses API and Anthropic conventions).
- **`</thinking>` hallucinations** (the model occasionally closes with
  `</thinking>` instead of `</think>`) are detected and handled.
- **No-user-query crash** replaced with a graceful fallback
  (`last_query_index = messages|length - 1`).
- **Empty `<think>` blocks** are dropped via length check rather than
  rendered as a stray prefix.
- **Tool argument rendering** uses direct dict lookup (not the `|items`
  filter) for compatibility with non-Python Jinja engines (e.g. minja in
  `llama.cpp`).
- **`|safe` removed** — also a portability fix.
- **Auto-close of unclosed `<think>` before `<tool_call>`** uses a
  split-based pattern instead of `rfind`/slicing.
- **`<|think_on|>` / `<|think_off|>` toggle** can appear anywhere in any
  message, not just the leading system block.

### Re-syncing to a newer upstream

```bash
# Pick the new commit (or branch tip) from the HF repo, then:
COMMIT=<new-commit-sha>
URL="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/${COMMIT}/qwen3.6/chat_template-v9.jinja"

# Fetch + verify
curl -fsSL "$URL" -o /tmp/froggeric_v9.jinja
sha256sum /tmp/froggeric_v9.jinja   # record this in the header + this README

# Replace body, keep our leading {#- ... -#} attribution block.
# Update the header (URL, commit, sha256) and the table above to match.
```

Body changes from upstream that we apply locally: **none.** If froggeric
ships a `v10`, save it as `chat-templates/chat_template-v10.jinja`,
update `Dockerfile` and this README, and bump the `--chat-template`
default in `docker-entrypoint.sh`.

### Disabling the override

Set `CHAT_TEMPLATE_PATH=` (empty) on the endpoint to fall back to the
template bundled inside the model directory. See `.env.example` and the
root `README.md` for runtime config.
