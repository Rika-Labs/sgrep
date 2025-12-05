# Offline and airgapped installs

sgrep runs locally and ships with a local embedding pipeline. The first run downloads `jina-embeddings-v2-base-code` from Hugging Face unless the files are already cached.

## Fail fast offline

```bash
sgrep --offline search "init query"
# or
SGREP_OFFLINE=1 sgrep index
```

Offline mode sets `HF_HUB_OFFLINE=1` and exits with an error if the model cache is missing.

## Find the model directory

```bash
sgrep config --show-model-dir
```

By default the cache lives in `FASTEMBED_CACHE_DIR` or the OS cache dir (e.g., `~/.local/share/sgrep/cache/fastembed/jina-embeddings-v2-base-code`), falling back to `~/.sgrep/cache/fastembed/jina-embeddings-v2-base-code`.

## Manual download

Download these files into the model directory:

- `model_quantized.onnx`
- `tokenizer.json`
- `config.json`
- `special_tokens_map.json`
- `tokenizer_config.json`

Source URL: https://huggingface.co/jinaai/jina-embeddings-v2-base-code/tree/main

Verify afterward:

```bash
sgrep config --verify-model
```

## Proxies and mirrors

- Respect `HTTP_PROXY` / `HTTPS_PROXY` for download access.
- Set `FASTEMBED_CACHE_DIR` if you want the cache in a custom location.
- In airgapped environments, copy a prefilled cache directory to the same path and run with `--offline`.
