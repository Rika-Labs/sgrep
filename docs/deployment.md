# Deployment

`sgrep` is a local CLI. The recommended deployment model is simple: install the binary on the machine where the repository lives and keep the index in your local `SGREP_HOME`.

## Typical setup

```bash
sgrep config --init
sgrep index
sgrep search "database connection"
```

## Offline setup

- pre-download the embedding model once
- point `SGREP_HOME` at a persistent directory
- run `sgrep --offline index` and `sgrep --offline search`

## Watch mode

```bash
sgrep watch
```

This keeps the local index fresh for repeated queries and agent sessions.

## Suggested environments

- developer laptops
- local workstations
- private build boxes
- air-gapped or restricted environments with a warmed model cache
