# Troubleshooting

Common local-only issues and how to resolve them.

## Model missing in offline mode

```bash
sgrep config --show-model-dir
sgrep config --verify-model
```

Warm the cache once with network access if the model is missing.

## Model initialization fails

Try forcing CPU mode and rebuilding the cache:

```bash
sgrep index --device cpu
rm -rf ~/.sgrep/cache/fastembed
sgrep index --device cpu
```

## No index found

Build one first:

```bash
sgrep index
```

## Index seems stale or corrupted

Force a rebuild:

```bash
sgrep index --force
sgrep index --stats
```

## Indexing feels slow

- use `sgrep index --profile` to see phase timings
- keep the repo on fast local storage
- tune `SGREP_MAX_THREADS` or `SGREP_CPU_PRESET`
- use `--offline` once the model cache is already warm

## Search returns no results

- confirm the index exists with `sgrep index --stats`
- rebuild with `sgrep index --force`
- try broader natural-language queries
- remove overly strict globs or filters

## Search is slow

- the first search after indexing may be slower while caches warm up
- keep the index on fast local storage
- use a warmed local model cache

## Watch mode issues

If watch stops updating:

```bash
pkill -f "sgrep watch"
sgrep watch
```
