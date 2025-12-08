# Troubleshooting

Common issues and how to resolve them.

## Model & Cache Issues

### "Offline mode enabled but no cached model found"

**Cause:** You're running with `--offline` or `SGREP_OFFLINE=1` but the embedding model hasn't been downloaded yet.

**Solution:**
```bash
# Check where the model should be
sgrep config --show-model-dir

# Download the model (requires internet)
sgrep index   # Run once without --offline

# Verify the model exists
sgrep config --verify-model
```

**For air-gapped environments:** Download the model on a connected machine, then copy the cache directory to the target machine. See [Offline Mode](offline.md) for details.

---

### "Model initialization timed out"

**Cause:** The model is taking too long to load, possibly due to slow disk or large model size.

**Solution:**
```bash
# Increase the timeout (default: 120 seconds)
export SGREP_INIT_TIMEOUT_SECS=300
sgrep index
```

---

### "No embedding generated"

**Cause:** The embedder returned an empty result, typically due to:
- Corrupted model files
- Memory exhaustion
- Incompatible ONNX runtime

**Solution:**
```bash
# Re-download the model
rm -rf ~/.sgrep/cache/fastembed
sgrep index

# Or with explicit cache path
rm -rf $FASTEMBED_CACHE_DIR
sgrep index
```

---

### "Model initialization failed" or "Model initialization thread crashed"

**Cause:** The ONNX runtime failed to load the model, often due to:
- Incompatible system libraries
- GPU driver issues (when using CUDA/CoreML)
- Corrupted download

**Solution:**
```bash
# Force CPU mode
sgrep index --device cpu

# Clear cache and retry
rm -rf ~/.sgrep/cache/fastembed
sgrep index --device cpu
```

---

## Index Issues

### "No index found. Run 'sgrep index' first."

**Cause:** You're trying to search a directory that hasn't been indexed.

**Solution:**
```bash
# Index the current directory
sgrep index

# Or index a specific path
sgrep index /path/to/your/repo
```

---

### Index seems corrupted or outdated

**Cause:** The index may have been interrupted mid-write or is from an older version.

**Solution:**
```bash
# Force a full rebuild
sgrep index --force

# Check index stats after rebuild
sgrep index --stats
```

---

### Index is huge / taking too long

**Cause:** Large codebases can create significant indexes.

**Solutions:**
```bash
# Use GPU offload for faster indexing
sgrep index --offload

# Check what's being indexed (stats without rebuild)
sgrep index --stats

# Profile the indexing phases
sgrep index --profile
```

---

## Remote Storage Issues

### "Turbopuffer API key missing"

**Cause:** Turbopuffer is configured but no API key was provided.

**Solution:**
```bash
# Set via environment
export TURBOPUFFER_API_KEY="tpuf_your_key"

# Or in config file (~/.sgrep/config.toml)
[turbopuffer]
api_key = "tpuf_your_key"
```

---

### "Pinecone API key missing" / "Pinecone endpoint missing"

**Cause:** Pinecone is configured but missing required settings.

**Solution:**
```toml
# In ~/.sgrep/config.toml
[pinecone]
api_key = "your-key"
endpoint = "https://YOUR-INDEX.svc.REGION.pinecone.io"
```

---

### "Authentication failed: invalid Turbopuffer API key"

**Cause:** The API key is invalid or expired.

**Solution:**
1. Verify your API key at [Turbopuffer dashboard](https://turbopuffer.com)
2. Check for typos in the key
3. Ensure the key has write permissions

---

### "Rate limited by Turbopuffer"

**Cause:** Too many requests in a short period.

**Solution:** Wait a few minutes and retry. For large indexes, consider:
```bash
# Reduce batch size
sgrep index --remote --batch-size 50
```

---

### "Pinecone upsert/query/delete failed"

**Cause:** Network issues, invalid index, or permission problems.

**Solutions:**
1. Verify the endpoint URL is correct
2. Check that the index exists in Pinecone console
3. Ensure your API key has the necessary permissions
4. Check network connectivity

---

## Modal.dev Offload Issues

### Modal deployment fails

**Cause:** Invalid credentials or account issues.

**Solution:**
```bash
# Verify Modal authentication
modal token new  # Re-authenticate via browser

# Or set tokens directly
export MODAL_TOKEN_ID="ak-..."
export MODAL_TOKEN_SECRET="as-..."

# Test the connection
sgrep index --offload
```

---

### "Modal service not responding"

**Cause:** The Modal endpoint may have cold-started or the service needs redeployment.

**Solution:**
```bash
# The first request after idle will be slow (cold start)
# Wait for the request to complete

# Force redeployment
modal deploy  # From the sgrep repo
```

---

## Search Issues

### Search returns no results

**Troubleshooting steps:**

1. **Check if index exists:**
   ```bash
   ls ~/.sgrep/indexes/
   sgrep index --stats
   ```

2. **Verify index is fresh:**
   ```bash
   sgrep index  # Re-index if needed
   ```

3. **Try a broader query:**
   ```bash
   # Instead of specific function names
   sgrep search "error handling"

   # Instead of exact phrases
   sgrep search "authentication login user"
   ```

4. **Check file filters aren't too restrictive:**
   ```bash
   # Remove glob filters
   sgrep search "query" --glob "**/*"
   ```

---

### Search results are low quality

**Solutions:**

1. **Use GPU-accelerated embeddings for better quality:**
   ```bash
   sgrep search --offload "query"
   ```

2. **Increase result count:**
   ```bash
   sgrep search --limit 20 "query"
   ```

---

### Search is slow

**Solutions:**

1. **First search after indexing is slow** (building BM25 cache):
   ```bash
   # Subsequent searches will be faster
   ```

2. **Use GPU offload for faster embedding:**
   ```bash
   sgrep search --offload "query"
   ```

---

## Watch Mode Issues

### Watch process not updating

**Cause:** Filesystem events may not be triggering properly.

**Solution:**
```bash
# Kill existing watch processes
pkill -f "sgrep watch"

# Restart watch
sgrep watch
```

---

### Multiple watch processes running

**Cause:** Watch was started multiple times.

**Solution:**
```bash
# Find sgrep processes
ps aux | grep sgrep

# Kill all watch processes
pkill -f "sgrep watch"

# Restart single instance
sgrep watch --detach
```

---

## HNSW Index Errors

### "HNSW creation failed" / "HNSW reserve failed"

**Cause:** Memory exhaustion or vector dimension mismatch.

**Solutions:**

1. **Reduce memory pressure:**
   ```bash
   # Lower thread count
   sgrep index --threads 2

   # Use background preset
   sgrep index --cpu-preset background
   ```

2. **Force index rebuild:**
   ```bash
   sgrep index --force
   ```

---

## Getting Help

### Enable debug logging

```bash
# Show detailed logs
RUST_LOG=sgrep=debug sgrep search "query"

# Show all logs including dependencies
RUST_LOG=debug sgrep search "query"
```

### Check version

```bash
sgrep --version
```

### View index statistics

```bash
sgrep index --stats
sgrep index --stats --json  # Machine-readable
```

### Report issues

Open an issue at [github.com/rika-labs/sgrep/issues](https://github.com/rika-labs/sgrep/issues) with:
- sgrep version (`sgrep --version`)
- OS and architecture
- Error message (with `RUST_LOG=debug` output)
- Steps to reproduce
