---
name: sgrep
description: Use sgrep for semantic code search. Use when you need to find code by meaning rather than exact text matching. Perfect for finding concepts like "authentication logic", "error handling patterns", or "database connection pooling".
allowed-tools: Bash
---

# sgrep - Semantic Code Search

Use `sgrep` to search code semantically using natural language queries. sgrep understands code meaning, not just text patterns.

## When to Use

- Finding code by concept or functionality ("where do we handle authentication?")
- Discovering related code patterns ("show me retry logic")
- Exploring codebase structure ("how is the database connection managed?")
- Searching for implementation patterns ("where do we validate user input?")

## Prerequisites

Ensure `sgrep` is installed:
```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh
```

## Basic Usage

### Search Command

```bash
sgrep search "your natural language query"
```

### Common Patterns

**Find functionality:**
```bash
sgrep search "where do we handle user authentication?"
```

**Search with filters:**
```bash
sgrep search "error handling" --filters lang=rust
sgrep search "API endpoints" --glob "src/**/*.rs"
```

**Get more results:**
```bash
sgrep search "database queries" --limit 20
```

**Show full context:**
```bash
sgrep search "retry logic" --context
```

## Command Options

- `--limit <n>` or `-n <n>`: Maximum results (default: 10)
- `--context` or `-c`: Show full chunk content instead of snippet
- `--path <dir>` or `-p <dir>`: Repository path (default: current directory)
- `--glob <pattern>`: File pattern filter (repeatable)
- `--filters key=value`: Metadata filters like `lang=rust` (repeatable)
- `--threads <n>`: Maximum threads for parallel operations
- `--cpu-preset <preset>`: CPU usage preset (auto|low|medium|high|background)

## Indexing

If no index exists, sgrep will automatically create one on first search. To manually index:

```bash
sgrep index           # Index current directory
sgrep index --force   # Rebuild from scratch
```

## Watch Mode

The plugin automatically manages `sgrep watch` during Claude sessions. For manual watch:

```bash
sgrep watch           # Watch current directory
sgrep watch --debounce-ms 200
```

## Configuration

sgrep uses local embeddings by default:

```bash
sgrep config                    # Show current configuration
sgrep config --init             # Create default config file
sgrep config --show-model-dir   # Show model cache directory
sgrep config --verify-model     # Check if model files are present
```

If HuggingFace is blocked, set `HTTPS_PROXY` environment variable or see the [offline installation guide](https://github.com/rika-labs/sgrep#offline-installation).

## Examples

**Find authentication code:**
```bash
sgrep search "how do we authenticate users?"
```

**Find error handling:**
```bash
sgrep search "error handling patterns" --filters lang=rust
```

**Search specific file types:**
```bash
sgrep search "API rate limiting" --glob "src/**/*.rs"
```

**Get detailed results:**
```bash
sgrep search "database connection pooling" --context --limit 5
```

## Understanding Results

Results show:
- **File path and line numbers**: Where the code is located
- **Score**: Relevance score (higher is better)
- **Semantic score**: How well it matches the query meaning
- **Keyword score**: Text matching score
- **Code snippet**: Relevant code excerpt

## Best Practices

1. **Use natural language**: Ask questions like you would ask a colleague
2. **Be specific**: "authentication middleware" is better than "auth"
3. **Combine with filters**: Use `--filters lang=rust` to narrow by language
4. **Use globs**: `--glob "src/**/*.rs"` to search specific directories
5. **Check context**: Use `--context` when you need full function/class definitions

## Integration with Claude

When Claude needs to find code:
1. Use sgrep to search semantically
2. Review results to understand codebase structure
3. Use results to inform code changes or explanations
4. Combine multiple searches to build comprehensive understanding

