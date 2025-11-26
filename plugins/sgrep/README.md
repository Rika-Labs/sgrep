# sgrep Claude Code Plugin

A Claude Code plugin that integrates [sgrep](https://github.com/rika-labs/sgrep) for lightning-fast semantic code search. This plugin automatically manages `sgrep watch` during Claude Code sessions and provides a skill that enables Claude to use sgrep via CLI with structured `--json` output.

## Features

- **Automatic Index Management**: Automatically indexes your repository when starting a Claude session (rebuilds if missing/corrupted)
- **Background Watching**: Starts `sgrep watch` in the background to keep your index fresh
- **Skill Integration**: Provides a skill that enables Claude to use sgrep CLI commands for semantic code search
- **Session Lifecycle Management**: Automatically starts watch on session start and stops it on session end
- **Agent-Ready Output**: Uses `sgrep search --json` so Claude receives structured results (scores, lines, paths, snippets)

## Prerequisites

1. **sgrep must be installed**: Install sgrep before using this plugin:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh
   ```

2. **Claude Code**: Latest version with plugin support

## Installation

### From GitHub Marketplace

```bash
/plugin marketplace add rika-labs/sgrep
/plugin install sgrep
```

### From Local Directory

If you have the plugin locally:

```bash
/plugin marketplace add /path/to/sgrep/plugins/sgrep
/plugin install sgrep
```


## Usage

### Automatic Behavior

Once installed, the plugin automatically:

1. **On Session Start**: 
   - Checks if an index exists for your project
   - Creates or repairs the index if needed (`sgrep index` happens implicitly)
   - Starts `sgrep watch` in the background to keep the index updated

2. **During Session**:
- The `sgrep_search` MCP tool is available to Claude (invokes `sgrep search --json`)
   - You can ask Claude to search your codebase semantically

3. **On Session End**:
   - Gracefully stops the `sgrep watch` process
   - Cleans up process tracking files

### Using the Skill

Claude will automatically use the sgrep skill when you ask questions like:

- "Search for where we handle authentication"
- "Find all retry logic in the codebase"
- "Where do we connect to the database?"
- "Show me error handling patterns"

Claude will execute `sgrep search` commands with appropriate parameters based on your request.

### Skill Capabilities

The sgrep skill enables Claude to:

- Execute semantic searches using natural language queries
- Filter results by language, file patterns, or other metadata
- Control result limits and context display
- Understand when to use sgrep vs other search methods

## Configuration

The plugin uses the following directories:

- **Watch PIDs**: `~/.sgrep/watch-pids/` - Stores process IDs for cleanup
- **Indexes**: `~/.sgrep/indexes/` - sgrep index storage (managed by sgrep itself)

You can configure sgrep behavior:

**Config file** (`~/.sgrep/config.toml`):
```toml
[embedding]
provider = "local"
```

**Environment variables:**
- `SGREP_HOME`: Override default data directory
- `SGREP_CONFIG`: Override config file path
- `SGREP_DEVICE`: Set device (cpu|cuda|coreml)
- `SGREP_BATCH_SIZE`: Override embedding batch size
- `SGREP_MAX_THREADS`: Maximum threads for parallel operations
- `SGREP_CPU_PRESET`: CPU usage preset (auto|low|medium|high|background)
- `HTTP_PROXY` / `HTTPS_PROXY`: Proxy for model downloads

**Thread control presets:**
| Preset | CPU Usage | Use Case |
| --- | --- | --- |
| `auto` | 75% | Default balanced mode |
| `low` | 25% | Laptop/battery saving |
| `medium` | 50% | Multi-tasking |
| `high` | 100% | Maximum performance |
| `background` | 25% | Watch/daemon mode |

**Model management:**
```bash
sgrep config --show-model-dir   # Show model cache directory
sgrep config --verify-model     # Check if model files are present
```

## Troubleshooting

### Plugin Not Found

If Claude Code can't find the plugin:

1. Verify the plugin is installed: `/plugin list`
2. Check that `plugin.json` exists in `.claude-plugin/` directory
3. Ensure the plugin directory structure is correct

### Skill Not Working

If Claude doesn't use sgrep:

1. **sgrep not found**: Ensure sgrep is installed and in your PATH
2. **Index missing/corrupt**: The plugin auto-creates or repairs indexes on first search; you can manually run `sgrep index` if needed
3. **Skill not loaded**: Verify plugin is installed: `/plugin list`

### Watch Process Issues

If `sgrep watch` doesn't start or stop properly:

1. **Check PID files**: Look in `~/.sgrep/watch-pids/` for stale PID files
2. **Manual cleanup**: Kill processes manually if needed:
   ```bash
   pkill -f "sgrep watch"
   rm -rf ~/.sgrep/watch-pids/*
   ```
3. **Check logs**: Hook scripts output JSON to stderr for debugging

### Search Returns No Results

If searches return no results:

1. **Index exists?**: Check if index exists: `ls ~/.sgrep/indexes/`
2. **Index is fresh?**: Run `sgrep index` manually to rebuild
3. **Query too specific?**: Try broader natural language queries
4. **Wrong directory?**: Ensure you're searching the correct project path

### Model Download Issues

If model download fails (e.g., HuggingFace blocked in your region):

1. **Use proxy**: Set `HTTPS_PROXY` environment variable
2. **Verify model**: Run `sgrep config --verify-model` to check status
3. **Manual download**: See [sgrep README](https://github.com/rika-labs/sgrep#offline-installation) for manual installation

## How It Works

### Hook System

The plugin uses Claude Code hooks to manage the lifecycle:

- **SessionStart hooks**: Triggered on `startup`, `resume`, `clear`, and `compact` events
- **SessionEnd hooks**: Triggered when sessions end (all types)

### Process Management

1. **Session Start**: 
   - Extracts session ID from hook input
   - Checks for existing watch process (by PID)
   - Creates index if needed
   - Starts `sgrep watch` in background with `nohup`
   - Stores PID in `~/.sgrep/watch-pids/$SESSION_ID`

2. **Session End**:
   - Retrieves PID from stored file
   - Sends SIGTERM to gracefully stop
   - Falls back to SIGKILL if needed
   - Cleans up PID file

### Skill System

The skill:

1. Provides instructions to Claude on when and how to use sgrep
2. Enables Claude to execute `sgrep search` commands via Bash tool
3. Guides Claude on interpreting search results
4. Helps Claude understand when semantic search is appropriate vs text search

## Development

### Project Structure

```
plugins/sgrep/
├── .claude-plugin/
│   └── plugin.json          # Plugin manifest
├── skills/
│   └── sgrep/
│       └── SKILL.md         # Skill definition (2025 schema)
├── hooks/
│   ├── hooks.json           # Hook configurations
│   └── scripts/
│       ├── session-start.sh # Session start handler
│       └── session-stop.sh  # Session end handler
└── README.md
```

### Testing Locally

1. Test hooks manually:
   ```bash
   echo '{"session_id":"test-123"}' | hooks/scripts/session-start.sh
   echo '{"session_id":"test-123"}' | hooks/scripts/session-stop.sh
   ```
2. Verify skill is loaded: Install plugin and ask Claude to search your codebase

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

Same license as sgrep (Apache 2.0)

## Related Projects

- [sgrep](https://github.com/rika-labs/sgrep) - The semantic code search tool
- [osgrep](https://github.com/Ryandonofrio3/osgrep) - Similar plugin implementation reference

## Support

For issues related to:
- **sgrep itself**: [sgrep GitHub Issues](https://github.com/rika-labs/sgrep/issues)
- **This plugin**: [Create an issue in the sgrep repository](https://github.com/rika-labs/sgrep/issues)
