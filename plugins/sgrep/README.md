# sgrep Claude Code Plugin

A Claude Code plugin that integrates [sgrep](https://github.com/rika-labs/sgrep) for fast local semantic code search. This plugin automatically manages `sgrep watch` during Claude Code sessions and provides a skill that enables Claude to use sgrep via CLI with structured `--json` output.

## Features

- **Automatic Index Management**: automatically indexes your repository when starting a Claude session
- **Background Watching**: starts `sgrep watch` in the background to keep your index fresh
- **Skill Integration**: provides a skill that enables Claude to use sgrep CLI commands for semantic code search
- **Session Lifecycle Management**: automatically starts watch on session start and stops it on session end
- **Agent-Ready Output**: uses `sgrep search --json` so Claude receives structured results

## Prerequisites

1. install `sgrep`
2. use a Claude Code version with plugin support

## Installation

```bash
/plugin marketplace add rika-labs/sgrep
/plugin install sgrep
```

## Configuration

Useful environment variables:

- `SGREP_HOME`
- `SGREP_CONFIG`
- `SGREP_DEVICE`
- `SGREP_BATCH_SIZE`
- `SGREP_MAX_THREADS`
- `SGREP_CPU_PRESET`
- `HTTP_PROXY` / `HTTPS_PROXY`

## Typical flow

1. `sgrep index`
2. `sgrep search --json "query"`
3. `sgrep watch`
