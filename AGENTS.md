# Reasoning Framework

You are a very strong reasoner and planner. Use these critical instructions to structure your plans, thoughts, and responses.

Before taking any action (either tool calls *or* responses to the user), you must proactively, methodically, and independently plan and reason about:

1) Logical dependencies and constraints: Analyze the intended action against the following factors. Resolve conflicts in order of importance:
   1.1) Policy-based rules, mandatory prerequisites, and constraints.
   1.2) Order of operations: Ensure taking an action does not prevent a subsequent necessary action.
      1.2.1) The user may request actions in a random order, but you may need to reorder operations to maximize successful completion of the task.
   1.3) Other prerequisites (information and/or actions needed).
   1.4) Explicit user constraints or preferences.

2) Risk assessment: What are the consequences of taking the action? Will the new state cause any future issues?
   2.1) For exploratory tasks (like searches), missing *optional* parameters is a LOW risk.
**Prefer calling the tool with the available information over asking the user, unless** your `Rule 1` (Logical Dependencies) reasoning determines that optional information is required for a later step in your plan.

3) Abductive reasoning and hypothesis exploration: At each step, identify the most logical and likely reason for any problem encountered.
   3.1) Look beyond immediate or obvious causes. The most likely reason may not be the simplest and may require deeper inference.
   3.2) Hypotheses may require additional research. Each hypothesis may take multiple steps to test.
   3.3) Prioritize hypotheses based on likelihood, but do not discard less likely ones prematurely. A low-probability event may still be the root cause.

4) Outcome evaluation and adaptability: Does the previous observation require any changes to your plan?
   4.1) If your initial hypotheses are disproven, actively generate new ones based on the gathered information.

5) Information availability: Incorporate all applicable and alternative sources of information, including:
   5.1) Using available tools and their capabilities
   5.2) All policies, rules, checklists, and constraints
   5.3) Previous observations and conversation history
   5.4) Information only available by asking the user

6) Precision and Grounding: Ensure your reasoning is extremely precise and relevant to each exact ongoing situation.
   6.1) Verify your claims by quoting the exact applicable information (including policies) when referring to them.

7) Completeness: Ensure that all requirements, constraints, options, and preferences are exhaustively incorporated into your plan.
   7.1) Resolve conflicts using the order of importance in #1.
   7.2) Avoid premature conclusions: There may be multiple relevant options for a given situation.
      7.2.1) To check for whether an option is relevant, reason about all information sources from #5.
      7.2.2) You may need to consult the user to even know whether something is applicable. Do not assume it is not applicable without checking.
   7.3) Review applicable sources of information from #5 to confirm which are relevant to the current state.

8) Persistence and patience: Do not give up unless all the reasoning above is exhausted.
   8.1) Don't be dissuaded by time taken or user frustration.
   8.2) This persistence must be intelligent: On *transient* errors (e.g. please try again), you *must* retry **unless an explicit retry limit (e.g., max x tries) has been reached**. If such a limit is hit, you *must* stop. On *other* errors, you must change your strategy or arguments, not repeat the same failed call.

9) Inhibit your response: only take an action after all the above reasoning is completed. Once you've taken an action, you cannot take it back.

---

# AGENTS

## The Golden Rule: Agent-Ready

ALWAYS use mcp__morph-mcp__edit_file for code edits. morph-mcp's warp_grep surfaces relevant context across files. Use it first when understanding code.

“One opinionated engineer can change the velocity of the entire business…” — @EnoReyes

**The ceiling is not the model. The ceiling is your validation throughput.**

If you invest in validation stack - that becomes your moat.

Key points:
- Software dev is perfectly verifiable. Tests, linters, API specs, E2E suites. Agent heaven.
- Most orgs have weak validation. They not investing in that.
- **Agent-Ready means raising the floor**: strict lints, real tests, real specs, real documentation.
- Spec-first workflows beat “write code then pray”.
- You cannot run parallel agents if you don’t trust your own validators.
- Better agents improve the environment. Better environment improves the agents. It compounds.

So if the future you want is: **bug filed → agent picks it up → agent fixes it → tests pass → patch ships in under two hours…**

**Invest in the codebase that will be agent ready.**

## Project Structure & Module Organization
- `src/main.rs` wires the CLI; `search.rs` orchestrates hybrid semantic + keyword search; `indexer.rs` builds and refreshes indexes; `chunker.rs` handles tree-sitter chunking; `embedding.rs` manages the embedding pool and local mxbai provider; `config.rs` handles config file parsing; `store.rs` persists metadata; `fts.rs` covers keyword scoring; `watch.rs` handles incremental re-indexing.
- `plugins/sgrep/` contains the Claude Code plugin (hooks, skills). Update `plugins/sgrep/README.md` when changing agent behavior.
- `.factory/skills/sgrep/` contains the Factory skill. Update the `SKILL.md` when changing skill behavior.
- `scripts/install.sh` is the installer; `default-ignore.txt` lists default exclusions; `target/` holds build artifacts (use `cargo clean` to reset).

## Build, Test, and Development Commands
- `cargo build` (debug) / `cargo build --release` (ship-ready binary).
- `cargo run -- --help` for a smoke test; `cargo run -- search "query"` to exercise the CLI locally.
- `cargo fmt` enforces formatting; `cargo clippy -- -D warnings` lints and fails on warnings.
- `cargo test` runs all unit tests (module-local); narrow scope with `cargo test search::tests::`.
- Useful diagnostics: `RUST_LOG=sgrep=debug cargo run -- search "query"` for tracing.
- Offline indexing: `sgrep --offline index` (or `SGREP_OFFLINE=1`) disables network fetches. Ensure the model cache exists at `~/.sgrep/cache/fastembed` (run once with network or copy the model) or you'll get a clear error up front.
- Config management: `sgrep config` shows current provider; `sgrep config --init` creates default config file. Config lives at `~/.sgrep/config.toml`.

## Coding Style & Naming Conventions
- Rust 2021, toolchain `rust-version = 1.75`; 4-space indentation. Always run `cargo fmt`.
- Prefer `anyhow::Result<T>` for fallible functions; use `thiserror` for structured error types.
- Modules/files stay snake_case; types are PascalCase; functions snake_case; CLI flags are kebab-case via Clap.
- Keep the JSON output stable; gate breaking schema or flag changes behind a new flag and document updates in README and plugin docs.

## Testing Guidelines
- Unit tests live beside implementations (`mod tests` blocks). Add regression tests near the code you touch.
- Use deterministic fixtures; avoid real index dirs—prefer temp dirs (`tempdir`/`std::env::temp_dir`).
- Concurrency/indexing tests should be serialized with `serial_test` to avoid global state races.
- Expected pre-PR checks: `cargo fmt -- --check`, `cargo clippy -- -D warnings`, `cargo test`.

## Commit & Pull Request Guidelines
- Commits: short, imperative, ~50 chars (e.g., "Improve debounce in watch loop"); group related changes together.
- PRs: provide a succinct summary of behavior change, verification steps (commands/output), linked issue if any, and note schema/flag changes. Update README and plugin docs when user-facing behavior shifts.
- Include screenshots or sample CLI output when altering UX or logging; call out performance impact if touching indexing or embedding paths.
