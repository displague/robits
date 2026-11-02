# AI-Powered Organization Simulation

A Python-based simulation of an AI-driven organisation where agents with distinct roles collaborate through a shared conversation loop backed by SQLite memory and OpenAI-compatible model APIs.

## Table of Contents

- [Roles](#roles)
- [How It Works](#how-it-works)
- [Memory](#memory)
- [Personas](#personas)
- [Clock States and Circadian Rhythm](#clock-states-and-circadian-rhythm)
- [Embedding Search](#embedding-search)
- [Observability](#observability)
- [Workspace privacy](#workspace-privacy)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Simulation](#running-the-simulation)
- [Local Models (Ollama / LMStudio)](#local-models-ollama--lmstudio)
- [Testing](#testing)

## Roles

| Role | Description |
|---|---|
| CEO | Human-in-the-loop: reads from stdin, issues directives. |
| Ops | Operations: oversees the agent environment; holds `operator` capability. |
| SE | Software Engineer: designs and proposes new trusted tools; uses the costly model. |
| HR | Human Resources: manages role lifecycle; holds `hr` capability. |
| Samandriel | Angel guardian: speaks Enochian, protects employees. |

HR can create additional dynamic roles at runtime via `org.create_role`.

Personas (see below) let individual identities fill these role slots with their own name, backstory, and memories.

## How It Works

Agents can use configured MCP servers through native function tools and delegate
focused subtasks with configurable nesting, inspection, events and parent input.
See [MCP and subtasks](docs/mcp-subtasks.md) for configuration and limits.

Sessions own a run ID, participant list, transcript, and a `RoundRobinScheduler`. Undirected messages cycle through all active participants; directed messages (`HR, ...`) still go through the scheduler via `observe()` to keep ordering consistent.

The System role parses JSON tool instructions from agent responses and executes registered tools. Approved tool proposals (from SE) are activated at rollout without a restart.

## Memory

SQLite memory store at `robits/memory/sqlite.py` with tables for sessions, agents, contacts, channels, messages, thoughts, todos, tool calls, memory entries, and memory digests. FTS5 full-text search with cascade expansion surfaces parent digests from raw hits. An async variant (`async_sqlite.py`) wraps the same interface for async runtimes.

Memory digests are compacted artifacts with generation/version tracking, source references, and recursive expansion so future runs can re-analyse the original material.

Enable automatic digest creation with:

```
ROBITS_DIGEST_INTERVAL=10          # episodic digest every 10 turns
ROBITS_IDENTITY_DIGEST_INTERVAL=5  # identity checkpoint every 5 meaningful turns
ROBITS_GOAL_DIGEST_INTERVAL=5      # goal checkpoint every 5 meaningful turns
```

## Personas

`personas.yaml` pre-seeds individual identity memories for agents before their first session. The file uses `username` + `role` + `full_name` keys so multiple distinct individuals can fill the same role type:

```yaml
- username: alex_chen
  full_name: Alex Chen
  role: SE
  memories:
    - kind: thought
      content: "Python was my first serious language."
      visibility: private
    - kind: digest
      digest_type: identity
      content: "Alex is a pragmatic backend engineer who values clean APIs."
      relationship_type: personal

- username: jamie_okonkwo
  full_name: Jamie Okonkwo
  role: SE
  memories:
    - kind: digest
      digest_type: identity
      content: "Jamie specialises in distributed systems."
      relationship_type: personal
```

When `build_employee_dict()` receives the persona map, it instantiates each persona under its `username` key instead of the default role-type key. The `agents` table stores `username`, `first_name`, `last_name`, and `full_name` for mention matching.

Seeding is idempotent: if any memory record already exists for the agent, the seed is skipped.

Configure the file path with `ROBITS_PERSONAS_FILE` (defaults to `personas.yaml` in the working directory).

### Tool allowlists and budgets

Every model-backed role starts with broad default grants, which suits large hosted models. Small local models do better with a handful of tools. A persona can declare its own:

```yaml
- username: sam_small
  full_name: Sam Small
  role: SE
  tools:
    - agent.think
    - memory.search
    - mcp:weather.lookup
  tool_budget: 3
  max_tool_rounds: 3
  memories: []
```

- `tools` uses the grant syntax (exact names or `namespace.*`, dotted form) and **replaces** the role's default grants. Schemas are sent in list order. Name one MCP tool as `mcp:<server>.<tool>` (the server key from the MCP config and the tool's remote name); it resolves to the registry name `tools.list` shows (`mcp.t_<slug>_<hash>`), which also works. `mcp.*` selects every MCP tool. Listing an MCP tool gives the persona the `mcp` capability.
- `tool_budget` caps how many tool schemas are sent per round (default `ROBITS_TOOL_BUDGET`). Tools past the budget stay granted but are not offered.
- `max_tool_rounds` caps consecutive tool-call rounds per model call (default `ROBITS_MAX_TOOL_ROUNDS`).

With a `tools` list, a call to an unknown or disallowed tool, or with bad arguments, gets an error that ends with the tool names the persona may use. The error uses up a round. `tools.grant` and other runtime grants cannot widen the persona beyond its list unless an `operator` role (or the human) makes the grant. Personas without `tools` keep the default grants and plain errors. Subtasks inherit the budget and round cap.

### @mention detection

When an org-chat message contains `@username` or any recognisable form of an agent's name (first name, last name, full name), the mentioned agent receives a lightweight note prepended to their next turn's prompt. The scheduler is not interrupted; mentions stay in the normal rotation.

## Clock States and Circadian Rhythm

Agents operate in one of three clock states that reflect the organisation's work rhythm:

| State | Meaning | Temperature | top_p | Org-chat context |
|---|---|---|---|---|
| `on` | Work time | low (×0.6) | low (×0.6) | included |
| `break` | Break time | moderate (×0.9) | moderate (×0.9) | suppressed |
| `off` | Personal/family time | high (×1.3) | high (×1.3) | suppressed |

`ROBITS_CLOCK_STATE` sets the base state. Temperature and `top_p` are modulated relative to each agent's own `base_temperature` and `base_top_p` so personality variation is preserved.

### Scheduled breaks

Set `ROBITS_BREAK_SCHEDULE` to comma-separated `HH:MM-HH:MM` windows in local time. The session auto-transitions to `break` during these windows without changing the base clock state:

```
ROBITS_BREAK_SCHEDULE=12:00-13:00,15:00-15:30
```

Agents can call `org.schedule` to see the current clock state and configured break windows.

## Embedding Search

sqlite-vec provides vector-similarity search over memory records. Set an embedding model and the search automatically enriches FTS results with semantic matches:

```
ROBITS_EMBEDDING_MODEL=granite-embedding:latest
ROBITS_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1/
ROBITS_EMBEDDING_API_KEY=ollama
```

Embeddings are computed lazily at agent wake-up (`_build_wait_summary`) and at session end. `embed_pending` processes only records not yet embedded for the configured model; different models can coexist in the same database.

`search_hybrid` merges FTS cascade results with semantic results, deduplicating on `(kind, record_id)`.

## Observability

Sessions emit a `RuntimeEventStream` for: `session.created`, `session.completed`, `message.routed`, `message.recorded`, `tool.executed`, `tool_call.requested`, `tool_call.executed`, `tool_call.failed`, `thought.recorded`, `agent.waited`, `clock.state.changed`, and in headless mode `session.sleeping` (reason, seconds, next wake time per persona), `session.idle_exit` and `session.model_error`. Access to another persona's workspace emits `workspace.access` (private visibility). Tests can subscribe to the stream for assertions without a terminal.

## Sandboxes

Agents carry optional `SandboxMetadata`. The current implementation provides a fakeable abstraction for tests; production backends (kind/Kubernetes) are described in `docs/kind-sandbox-runtime.md`.

## Workspace privacy

Each persona has a private workspace (`var/agents/<name>`). Roles with the `hr` or `operator` capability can open another persona's workspace, through the `agent.files_*` tools, `builtin.file_search` and `builtin.shell_run`. That access is possible but accountable:

- **Reason required.** A call into another persona's workspace must pass `reason`. Without one it is refused.
- **Audit log.** Each access is appended to `ROBITS_ACCESS_AUDIT_FILE` (default `var/access_audit.jsonl`), outside every workspace. An entry records the caller, the owner, the operation, the path, the time and the reason, never file contents or shell commands. Each access is also written to the JSONL log and emitted as a `workspace.access` event.
- **Owner notice.** The owner sees a notice on its next turn. `ROBITS_ACCESS_NOTICE` chooses how much it is told:
  - `immediate` (the default): who, which path, when and the reason;
  - `digest`: counts per caller and operation only;
  - `none`: no notice, for example during an investigation. The access is still logged.
- **Review.** Operators can list recent accesses with `org.access_audit` (filter by `owner` or `caller`). The Ops role has it by default.
- **Prompt.** Roles that hold this power are told in their system prompt that access is logged and reviewed, and is meant for lifecycle and wellbeing duties only.

This is accountability, not isolation. OS-level separation between personas is #70.

## Installation

```bash
pip install -r requirements.txt
```

For local models with embedding support, `sqlite-vec` is included in `requirements.txt`.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | API key (any non-empty string for local servers) |
| `OPENAI_BASE_URL` | — | Compatible endpoint (Ollama: `http://127.0.0.1:11434/v1/`) |
| `ROBITS_MODEL` / `OPENAI_MODEL` | `gpt-4o-mini` | Default chat model |
| `ROBITS_CHEAP_MODEL` | `ROBITS_MODEL` | Model for cheap interactions |
| `ROBITS_COSTLY_MODEL` | `ROBITS_MODEL` | Model for SE and costly interactions |
| `ROBITS_SPAWN_MODEL` | `ROBITS_CHEAP_MODEL` | Default model for `agent.spawn` sub-agents; overrides `ROBITS_CHEAP_MODEL` for spawned executors (e.g. `functiongemma-270m-it`) |
| `ROBITS_LOOP_DETECT_THRESHOLD` | `3` | Consecutive identical idle responses (no tool calls, no directed routing) before the session halts with `session.loop_detected`; must be an integer ≥ 2 |
| `ROBITS_PROVIDER_API` | `responses` | `responses` or `chat` / `chat_completions` |
| `ROBITS_MAX_CONTEXT_TOKENS` | `0` (unlimited) | Trim conversation history to this token budget before each model call (~4 chars/token). Useful for small-context local models (e.g. `3500` for 4k-context models) |
| `ROBITS_MAX_TOOL_ROUNDS` | `8` | Consecutive tool-call rounds per model call; personas can override with `max_tool_rounds` |
| `ROBITS_TOOL_BUDGET` | `0` (unlimited) | Most tool schemas sent per round; personas can override with `tool_budget` |
| `ROBITS_MEMORY_DB` | `~/.local/share/robits/memory.db` | SQLite path for memory storage |
| `ROBITS_CLOCK_STATE` | `on` | Base clock state: `on`, `break`, or `off` |
| `ROBITS_BREAK_SCHEDULE` | — | Scheduled break windows, e.g. `12:00-13:00,15:00-15:30` |
| `ROBITS_PERSONAS_FILE` | `personas.yaml` | Path to persona seed file |
| `ROBITS_EMBEDDING_MODEL` | — | Embedding model name; disables semantic search when unset |
| `ROBITS_EMBEDDING_BASE_URL` | `OPENAI_BASE_URL` | Embedding endpoint (can differ from chat endpoint) |
| `ROBITS_EMBEDDING_API_KEY` | `OPENAI_API_KEY` | Embedding API key |
| `ROBITS_MEMORY_MAX_DEPTH` | `3` | Max recursive digest expansion depth |
| `ROBITS_MEMORY_MAX_ROWS` | `100` | Max rows per memory tool result |
| `ROBITS_MEMORY_CACHE_THRESHOLD` | `8192` | Bytes above which memory results are offloaded to workspace |
| `ROBITS_ACCESS_AUDIT_FILE` | `var/access_audit.jsonl` | Append-only log of access to other personas' workspaces (see [Workspace privacy](#workspace-privacy)) |
| `ROBITS_ACCESS_NOTICE` | `immediate` | Tell a workspace's owner about access: `immediate`, `digest` or `none` |
| `ROBITS_DIGEST_INTERVAL` | `0` | Turns between episodic digests (0 = disabled) |
| `ROBITS_DIGEST_CONTEXT_CHARS` | `0` | Accumulated chars that trigger digestion (0 = disabled) |
| `ROBITS_DIGEST_ELAPSED_SECONDS` | `0` | Elapsed seconds that trigger digestion (0 = disabled) |
| `ROBITS_IDENTITY_DIGEST_INTERVAL` | `0` | Meaningful turns between identity checkpoints |
| `ROBITS_GOAL_DIGEST_INTERVAL` | `0` | Meaningful turns between goal checkpoints |
| `ROBITS_MAX_PARALLELISM` | `1` | Max concurrent model calls |
| `ROBITS_MAX_API_RETRIES` | `3` | Retries on transient API errors |
| `ROBITS_HEADLESS` | off | Run without a stdin Human (same as `--headless`); `1`/`true`/`yes`/`on` enables |
| `ROBITS_MIN_TURN_INTERVAL_SECONDS` | `0` | Headless: minimum seconds between two turns of the same persona |
| `ROBITS_MAX_TURNS_PER_HOUR` | `0` (unlimited) | Headless: session-wide turn budget over a sliding hour |
| `ROBITS_IDLE_MAX_SLEEP_SECONDS` | `300` | Headless: longest single idle sleep before re-checking (0 = no cap) |
| `ROBITS_ERROR_BACKOFF_SECONDS` | `5` | Headless: first backoff after a failed turn; doubles per consecutive failure |
| `ROBITS_MAX_CONSECUTIVE_ERRORS` | `5` | Headless: consecutive failed turns before the run gives up and exits 1 (`0` retries forever) |
| `ROBITS_LOCATION` | — | Default location string for agent context |
| `ROBITS_TIMEZONE` | — | Default timezone (e.g. `America/New_York`) |

## Running the Simulation

```bash
python main.py
```

For non-interactive smoke checks:

```bash
python main.py --prompt "Ops, say hello to HR" --turns 1
```

### Headless mode

`python main.py --headless` (or `ROBITS_HEADLESS=1`) runs model-backed personas without a stdin Human:

- No CEO is added to the roster, Human personas are skipped, and no role gets the CEO's unconditional tool access. Operator powers come from the `operator` capability.
- Inputs come from `--prompt` (delivered as a message from `external`), alarms, and `Session.deliver()` for events from integrations. An input goes to the next due persona, or to the persona it addresses (`Name: ...`).
- A turn happens only when there is something to attend to: a queued input, a non-empty reply to pass on, an expired wait, or a due alarm. Silence or `agent.wait` ends a turn without forcing anyone else's.
- When nobody is due (all waiting, off the clock, minimum interval, turn budget, or error backoff) the session sleeps until the earliest wake time or until `Session.deliver()`, `wake()`, `set_clock_state()` or SIGTERM. A run with `--turns` that has nothing scheduled ends; without `--turns` it runs until stopped.
- Addressed inputs and replies remain queued while their recipient rests or is inside its minimum interval. Other eligible recipients can still receive their inputs; delivery does not require a reply.
- Model failures before any tool attempt retry the same recipient with its reminder and input intact. After a tool attempt, a failed turn exits with `session.error_exit` and `tool_outcome_requires_inspection`, avoiding automatic replay of uncertain side effects.
- On POSIX, SIGTERM stops after the current turn and exits 0. Windows callers should use `Session.request_stop()` for cooperative shutdown; `os.kill(SIGTERM)` terminates the process directly. JSONL events go to stderr unless `--log` is given.

Capture to a log file while watching:

```bash
python main.py --prompt "Ops, say hello to HR" --turns 1 --log run.log
```

## Local Models (Ollama / LMStudio)

`run_local.py` seeds a fresh memory database and delegates to `main.py`. It demonstrates persona recall with pre-seeded work and personal memories.

### Ollama

Ollama exposes an OpenAI-compatible Responses API at `/v1/responses` (since v0.13.3), but it is **non-stateful**: `previous_response_id` is silently ignored. This breaks the within-turn tool-call continuation in `ResponsesProvider`. Use `ROBITS_PROVIDER_API=chat` with Ollama:

```bash
OPENAI_BASE_URL=http://127.0.0.1:11434/v1/ \
OPENAI_API_KEY=ollama \
OPENAI_MODEL=granite4.1:3b \
ROBITS_PROVIDER_API=chat \
ROBITS_EMBEDDING_MODEL=granite-embedding:latest \
ROBITS_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1/ \
ROBITS_EMBEDDING_API_KEY=ollama \
python run_local.py --prompt "SE, do you have experience with Python or cooking?"
```

Tested with `granite4.1:3b` for tool calling; `granite-embedding:latest` and `embeddinggemma:latest` for embeddings. Models are pulled with `ollama pull <name>`.

### LM Studio

LM Studio's Responses API is stateful and honours `previous_response_id`, so the default `ROBITS_PROVIDER_API=responses` works correctly:

```bash
OPENAI_BASE_URL=http://127.0.0.1:1234/v1/ \
OPENAI_API_KEY=lmstudio \
OPENAI_MODEL=granite-4-micro \
ROBITS_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5 \
ROBITS_EMBEDDING_BASE_URL=http://127.0.0.1:1234/v1/ \
ROBITS_EMBEDDING_API_KEY=lmstudio \
python run_local.py --prompt "SE, do you have experience with Python or cooking?"
```

Note: tool calling and remote MCP must be enabled in LM Studio's Developer Settings.

## Testing

```bash
python -m pytest tests/
```

Tests do not require an external model service or a running Ollama instance. Embedding tests use injected vectors (`_query_vector` parameter) to avoid network calls.
