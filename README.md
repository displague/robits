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

Sessions emit a `RuntimeEventStream` for: `session.created`, `session.completed`, `message.routed`, `message.recorded`, `tool.executed`, `tool_call.requested`, `tool_call.executed`, `tool_call.failed`, `thought.recorded`, `agent.waited`, `clock.state.changed`. Tests can subscribe to the stream for assertions without a terminal.

## Sandboxes

Agents carry optional `SandboxMetadata`. The current implementation provides a fakeable abstraction for tests; production backends (kind/Kubernetes) are described in `docs/kind-sandbox-runtime.md`.

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
| `ROBITS_DIGEST_INTERVAL` | `0` | Turns between episodic digests (0 = disabled) |
| `ROBITS_DIGEST_CONTEXT_CHARS` | `0` | Accumulated chars that trigger digestion (0 = disabled) |
| `ROBITS_DIGEST_ELAPSED_SECONDS` | `0` | Elapsed seconds that trigger digestion (0 = disabled) |
| `ROBITS_IDENTITY_DIGEST_INTERVAL` | `0` | Meaningful turns between identity checkpoints |
| `ROBITS_GOAL_DIGEST_INTERVAL` | `0` | Meaningful turns between goal checkpoints |
| `ROBITS_MAX_PARALLELISM` | `1` | Max concurrent model calls |
| `ROBITS_MAX_API_RETRIES` | `3` | Retries on transient API errors |
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
