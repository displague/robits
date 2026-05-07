# Robits Architecture Vision

Date: 2026-05-07

Robits is moving from a single-file organization simulation into a local-first
agent runtime. The immediate goal is not to hide all complexity behind a large
framework. The goal is to create small, testable runtime boundaries so agents
can run headlessly, use tools, persist memory, and expose enough observability
for a human CEO to inspect what happened.

## Current Runtime

`main.py` currently owns the working runtime:

- `Role` and concrete role classes define initial organization members.
- `Session` owns run IDs, participants, turn counts, transcript entries, and
  system tool processing.
- `RoundRobinScheduler` provides deterministic time-share routing for
  undirected messages; directed messages such as `HR, ...` take precedence.
- `System` loads trusted tool definitions from `tools.yaml` and executes
  registered tools by namespaced name.
- `ToolRegistry` normalizes trusted repo-owned tools into metadata that can be
  exposed to OpenAI-compatible Responses or Chat Completions APIs.
- Tests use fake roles and temporary files so runtime behavior does not require
  a live model service.

Keeping this in `main.py` is acceptable while boundaries are still forming. New
work should make those boundaries clearer, not scatter half-finished abstractions
across modules.

## Target Boundaries

### Roles and Agents

Roles are prompt and policy definitions. Agents are role instances with identity,
lifecycle state, memory, contacts, tools, and session participation history.
Lifecycle states should include at least `proposed`, `active`, `paused`, and
`retired`.

HR owns lifecycle proposals and approvals. Operators own runtime health,
availability, and environment coordination. Software engineers propose trusted
tools and code changes. Every active agent may request tools, but executable
tool definitions remain trusted runtime artifacts until a later permission model
exists.

### Sessions and Messages

Sessions own run metadata, participants, routing, turn budgets, and transcripts.
They are the unit that headless tests and smoke runs execute. A message record
should distinguish:

- public conversation content
- system/runtime events
- tool call requests and results
- private thoughts or inner monologue
- memory retrieval or digest insertion

The current `TranscriptEntry` is a short-term in-memory shape. SQLite-backed
message tables should become the durable source of truth.

The target message boundary is a canonical envelope with sender, receiver,
content, kind, tool-call IDs, tool-result IDs, token metadata, timestamps, and
visibility policy. Provider-specific request and response formats should be
translated at the edge, not passed through the whole runtime.

### Provider Adapters

Model invocation should move behind provider adapters. The first adapter should
preserve the current Chat Completions behavior. A Responses-compatible adapter
can then be added without changing session, memory, or lifecycle code.

Provider adapters own:

- request formatting
- streaming and final-response assembly
- tool-call delta parsing
- tool-result message formatting
- token usage metadata
- capability flags for tools, structured output, and reasoning fields

### Tools

Tools are trusted, namespaced capabilities. The runtime may expose tool metadata
to a model, but model output can only request registered tools. It cannot define
new executable tools directly.

The registry should keep supporting:

- canonical names such as `org.create_role`
- model-facing names such as `org__create_role`
- JSON Schema parameters
- explicit tool-call result records

Local OpenAI-compatible servers are a validation target, but durable docs should
describe the generic API shape rather than a machine-specific endpoint or model.
Structured output should be capability-gated: prefer provider-native schema
support when available, and fall back to validated JSON extraction when it is
not.

### Memory and Context

Robits should use SQLite as the local memory substrate, with FTS over messages,
thoughts, tool results, and memory records. Memory should support filters by
agent, session, relationship, source, conversation type, and date window.

Growing context windows still break fast local experimentation. The runtime
should keep prompts small by combining recent turns with retrieved memory and
compact summary records. Use `memory digest` as the durable term for compacted
context artifacts. A memory digest must keep provenance:

- source record IDs
- time range
- agent and relationship filters
- prompt or digest policy version
- created-at timestamp
- links that allow reanalysis or re-digestion from the original records

Digests are retrieval aids, not destructive replacements for raw records.

### Thinking and Observability

Thinking or inner monologue should be modeled as private runtime records, not as
plain public transcript text. The human CEO needs a way to inspect these records,
but the runtime should keep access policy explicit so future agents can have
selectively private coworker, family, therapy, and personal project contexts.

Observability should be headless first:

- emit session, routing, message, tool, thought, and memory events
- persist those events into SQLite
- allow tests to assert event emission without a terminal
- allow a later TUI to attach to live events or replay stored sessions

The TUI is an observer over the runtime. It should not become the runtime.

### Local Model Operation

Robits should remain usable with local OpenAI-compatible model servers. LM Studio
documents local server operation, tool/function calling support, structured
output, and context overflow policies such as stopping at the limit, truncating
the middle, or rolling window behavior. Robits should still implement its own
runtime-level memory and digest policy rather than relying only on model-server
context overflow settings.

Prompt assembly should estimate or count token budgets before invoking a model.
The runtime should compact or retrieve memory before provider overflow behavior
is triggered. Provider overflow policies are a last line of defense, not the
primary memory strategy.

Headless validation remains required. Live model smoke runs are optional and
should be bounded with small turn counts and generic OpenAI-compatible
configuration.

## What Moves Out of `main.py`

Move only after tests cover behavior:

1. `robits/runtime/session.py`: `Session`, transcript event types, and routing.
2. `robits/runtime/scheduler.py`: deterministic schedulers and later parallel
   schedulers.
3. `robits/runtime/tools.py`: tool registry, metadata export, and execution.
4. `robits/agents/`: role definitions, agent lifecycle, and contact policy.
5. `robits/memory/`: SQLite schema, repositories, FTS search, and memory digests.
6. `robits/ui/`: TUI adapters over persisted or streamed runtime events.

Avoid large file splits before behavior is stable. Branch salvage showed that
mechanical module moves can easily erase tests or current runtime fixes.

## Migration Sequence

1. **Architecture and constraints**: document target boundaries and local-model
   constraints. This closes issue #5.
2. **SQLite memory substrate**: add schema, repositories, FTS search, and
   gitignored local database paths. This is issue #6.
3. **Memory digests**: add compacted context artifacts with provenance and
   source expansion. This is issue #15.
4. **Provider adapters and message envelopes**: isolate Chat Completions and
   Responses formatting from sessions and memory, including capability flags for
   tools, structured output, and token metadata.
5. **Agent lifecycle**: formalize HR lifecycle actions and preserve
   `org.create_role` compatibility. This is issue #9.
6. **Observability and TUI**: emit headless runtime events and add a TUI spyglass
   over live or persisted sessions. This is issue #16.
7. **Responses tool loop**: replace ad hoc JSON tool extraction with tested
   OpenAI-compatible tool-call routing while keeping text JSON compatibility
   during migration.
8. **Module extraction**: split `main.py` after the boundaries above have tests.
9. **Parallel execution**: add parallel role execution only after SQLite-backed
   state coordination and event ordering are explicit.

## References

- LM Studio docs index: https://lmstudio.ai/llms.txt
- LM Studio OpenAI-compatible Responses API:
  https://lmstudio.ai/docs/developer/openai-compat/responses
