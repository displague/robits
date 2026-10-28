# Native MCP tools and focused subtasks

Implements the bounded client profile for [MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28)
and its [Tasks extension](https://modelcontextprotocol.github.io/ext-tasks/specification/2026-07-28/tasks.html).
The [Tasks overview](https://modelcontextprotocol.io/extensions/tasks/overview)
describes the durable-handle, polling and mid-flight input lifecycle.
Tracked in #130. PR #80 is independent weather-tool work.

## MCP configuration

Set `ROBITS_MCP_CONFIG` to a trusted JSON file containing named servers:

```json
{
  "local": {"url": "http://localhost:8000/mcp", "timeout": 30},
  "stdio": {"command": "/absolute/path/to/server", "args": [], "timeout": 30}
}
```

HTTP entries optionally accept `headers`; stdio entries accept `env`. Keep
credentials in private configuration. Redirects are disabled. Use servers that
support July's stateless protocol, not servers requiring `initialize` or a legacy
session. Stdio starts a bounded process for each RPC: server task state must
survive across processes if the server returns continuations or task handles.
HTTP is preferred for long-running server tasks.

Give a role the `mcp` capability and `mcp.*` grant (or individual tools selected
from `tools.list`). Discovery loads tools/list pages lazily before the role's
first model invocation. Remote descriptions and inputSchema become ordinary
OpenAI-compatible function tools for both Chat and Responses. Provider-safe
names are stable hashes of server and remote tool names. Arguments and form
responses are validated with JSON Schema; remote schema references are disabled.
The MCP result, including structuredContent and isError, returns under the
model's original tool-call ID. Model text cannot register servers or executable
implementations. `builtin.mcp_call` remains a compatibility bridge and enforces
the same registry grants.

The client supplies version and capabilities on each RPC. It accepts JSON and
POST SSE responses, surfaces in-request notifications as private runtime events,
and preserves opaque MRTR requestState internally. Input-required results return
an `operation_id` and `inputRequests`. Call `mcp.operation` with matching response
IDs to continue. A form response is `{ "action": "accept", "content": {...} }`,
or action `decline`/`cancel`. Schema correctness does not establish user consent:
agents must ask their parent for information or authorization they lack.
URL elicitation and deprecated sampling/roots are not advertised. No OAuth login
or automatic credential acquisition is implemented.

Remote tasks return the same operation handle. `mcp.operation` without responses
polls tasks/get; responses use tasks/update; cancel uses tasks/cancel. Polling
continues after an empty update/cancel acknowledgement: it does not imply a
terminal status, and a cancelled request may still complete successfully.
Polling is
caller-driven and respects pollIntervalMs. No unattended polling loop or
subscriptions/listen stream is started. Pending operations are scoped to caller
and runtime session, capped at 128, with 32 continuation requests per operation.
HTTP payloads and stdio lines have a 4 MiB limit, and requests have timeouts.
HTTP watchdogs interrupt slow response headers/bodies. Stdio writes and reads
share a deadline; owned process trees are reaped even if their original parent
exits first (Windows job objects or POSIX process groups).
Pending operations and local tasks live in process memory; restarting does not
resume them. Persisting and reattaching work is separate future work.

## Subtasks

The main role remains a conversational persona. `agent.focus(task)` is the simple
task-only entry point; `agent.spawn` adds optional tool, model and queue controls.
Both use the same permission, depth, event and input/resume machinery.
Subtasks are private compartments
for focused reflection, inner dialogue, checks and mundane work; they are not
additional public organization members. The role chooses when this is useful,
passes minimal context, and brings back concise findings in its own voice.
Routine replies should not narrate internal tool mechanics. Questions, missing
facts and failures still need clear communication when they affect the answer.
Direct tool use remains appropriate for simple work; spawning is not mandatory.

Private notes and subtask-control tool events are marked private, including
requests made by the surface role. Child execution emits structured observer
events without printing task previews or results into the conversational console.
Operators can inspect these records separately; the parent conversation retains
its own messages and concise returned findings, not the child's working history.

`agent.spawn` keeps its synchronous result behavior. It uses the explicit model
override, `ROBITS_SPAWN_MODEL`, the parent's current model, then the cheap model.
Each child gets an isolated system prompt and task, inherited capabilities and a
subset of its parent's tool grants. It never receives the parent's conversation
history. Include the relevant context and desired deliverable in `task`.
Memory, organization, proposal and wait tools remain blocked.

Set `defer=true` to receive a queued task ID without running it. `agent.subtask`
supports `inspect`, `run`, `respond`, and `cancel`. Inspection returns status,
result, paginated event history, pending question and child IDs; parents may inspect their
descendants, while unrelated agents and sessions cannot. Execution is
cooperative: run advances until completion, failure, or an input request. It
does not run concurrently with the parent.
Inspection defaults to the last 10 events with 2048-character string previews.
Use `after_sequence=-1` to start at the beginning, then `next_sequence` to request
the next page; `limit` accepts 1–50. Full private tool history stays with the child.
Creation, run, response and cancellation return compact handoffs without event
history. Fetch that history explicitly with inspect when it is useful, rather
than replaying the work into the surface persona's context on every action.

A child calls `agent.elicit(message)` when it needs help. This yields after the
tool-call result has been added to its private history. The parent inspects the
question, calls agent.subtask with action respond and a response string, then
action run. The child resumes with its earlier tool-call IDs/results and the
parent's answer. Subsequent calls in the same batch as elicitation receive a
pause error and must be retried after resuming.

`ROBITS_SUBTASK_MAX_DEPTH` defaults to **2**: root 0 → child 1 → grandchild 2.
At depth 2, agent.focus and agent.spawn are absent from exposed schemas and rejected by runtime
execution. Set 0 to disable delegation. `ROBITS_SUBTASK_MAX_TASKS` defaults to 32
per root/session, including completed tasks; each task has a cumulative budget
of 32 model rounds across resumes, and each generate call permits 8 tool rounds.
Cancellation marks queued/waiting descendants cancelled. It cannot interrupt a
synchronous model request already running. Private runtime events include
creation, running, tool results, input requests/responses, completion and errors.

## Validation

Run `python -m unittest`. Protocol, native function continuation, nested depth,
grant restrictions, cross-session isolation and pause/resume tests require no
model service. Live smoke notes belong in local validation logs, including the
actual model tested; local model names and endpoints are not deployment defaults.
Live behavioral checks should use ordinary requests in an established persona,
observe tool/subtask events separately, and check the outward reply and follow-up
for continuity and accurate facts. Explicit "call agent.spawn" prompts remain
protocol diagnostics rather than evidence of natural conversational behavior.
