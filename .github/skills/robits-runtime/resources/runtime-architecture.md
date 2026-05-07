# Runtime Architecture

`main.py` contains the current runtime.

- `Role` stores role-specific prompt templates and calls chat completions through `interact`.
- `Human`, `Ops`, `HR`, `SoftwareEngineer`, and `Angel` define the initial organization roles.
- `System` loads trusted tools and executes them with the current `employee_dict`.
- `tools.yaml` defines trusted tools such as `org.create_role`, `org.pause_role`,
  and `org.retire_role`.
- `parse_tool_instruction` extracts the first valid JSON object or array from a model response.
- `Session` owns a run ID, participants, turn count, system tool handling, and transcript entries.
- `RoundRobinScheduler` provides deterministic time-share recipient selection when a message is not directed.
- `RuntimeEventStream` emits headless session, routing, message, tool, and
  thought events for tests and future TUI observers.
- Role lifecycle state is tracked in memory for active, paused, and retired
  agents, with lifecycle events recording requester and approver context when a
  trusted lifecycle tool provides it.
- Agents carry optional sandbox metadata. Sandboxing is disabled by default; when
  enabled, metadata separates a private per-agent workspace from a shared
  organization workspace and names the backend policy.

## Current vs Target

Current runtime:

- Roles call Chat Completions.
- The runtime can still parse JSON tool instructions from assistant text for compatibility.
- Trusted tool definitions are repo-owned and loaded from `tools.yaml`.
- Runtime sessions record structured turn transcripts and enforce bounded turn counts.
- Undirected recipient selection is deterministic round-robin scheduling; directed messages still take precedence.
- Runtime observability is headless-first: an in-memory event stream can be
  subscribed to during active sessions, and event records can be persisted to
  SQLite for replay.
- Sandbox execution is represented by fakeable runtime abstractions so tests do
  not require containers. Future container or cluster backends should implement
  the same request/result boundary.
- Local runs should treat sandbox metadata as policy metadata until a backend is
  explicitly configured. Headless automation should use fake or local-process
  backends, while future container backends should mount private agent workspaces
  separately from the approved shared organization workspace.

Target runtime:

- Roles use OpenAI-compatible Responses-style interactions when available.
- Approved tools are exposed as function tools with JSON Schema metadata.
- The runtime handles zero or more function calls, executes trusted tools, returns tool outputs by call ID, and then asks the model for final text.
- Parallel execution is added only after state coordination is explicit.

Keep tests isolated from model services. Use live model checks only as smoke validation after deterministic tests pass.
