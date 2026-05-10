# Agent-Driven Tool Creation Pipeline

Agents in robits can propose, approve, and activate new tools within a live session —
without editing any files or restarting. This document describes the pipeline, shows
the exact tool call format at each step, and gives guidance on running it.

## Overview

```
SE ──tools.propose──► proposal store (status: proposed)
                              │
Ops ──tools.approve_proposal──► (status: approved)
                              │
Ops ──tools.rollout_proposal──► ToolRegistry (compiled, live)
                              │
Any agent ──team.pulse──► "SE: on-task"
```

The key constraint: tool code runs in a **restricted sandbox** — no imports, limited
builtins. Approved proposals compile and register atomically at rollout, so the tool
is available immediately in the same session.

---

## Step-by-step with exact tool call format

Agents use native function calling. The tool name `tools.propose` is presented to the
model as `tools__propose` (dots replaced with double underscores) in the Chat
Completions and Responses APIs.

### 1 — SE proposes the tool

SE calls `tools__propose` with a `code` field containing a **direct return statement**
(no function or class definitions, no import statements).

```json
{
  "tool_name": "team.pulse",
  "description": "Report the calling agent's current status.",
  "parameters": {
    "type": "object",
    "properties": {
      "status": {
        "type": "string",
        "description": "Current activity or state to broadcast."
      }
    },
    "required": ["status"]
  },
  "code": "return f\"{get_caller_name()}: {status}\""
}
```

The runtime compiles `code` immediately. If it contains a `def`, `class`, or `import`,
the proposal is rejected with a clear error. On success, the response includes a
`proposal_id` that Ops will need.

**Example success response (abbreviated):**
```json
{
  "proposal_id": "tool-proposal-ef9cf15f-...",
  "tool_name": "team.pulse",
  "status": "proposed",
  "code": "return f\"{get_caller_name()}: {status}\""
}
```

### 2 — Ops lists and reviews pending proposals

Call `tools__list_proposals` with `{"status": "proposed"}`.

### 3 — Ops approves the proposal

Call `tools__approve_proposal` with `{"proposal_id": "tool-proposal-ef9cf15f-..."}`.

`approved_by` defaults to the calling agent's name if omitted.

### 4 — Ops rolls out the tool (compiles and registers it live)

Call `tools__rollout_proposal` with:
```json
{
  "proposal_id": "tool-proposal-ef9cf15f-...",
  "role_name": "SE"
}
```

At this point, `team.pulse` is compiled from the proposal code and registered in
`ToolRegistry`. It is immediately callable — no restart needed. The response confirms:

```json
{
  "proposal": { "status": "operationalized", "granted_roles": ["se"] },
  "grant_result": "Granted tool 'team.pulse' to role 'SE'."
}
```

To grant the tool to additional roles, call `tools__rollout_proposal` again with a
different `role_name`.

### 5 — Agents call the new tool

Call `team__pulse` with `{"status": "on-task, reviewing PRs"}`.

**Result:** `SE: on-task, reviewing PRs`

---

## Sandbox environment

Tool code runs inside a compiled Python function. Available globals:

| Global | Description |
|--------|-------------|
| `get_caller_name()` | Returns the calling agent's name (e.g., `"SE"`) |
| `workspace_read(employee_dict, agent_name, path)` | Read a file from an agent's workspace |
| `workspace_write(employee_dict, agent_name, path, content)` | Write a file to an agent's workspace |
| `workspace_list(employee_dict, agent_name, path)` | List workspace files |
| `memory_search(employee_dict, agent_name, query)` | Search stored memory |
| `work_todo_add(employee_dict, title, content)` | Add a todo item |
| `current_agent_context(employee_dict)` | Return the calling agent's full runtime context (JSON) |

**Builtins:** `abs`, `bool`, `dict`, `enumerate`, `float`, `int`, `len`, `list`,
`max`, `min`, `range`, `round`, `set`, `sorted`, `str`, `sum`, `tuple`, `zip`

**Prohibited:** `import`, `exec`, `eval`, function definitions (`def`), class
definitions, file I/O, network calls outside the provided globals.

**Lambdas** are permitted as expressions within a return statement.

### Code patterns

```python
# ✅ Simple return — the common case
return f"{get_caller_name()}: {status}"

# ✅ Conditional logic
if not message:
    return "Error: message is required."
return f"[{get_caller_name()}] {message}"

# ✅ Lambda as a local expression
transform = lambda x: x.strip()
return f"{get_caller_name()}: {transform(status)}"

# ✅ Using workspace to persist state
workspace_write(employee_dict, get_caller_name(), "status.txt", status)
return f"Status recorded for {get_caller_name()}."

# ✅ Getting current time — use current_agent_context; json is not importable
ctx_json = current_agent_context(employee_dict)
# ctx_json is a JSON string; parse with dict() if needed or extract with str operations

# ❌ Nested function definition — rejected at proposal time
def helper(s):
    return s.upper()
return helper(status)

# ❌ Import statement — rejected at proposal time
import json
return json.dumps({"caller": get_caller_name(), "status": status})
```

---

## Update an existing tool

To replace the implementation of an already-active tool, propose with `action: update`:

Call `tools__propose` with:
```json
{
  "tool_name": "team.pulse",
  "action": "update",
  "description": "Report status with caller name.",
  "parameters": {
    "type": "object",
    "properties": {
      "status": { "type": "string" }
    },
    "required": ["status"]
  },
  "code": "return f\"{get_caller_name()}: {status}\""
}
```

---

## Running the demo

The pipeline requires a model that can:

1. Follow multi-step tool-call instructions using the native function-calling protocol
2. Provide JSON Schema parameter definitions
3. Write a direct return expression in the `code` field

**Recommended:** 8B+ parameter models. Smaller models (3B and below) tend to drop the
`code` field or write function-definition-style code.

### With Ollama (local)

```bash
OPENAI_BASE_URL=http://127.0.0.1:11434/v1/ \
OPENAI_API_KEY=ollama \
ROBITS_PROVIDER_API=chat \
OPENAI_MODEL=granite4.1:8b \
.venv/bin/python main.py --turns 20 --log /tmp/robits-demo.log \
  --prompt "Team, we need a shared status signal.
SE: call tools.propose with tool_name=team.pulse,
description='Report caller status', and code exactly as:
return f\"{get_caller_name()}: {status}\"
Set parameters to: {\"type\":\"object\",\"properties\":{\"status\":{\"type\":\"string\"}},\"required\":[\"status\"]}
Do NOT write def or import in code.
Ops: after the proposal appears in tools.list_proposals, call tools.approve_proposal
with the proposal_id, then tools.rollout_proposal with proposal_id and role_name=SE.
Once rolled out, every agent calls team.pulse to report current status."
```

### With Claude (API)

```bash
ANTHROPIC_MODEL=claude-sonnet-4-6 \
.venv/bin/python main.py --turns 15 --log /tmp/robits-demo.log \
  --prompt "Team, build a team.pulse tool together.
SE proposes via tools.propose (include code: return f\"{get_caller_name()}: {status}\",
parameters must be proper JSON Schema).
Ops approves then rolls out to SE via tools.rollout_proposal.
Every agent reports their status once the tool is live."
```

### Expected transcript (condensed)

```
[Turn 1] CEO → Ops: Team, we need a shared status signal...

[Turn 2] Ops → SE:
  tool_call: tools.list_proposals({"status": "proposed"}) → []
  "No proposals yet — SE, please submit team.pulse."

[Turn 3] SE → Ops:
  tool_call: tools.propose({
    "tool_name": "team.pulse",
    "description": "Report calling agent status.",
    "parameters": {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]},
    "code": "return f\"{get_caller_name()}: {status}\""
  }) → {"proposal_id": "tool-proposal-abc123", "status": "proposed", ...}
  "Proposal submitted: tool-proposal-abc123"

[Turn 4] Ops → SE:
  tool_call: tools.approve_proposal({"proposal_id": "tool-proposal-abc123"})
           → {"status": "approved", ...}
  tool_call: tools.rollout_proposal({"proposal_id": "tool-proposal-abc123", "role_name": "SE"})
           → {"proposal": {"status": "operationalized"}, "grant_result": "Granted..."}
  "team.pulse is now live and granted to SE."

[Turn 5] SE → HR:
  tool_call: team.pulse({"status": "proposal complete"})
           → "SE: proposal complete"

[Turn 6] HR → Samandriel:
  tool_call: team.pulse({"status": "ready for tasks"})
           → "HR: ready for tasks"
```

---

## Architecture notes

**Proposal → registry flow:** `tools.rollout_proposal` checks if the tool is already
in `ToolRegistry`. If not, it compiles the proposal's `code` with the declared
parameter schema and registers it in-place. Subsequent agents can call it immediately.

**Update proposals:** When `action: update`, `tools.rollout_proposal` calls
`ToolRegistry.replace_definition`, which swaps the compiled function and re-registers
all aliases. System tools (`system_tool: true`) cannot be replaced this way.

**Proposal store:** Proposals persist across sessions in `var/tool_proposals.json`
(configurable via `ROBITS_TOOL_PROPOSALS_FILE`). Registered tools are only
in-process; restart re-registers from `tools.yaml` only. If you want a custom tool
to survive restarts, add it to `tools.yaml` after a successful session.

**Date, time, and location context:** Every agent's system prompt includes
`current_datetime_local`, `current_date_local`, and `timezone` from `agent_runtime_context()`.
Set `ROBITS_TIMEZONE` (e.g., `America/New_York`) and `ROBITS_LOCATION` to populate the
location field. Agent-authored tool code can call `current_agent_context(employee_dict)`
to read these at call time.
