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

### 1 — SE proposes the tool

SE calls `tools.propose` with a `code` field containing a **direct return statement**
(no function definitions, no lambdas).

```json
{
  "exec": "tools.propose",
  "args": {
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
}
```

The runtime compiles `code` immediately. If it contains a `def` or `lambda`, the
proposal is rejected with a clear error. On success, the response includes a
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

```json
{
  "exec": "tools.list_proposals",
  "args": { "status": "proposed" }
}
```

### 3 — Ops approves the proposal

```json
{
  "exec": "tools.approve_proposal",
  "args": {
    "proposal_id": "tool-proposal-ef9cf15f-..."
  }
}
```

`approved_by` defaults to the calling agent's name if omitted.

### 4 — Ops rolls out the tool (compiles and registers it live)

```json
{
  "exec": "tools.rollout_proposal",
  "args": {
    "proposal_id": "tool-proposal-ef9cf15f-...",
    "role_name": "SE"
  }
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

To grant the tool to additional roles, call `tools.rollout_proposal` again with a
different `role_name`.

### 5 — Agents call the new tool

```json
{
  "exec": "team.pulse",
  "args": { "status": "on-task, reviewing PRs" }
}
```

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

**Builtins:** `bool`, `int`, `str`, `float`, `list`, `dict`, `tuple`, `set`,
`len`, `max`, `min`, `sum`, `range`, `sorted`, `enumerate`, `zip`, `abs`, `round`

**Prohibited:** `import`, `exec`, `eval`, function definitions (`def`/`lambda`),
class definitions, file I/O, network calls outside the provided globals.

### Code patterns

```python
# ✅ Simple return — the common case
return f"{get_caller_name()}: {status}"

# ✅ Conditional logic
if not message:
    return "Error: message is required."
return f"[{get_caller_name()}] {message}"

# ✅ Using workspace to persist state
workspace_write(employee_dict, get_caller_name(), "status.txt", status)
return f"Status recorded for {get_caller_name()}."

# ❌ Nested function definition — rejected at proposal time
def helper(s):
    return s.upper()
return helper(status)

# ❌ Lambda — rejected at proposal time
transform = lambda x: x.strip()
return transform(status)
```

---

## Update an existing tool

To replace the implementation of an already-active tool, propose with `action: update`:

```json
{
  "exec": "tools.propose",
  "args": {
    "tool_name": "team.pulse",
    "action": "update",
    "description": "Report status with timestamp.",
    "parameters": {
      "type": "object",
      "properties": {
        "status": { "type": "string" }
      },
      "required": ["status"]
    },
    "code": "ctx = current_agent_context(employee_dict); import json; ts = json.loads(ctx).get('current_datetime_local',''); return f\"{get_caller_name()} [{ts}]: {status}\""
  }
}
```

Wait — `import` is not allowed in the sandbox. Use the `current_agent_context` return
value via a workaround if you need the timestamp. In practice, the agent can include
date/time from the runtime context provided in every system prompt instead.

---

## Running the demo

The pipeline requires a model that can:

1. Follow multi-step tool-call instructions
2. Provide JSON Schema parameter definitions
3. Write a direct return expression in the `code` field

**Recommended:** 8B+ parameter models. Smaller models (3B and below) tend to drop the
`code` field or write function-definition-style code.

### With Ollama (local)

```bash
OPENAI_BASE_URL=http://127.0.0.1:11434/v1/ \
OPENAI_API_KEY=ollama \
OPENAI_MODEL=granite4.1:8b \
.venv/bin/python main.py --turns 20 --log /tmp/robits-demo.log \
  --prompt "Team, we need a shared status signal.
SE: call tools.propose with tool_name=team.pulse,
description='Report caller status', and code exactly as:
return f\"{get_caller_name()}: {status}\"
Set parameters to: {\"type\":\"object\",\"properties\":{\"status\":{\"type\":\"string\"}},\"required\":[\"status\"]}
Do NOT write def or lambda in code.
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
`ToolRegistry.replace_definition`, which swaps the compiled function without touching
the tool's name or existing grants. System tools (`system_tool: true`) cannot be
replaced this way.

**Proposal store:** Proposals persist across sessions in `var/tool_proposals.json`
(configurable via `ROBITS_TOOL_PROPOSALS_FILE`). Registered tools are only
in-process; restart re-registers from `tools.yaml` only. If you want a custom tool
to survive restarts, add it to `tools.yaml` after a successful session.

**Date, time, and location context:** Every agent's system prompt includes
`current_datetime_local`, `current_date_local`, and `timezone` from `agent_runtime_context()`.
Set `ROBITS_TIMEZONE` (e.g., `America/New_York`) and `ROBITS_LOCATION` to populate the
location field. Agent-authored tool code can call `current_agent_context(employee_dict)`
to read these at call time.
