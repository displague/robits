# Runtime Architecture

`main.py` contains the current runtime.

- `Role` stores role-specific prompt templates and calls chat completions through `interact`.
- `Human`, `Ops`, `HR`, `SoftwareEngineer`, and `Angel` define the initial organization roles.
- `System` loads trusted tools and executes them with the current `employee_dict`.
- `tools.yaml` defines trusted tools such as `org.create_role`.
- `parse_tool_instruction` extracts the first valid JSON object or array from a model response.

## Current vs Target

Current runtime:

- Roles call Chat Completions.
- The runtime can still parse JSON tool instructions from assistant text for compatibility.
- Trusted tool definitions are repo-owned and loaded from `tools.yaml`.

Target runtime:

- Roles use OpenAI-compatible Responses-style interactions when available.
- Approved tools are exposed as function tools with JSON Schema metadata.
- The runtime handles zero or more function calls, executes trusted tools, returns tool outputs by call ID, and then asks the model for final text.
- Speaker selection moves from random routing to deterministic time-share scheduling with turn budgets and later parallel execution where state coordination is explicit.

Keep tests isolated from model services. Use live model checks only as smoke validation after deterministic tests pass.
