# Runtime Architecture

`main.py` contains the current runtime.

- `Role` stores role-specific prompt templates and calls chat completions through `interact`.
- `Human`, `Ops`, `HR`, `SoftwareEngineer`, and `Angel` define the initial organization roles.
- `System` stores preloaded escape-code functions and executes them with the current `employee_dict`.
- `preload.yaml` defines the default `create_role` escape code.
- `parse_escape_code` extracts the first valid JSON object or array from a model response.

Keep tests isolated from model services. Use live model checks only as smoke validation after deterministic tests pass.
