---
name: robits-runtime
description: Work on the Robits Python organization-simulation runtime, including role orchestration, escape-code preload/execution, JSON extraction from model responses, OpenAI-compatible model configuration, and non-interactive smoke validation.
---

# Robits Runtime

## Process

1. Inspect `main.py`, `preload.yaml`, and `tests/test_runtime.py` before changing runtime behavior.
2. Preserve support for OpenAI-compatible endpoints through environment variables; keep machine-specific endpoint and model names out of committed docs unless expressed generically.
3. Treat escape-code handling as a high-risk path because model output is parsed and executed.
4. Prefer focused tests around parsing, preload, and execution before relying on a live model smoke test.

## Validation

- Run `python -m unittest` after runtime changes.
- For optional live smoke validation, configure an OpenAI-compatible endpoint through environment variables and run `python main.py --prompt "<message>" --turns 1`.
- Do not make unit tests depend on network access or a local model server.

## References

- Read `resources/runtime-architecture.md` for the current runtime shape and known boundaries.
- Use `assets/create-role-message.json` as a minimal escape-code execution payload.
