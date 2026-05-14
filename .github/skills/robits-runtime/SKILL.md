---
name: robits-runtime
description: Work on the Robits Python organization-simulation runtime, including role orchestration, trusted tool loading/execution, JSON extraction from model responses, OpenAI-compatible Responses or chat model configuration, scheduling, and non-interactive smoke validation.
---

# Robits Runtime

## Process

1. Inspect `main.py`, `tools.yaml`, and `tests/test_runtime.py` before changing runtime behavior.
2. Preserve support for OpenAI-compatible endpoints through environment variables; keep machine-specific endpoint and model names out of committed docs unless expressed generically.
3. Treat tool execution as a high-risk path because model output can request side effects.
4. Keep trusted tool definition loading separate from untrusted model output.
5. Preserve `org.create_role` compatibility while keeping HR lifecycle state transitions explicit.
6. Keep observability headless-first; tests should assert emitted events without requiring a terminal UI.
7. Keep sandbox behavior optional and fakeable; unit tests must not require containers or a local cluster.
8. Prefer focused tests around parsing, tool loading, lifecycle, observability, sandbox metadata, and execution before relying on a live model smoke test.
9. For Responses API work, test function-call routing with fake response items before using a live endpoint.
   - **LM Studio**: stateful Responses API — `previous_response_id` is honoured; use the default `ROBITS_PROVIDER_API=responses`.
   - **Ollama**: non-stateful Responses API (since v0.13.3) — `previous_response_id` is silently ignored, breaking the within-turn tool-call continuation in `ResponsesProvider`. Use `ROBITS_PROVIDER_API=chat` with Ollama (tracked in #102).
10. For scheduling work, keep `Session` and `RoundRobinScheduler` unit-testable with fake roles before adding parallel execution.

## Validation

- Run `python -m unittest` after runtime changes.
- For optional live smoke validation, configure an OpenAI-compatible endpoint through environment variables and run `python main.py --prompt "<message>" --turns 1`.
- Do not make unit tests depend on network access or a local model server.

## References

- Read `resources/runtime-architecture.md` for the current runtime shape and known boundaries.
- Read `../../../docs/architecture-vision.md` before changing long-term runtime boundaries, memory, lifecycle, observability, or local-model strategy.
- Use `assets/create-role-message.json` as a minimal tool execution payload.
