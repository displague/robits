# Agent Guidance

## Scope

This repository is a Python experiment for simulating AI organization roles. Keep changes focused on making the runtime reliable, testable, and easy for future agents to understand.

## Workflow

- Inspect the current branch, open pull requests, and issues before starting GitHub-facing work.
- Create or reference GitHub issues for durable defects and documentation work. Avoid opening issues for one-machine local setup details.
- Keep local model endpoints and model names in validation notes unless the project needs a generic OpenAI-compatible configuration change.
- Prefer small branches that close one or two tracked issues.

## Validation

- Run `python -m unittest` for runtime and parsing changes.
- Use `python main.py --prompt "<message>" --turns 1` for an optional live smoke test against the configured OpenAI-compatible endpoint.
- Do not require a live model service for unit tests.

## Repo-Local Skills

Repo-local AI skills live in `.github/skills/`. Start with `.github/skills/robits-runtime/SKILL.md` when changing role orchestration, trusted tool execution, JSON extraction, scheduling, or model endpoint configuration.
