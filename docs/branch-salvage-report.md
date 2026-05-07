# Branch Salvage Report

Date: 2026-05-07

This report records the review of the remaining experimental branches after PR
#4 and PR #11 were merged. None of the branches should be merged wholesale:
each one is based on an older runtime shape and would remove current tests,
agent guidance, or the repo-local skill files.

## Summary

| Source | Status | Recommendation |
| --- | --- | --- |
| PR #1 / `origin/grok3` | Closed as superseded | Drop. PR #4 and PR #11 preserved the useful runtime fixes with tests and current terminology. |
| `origin/tools` | Partially salvaged | Keep the ideas: role modularity, a tool registry, tool metadata, and OpenAI-compatible tool-call direction. Do not merge the branch. |
| `origin/tools_fmt` | Salvaged | Keep the `tools.yaml` direction and tool metadata shape. PR #11 replaced it with tested current code. |
| `origin/yaml` | Not salvaged | Drop for now. YAML message parsing is not needed for the current JSON/tool-call path and the branch is stale. |

## Kept

- Trusted tool definitions moved to `tools.yaml`.
- Tools use namespaced identifiers such as `org.create_role`.
- Tool metadata can be exported for OpenAI-compatible Responses function tools
  and chat-completions function tools.
- Model-facing tool names such as `org__create_role` resolve back to canonical
  registered tools.
- Runtime tool behavior is covered by deterministic `unittest` coverage.

## Deferred

- Splitting role classes into separate modules remains a useful future cleanup,
  but it should be done after the session/runtime boundaries in issue #8 so the
  refactor has stable seams and test coverage.
- Broader role-owned tools and per-agent tool namespaces belong with the memory
  and lifecycle work in issues #6 and #9.
- YAML-oriented conversation payloads can be reconsidered later only if they
  serve a concrete session/transcript or human-editing workflow.

## Drop

- Do not merge `origin/tools`, `origin/tools_fmt`, `origin/yaml`, or
  `origin/grok3` into `main`.
- After this report lands, those remote branches can be deleted as superseded
  paths unless they are intentionally kept as historical references.
