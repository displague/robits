# AI-Powered Organization Simulation

Welcome to the AI-Powered Organization Simulation project. This project explores how an organization with AI-driven roles can communicate and work together.

## Table of Contents 📚

- [Getting Started](#getting-started)
- [Roles](#roles)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Simulation](#running-the-simulation)
- [Testing](#testing)
- [Contributing](#contributing)
- [Future Directions](#future-directions)

## Getting Started 🏁

The AI-Powered Organization Simulation is a Python-based project that simulates an organization with AI-driven roles, such as CEO, Ops, SE, and HR. The simulation demonstrates how these AI roles can communicate with each other and perform tasks specific to their roles.

## Roles 🧑‍💼

The organization has the following roles:

1. CEO (Human) - A human role responsible for making high-level decisions and setting the overall direction of the organization.
2. Ops (Operations) - A role responsible for overseeing the execution environment and operational success of the agent organization.
3. SE (Software Engineer) - A role responsible for designing, developing, and maintaining software applications, primarily proposing trusted tools when requested by other members of the organization.
4. HR (Human Resources) - A role responsible for managing AI resources and creating new roles within the organization.

## How It Works 🛠️

The simulation runs in a session, where organization members communicate through messages. A session owns a run ID, participant list, turn count, transcript entries, and the system tool handler. Undirected messages use deterministic round-robin scheduling; directed messages such as `HR, ...` still route to the named role.

The runtime can parse JSON tool instructions, load trusted tools from `tools.yaml`, and execute registered tools by name. Tools use namespaced identifiers such as `org.create_role`, with short aliases only where compatibility is useful.

HR lifecycle tools keep role creation compatible while adding explicit states:
`proposed`, `active`, `paused`, and `retired`. Current trusted tools create
active roles directly, pause active roles, and retire active or paused roles.
Lifecycle events record optional requester, approver, and reason fields.

The target runtime direction is OpenAI-compatible tool calling through modern Responses-style loops: approved tools are exposed with JSON Schema metadata, the model may request function calls, the runtime executes those calls, and tool outputs are returned to the model without relying on ad hoc text parsing.

See `docs/architecture-vision.md` for the staged runtime plan covering sessions, tools, SQLite-backed memory, memory digests, lifecycle, observability, TUI inspection, and local-model constraints.

## Memory

Robits includes a local SQLite memory substrate in `robits/memory/sqlite.py`.
It defines durable tables for sessions, agents, contacts, messages, thoughts,
todos, tool calls, and memory entries. FTS search indexes message, thought,
tool-result, and memory-entry content with filters for agent, session,
relationship type, conversation type, source, and date windows.

Generated local database files are ignored by git. Unit tests use temporary
SQLite databases. Local runs should place generated databases under `data/` or
`var/` unless the runtime is configured with another gitignored path.

Memory digests are compacted memory artifacts stored with prompt version, source
time range, retrieval filters, and ordered links back to raw source records so a
future run can expand or reanalyze the original material.

## Observability

Sessions emit a headless runtime event stream for session lifecycle, routing,
message, tool, and private thought events. Tests and future TUI code can
subscribe to the active stream without requiring an interactive terminal.
Runtime events can also be persisted to SQLite and replayed later, with
visibility fields separating public transcript events from private thoughts.

## Sandboxes

Agents can carry optional sandbox metadata. Default local runs keep sandboxing
disabled and execute through the current runtime. When enabled, metadata names a
backend plus a private per-agent workspace and a shared organization workspace.
The runtime validates that an explicit configured backend matches the metadata
before execution. The current implementation includes a fakeable runtime
abstraction for tests and future container or cluster backends; it does not
require containers for unit tests.

See `docs/kind-sandbox-runtime.md` for the target kind/Kubernetes sandbox model.
That design explains role-to-pod mapping, COO/operator permissions, capacity
constraints, persistence, and the current gap between sandbox metadata and a
real container backend.

## Installation

To install the required packages for this project, run:

```bash
pip install -r requirements.txt
```

## Configuration

The runtime uses the OpenAI Python client and can target OpenAI or an OpenAI-compatible local endpoint.

Relevant environment variables:

- `OPENAI_API_KEY`: API key. Local compatible servers may accept any non-empty value.
- `OPENAI_BASE_URL` or `OPENAI_API_BASE`: optional compatible endpoint URL.
- `ROBITS_MODEL` or `OPENAI_MODEL`: default chat model.
- `ROBITS_CHEAP_MODEL` and `ROBITS_COSTLY_MODEL`: optional per-role overrides.
- `ROBITS_MEMORY_DB`: optional SQLite memory database path for memory introspection tools.
- `ROBITS_MEMORY_MAX_DEPTH`: maximum recursive digest expansion depth (default `3`).
- `ROBITS_MEMORY_MAX_ROWS`: maximum rows returned by any memory tool (default `100`).
- `ROBITS_MEMORY_CACHE_THRESHOLD`: byte threshold above which memory tool results are offloaded to the agent's private workspace and a condensed snippet is returned (default `8192`).
- `ROBITS_PROVIDER_API`: model API path, `responses` by default or `chat_completions` for compatibility.
- `ROBITS_MAX_PARALLELISM`: maximum concurrent model calls, default `1`.
- `ROBITS_MAX_API_RETRIES`: maximum retry attempts for transient API failures, default `3`.
- `ROBITS_SEARCH_URL`: optional custom web search endpoint for `builtin.web_search`; falls back to the DuckDuckGo Instant Answers API.
- `ROBITS_DIGEST_INTERVAL`: turns between automatic episodic digest creation; default `0` (disabled).
- `ROBITS_DIGEST_CONTEXT_CHARS`: accumulated meaningful prompt/response characters that trigger automatic episodic digestion; default `0` (disabled).
- `ROBITS_DIGEST_ELAPSED_SECONDS`: elapsed seconds with meaningful activity that trigger automatic episodic digestion; default `0` (disabled).
- `ROBITS_IDENTITY_DIGEST_INTERVAL`: turns between automatic identity checkpoint digests; default `0` (disabled).
- `ROBITS_GOAL_DIGEST_INTERVAL`: turns between automatic short-term goal checkpoint digests; default `0` (disabled).

Automatic digest triggers skip silent/passive turns and tool-only runtime artifacts, so model-facing prompt context and operational noise do not bloat stored transcript memory. When enabled, digests are created for every agent and are searchable via `memory.search`.

## Tools

Trusted tools live in `tools.yaml`. Each entry includes a namespaced `name`, a JSON Schema `parameters` object, and trusted repo-owned code. Untrusted model output can request registered tools, but it cannot define new executable tools.
Agents can use trusted alarm tools to create, list, and cancel their own reminders,
and memory tools to inspect accessible SQLite memory. Memory digest creation and
re-digestion remain automatic system behavior rather than agent-callable tools.
Roles carry tool grants such as `agent.*` or `memory.search`; model providers
only expose tools allowed for the active role, and execution denies disallowed
tool calls. Operator roles can grant or revoke tool access, SE can propose
non-system tool changes, and system tools such as memory internals and HR role
management cannot be changed through SE proposals.

The `builtin.*` namespace provides equivalents of the OpenAI built-in tool types
as client-side function tools that work with any OpenAI-compatible endpoint:

| Tool | Description | Capability required |
|---|---|---|
| `builtin.web_search` | Search the web via DuckDuckGo or `ROBITS_SEARCH_URL` | — |
| `builtin.file_search` | Search text in an agent's private workspace files | — |
| `builtin.shell_run` | Run a shell command in the agent's workspace directory | `shell` |
| `builtin.tool_search` | Search the tool registry by name or description | — |
| `builtin.mcp_call` | Call a tool on an MCP server (not yet implemented) | `mcp` |
| `builtin.computer_use` | Computer-use actions (not implemented) | `computer` |
| `builtin.image_generation` | Generate images (not implemented) | — |

All `builtin.*` tools are grantable. None are in the default tool grants; operators
grant access explicitly via `tools.grant`.

Example execution payload:

```json
{"exec": "org.create_role", "args": {"role_name": "QA", "role_description": "Tests the organization"}}
```

## Running the Simulation

To run the AI-Powered Organization Simulation:

1. Run the Python file using a Python interpreter:
   ```bash
   python main.py
   ```

For non-interactive smoke checks, provide an initial prompt and turn limit:

```bash
python main.py --prompt "Ops, say hello to HR" --turns 1
```

To capture the same console transcript to a file while still watching the run:

```bash
python main.py --prompt "Ops, say hello to HR" --turns 1 --log run.log
```

## Testing

The focused runtime tests do not call an external model service:

```bash
python -m unittest
```

## Contributing 🤝

We welcome contributions to this project! Feel free to submit pull requests, report bugs, or suggest new features. To get started, check out the [Future Directions](#future-directions) section for some ideas on how you can contribute.

## Future Directions 🚀

There are many exciting ways you can improve and expand this project. Here are a few ideas to get you started:

1. Add more roles to the organization (e.g., marketing, sales, or finance roles) to explore new interactions between AI-driven roles.
2. Enhance the AI's ability to understand context and engage in more complex conversations.
3. Implement a graphical user interface (GUI) for a more immersive and user-friendly simulation experience.
4. Explore ways to use real-world data to drive the simulation and make it more engaging and relevant.
5. Experiment with different AI models or techniques to improve the performance and capabilities of the AI roles.
6. Extend the deterministic time-share scheduler toward parallel role execution once shared state coordination is explicit.

Have fun exploring the AI-Powered Organization Simulation! We can't wait to see what you come up with! 🎉💡
