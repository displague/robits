# Deploying robits

Headless mode makes robits a long-lived service: it sleeps until something is due, takes a turn, and
sleeps again. This page covers the container image and the single-pod Kubernetes example in
[`deploy/kubernetes/`](../deploy/kubernetes/). Keep your real roster, endpoint and credentials in your
own repository and mount them in; this repository stays generic.

## Container image

```bash
docker build -t robits .                                  # headless core only
docker build --build-arg EXTRAS="sqlite-vec" -t robits .  # plus vector memory search
docker run --read-only --tmpfs /tmp -v robits-data:/data \
  -e OPENAI_BASE_URL=http://host.docker.internal:11434/v1/ -e ROBITS_MODEL=granite4.1:3b robits
```

Tagged releases publish `ghcr.io/displague/robits:<version>` for amd64 and arm64.

The image:

- runs `main.py --headless` as uid 10001 and works with a read-only root filesystem;
- installs only `requirements-core.txt`, so the Textual viewer and sqlite-vec are left out unless you add them;
- keeps all state on `/data`: the memory DB, agent workspaces, tool proposals, wait state and the heartbeat file;
- defaults `ROBITS_PROVIDER_API` to `chat`, the API most local servers implement;
- writes JSONL events to stderr and the transcript to stdout.

## Configuration

Every config file can come from a mounted path:

| File | Setting | Notes |
|---|---|---|
| Persona roster, including per-persona `tools`, `tool_budget`, `max_tool_rounds` | `ROBITS_PERSONAS_FILE` | ConfigMap |
| Tool catalogue | `ROBITS_TOOLS_FILE` | Defaults to the bundled `tools.yaml` |
| MCP servers | `ROBITS_MCP_CONFIG` | Use a Secret: entries can carry auth headers |

### Model endpoint

Point `OPENAI_BASE_URL` at any OpenAI-compatible server and set `ROBITS_MODEL`.

- **Chat Completions (`ROBITS_PROVIDER_API=chat`)** works with Ollama, LM Studio, llama.cpp's server, vLLM and RKLLama.
- **Responses (`ROBITS_PROVIDER_API=responses`)** needs a server that implements `/v1/responses`.
  - For any base URL other than OpenAI's, robits replays the whole conversation on each tool round, because servers such as Ollama ignore `previous_response_id`.
  - Set `ROBITS_RESPONSES_STATEFUL=true` only for a server that keeps that state.
- **Small models:** give each persona a short `tools` list and a low `max_tool_rounds` (see the README's tool allowlists). Fewer tool schemas per round helps them more than anything else.

### Headless and idle settings

| Setting | Default | Use |
|---|---|---|
| `ROBITS_MAX_TURNS_PER_HOUR` | unlimited | Cap model spend; the session sleeps when the budget is used |
| `ROBITS_MIN_TURN_INTERVAL_SECONDS` | 0 | Space out one persona's turns |
| `ROBITS_IDLE_MAX_SLEEP_SECONDS` | 300 | Longest single sleep; keep it below the liveness `--max-age` |
| `ROBITS_MAX_CONSECUTIVE_ERRORS` | 5 | Failed turns in a row before exit 1, so the pod restarts |
| `ROBITS_BREAK_SCHEDULE`, `ROBITS_CLOCK_STATE`, `ROBITS_TIMEZONE` | | Off-clock personas don't take turns |

A first message can be seeded with `--prompt "..."` (container args). Without one and with nothing
scheduled, the pod idles and makes no model calls.

## Kubernetes example

```bash
kubectl create namespace robits
kubectl -n robits create secret generic robits-secrets \
  --from-literal=OPENAI_API_KEY=not-needed --from-file=mcp.json=./mcp.json
kubectl apply -k deploy/kubernetes
```

What the example contains:

- **Deployment:** one replica with `Recreate`, so two pods never share the SQLite DB. It runs non-root with a read-only root filesystem and all capabilities dropped, and doesn't mount a service-account token.
- **ConfigMap** (generated from `config/` plus a few settings) mounted at `/config`. Editing the roster rolls the pod, because the ConfigMap name changes with its content.
- **PersistentVolumeClaim** for `/data`.
- **Liveness probe:** `python -m robits.health --max-age 900` checks that the headless loop touched `ROBITS_HEARTBEAT_FILE` recently. It never calls the model. Raise `--max-age` above the idle cap plus your slowest model call.
- **NetworkPolicy:** no ingress, since robits serves nothing. Egress is allowed only to DNS, the model endpoint and the MCP servers you list. Adjust the selectors to match your cluster.
- **Termination:** SIGTERM finishes the current turn and exits 0. `terminationGracePeriodSeconds` should cover one model call.

## Per-persona credentials

Today all personas share the process's credentials: the model API key and the MCP headers in
`ROBITS_MCP_CONFIG`. Separate OAuth identities per persona are planned in #138. Until then, give each
MCP server the narrowest token the whole roster needs.

## Backups

Back up the data volume:

- **`memory.db`:** memories, digests, tasks and tool-call history. Copy it while the pod is stopped, or take a live copy with SQLite's online backup (`sqlite3.Connection.backup` in Python).
- **`agents/`:** per-agent workspaces.
- **`tool_proposals.json`:** pending tool proposals.

The heartbeat and wait-state files can be recreated. Config comes from your own repository.
