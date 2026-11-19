# Pod-per-agent execution mode (design)

Status: design for issue #70. Nothing here is implemented. It builds on
[`kind-sandbox-runtime.md`](kind-sandbox-runtime.md), which covers one pod per *tool sandbox*, and on the
single-pod deployment in [`deploy.md`](deploy.md). This page is about running each *agent* as its own
worker, with scoped storage, and about what has to be decided before code commits to a storage layout.

## What exists today

- One process runs every persona, one session per data volume. The Kubernetes example uses
  `replicas: 1` and `strategy: Recreate` so two pods never open the same SQLite file.
- State lives under `/data`: the memory database (`ROBITS_MEMORY_DB`), agent workspaces
  (`ROBITS_AGENT_WORKSPACE_ROOT`), tool proposals, wait state and a heartbeat file
  (`ROBITS_HEARTBEAT_FILE`, read by the liveness probe).
- Storage is already behind `MemoryStoreProtocol` and `AsyncMemoryStoreProtocol`
  (`robits/memory/protocols.py`): sessions, agents, contacts, channels, messages, thoughts, todos, tool
  calls, memory entries and digests, and runtime events. Every message, thought and event carries a
  `visibility` (`public` or `private`), and channels carry visibility, social distance and memory
  policy.
- Events are emitted into the in-process `RuntimeEventStream` (`emit`, `subscribe`, `events(visibility)`,
  with a per-stream counter as `sequence`) and persisted by `append_runtime_event` into `runtime_events`,
  where `sequence` is nullable and not unique and `event_id` is the table's autoincrement key. The TUI does
  not subscribe to the stream: `RobitsDbReader` opens the database file read-only and polls
  `runtime_events` for the selected session, ordered by `COALESCE(sequence, event_id)`, remembering what it
  has shown by `event_id`. It also filters `private` events itself, per its own policy.
- `append_message` and `append_runtime_event` have no idempotency key: a replayed call inserts again.
- Scheduling is in-process: `RoundRobinScheduler` and the headless due-time loop.
- SQLite is opened with WAL and a busy timeout (`async_sqlite.open`), which is right for one writer
  process and wrong for several machines (see below).

## Target topology

```
                 +---------------------- namespace: org ----------------------+
  TUI / CLI ---> |  org-store  (1 pod, sole writer of org SQLite, PVC "org")    |
  (observer)     |    ^  ^  ^        HTTP: store API + event log (append/follow) |
                 |    |  |  |                                                   |
                 |  agent-alex  agent-sam  agent-... (one pod each)             |
                 |   PVC alex    PVC sam                                        |
                 |   + mounts only the group/work scopes the agent belongs to   |
                 +-------------------------------------------------------------+
                          each agent --> model endpoint (egress only)
```

- **org-store**: one small service owning the organization state: org chat channels, the shared event
  log, shared memory digests, the agent registry and contacts. It is the only process that opens the
  org SQLite file. Single writer by construction, so no shared-volume locking questions.
- **agent worker**: one pod per active agent, running the existing headless loop for exactly one agent
  identity. It owns that agent's personal volume and talks to the org-store over the network.
- **Workload kind**: one single-replica `StatefulSet` per agent. A stable pod name and volume claim
  derive from the immutable agent ID (not the display name), as `kind-sandbox-runtime.md` already
  requires. Paused agents scale to zero and keep the claim; retired agents keep the claim and no pod.
  Short private subtasks stay inside the worker (see #132); a `Job` per task is not needed yet.
- **Modes**, selected by configuration and sharing one set of interfaces:
  1. local single-process (today);
  2. local async, many agents in one process, each with its own personal store and the org store as a
     separate object (the testable middle step, no cluster needed);
  3. pods: the same code, with the org store and event bus reached over HTTP.

## Scopes and volumes

| Scope | Holds | Volume | Access |
|---|---|---|---|
| Personal | the agent's private thoughts, todos, tool calls, private subtask checkpoints, workspace, OAuth credentials | one `ReadWriteOnce` claim per agent, SQLite + files | that agent's pod only |
| Org | public org chat, shared event log, org memory digests, registry, shared artifacts | `ReadWriteOnce` claim owned by the org-store pod; artifacts via the service, not a shared mount | read and append through the org-store API |
| Family / group | conversation context for a named group (not the whole org) | its own store instance (same service image, separate claim) per group | only agents listed as members |
| Work / team | a project's or peers' scoped context | same as group | only agents listed as members |

Rules:

- A worker mounts its own personal claim and nothing else. Group and work scopes are reached through
  that scope's store service, never by mounting another agent's or another scope's claim.
- Private thoughts and anything with `private` visibility are written to the personal store only. They
  are not copied to the org store, so a bug in the org service cannot leak them.
- Shared artifacts go through the org-store API (with the caller's identity), not a `ReadWriteMany`
  volume, so there is one place to enforce access and one writer.

## SQLite on shared volumes

Open question 1 from the issue, answered:

- SQLite with WAL needs shared memory and file locking that network file systems do not provide
  reliably. A `ReadWriteMany` volume (NFS, many CSI drivers) with several pods writing one SQLite file is
  not safe, and a local-path claim only works on one node. Reject it.
- **Personal state**: one SQLite file per agent on that agent's `ReadWriteOnce` claim. One writer (the
  worker), so the current store works unchanged.
- **Org state**: one writer service (org-store) in front of one SQLite file. Clients talk to it with the
  store protocol over HTTP. Throughput is not a concern at this scale, and the single writer gives
  ordered event sequence numbers for free.
- **Later**, if the org outgrows one writer: swap the org-store's backend for PostgreSQL behind the same
  API. Agents do not change. Do not design that now.
- **Event log**: an append API plus a cursor-based follow (long poll or server-sent events), backed by
  the org store's table. A shared JSONL file is rejected: append atomicity across pods on a network
  file system is not guaranteed. A message broker (NATS, Redis) is an option if fan-out becomes the
  bottleneck; it adds an operational dependency, so it is deferred.

## Interfaces

Most of these exist as protocols or in-process classes. The change is to name the seams and add the
remote implementations:

| Name | Today | Pod mode |
|---|---|---|
| `MemoryStore` / `MessageStore` | `MemoryStoreProtocol`, SQLite | a composite that routes by scope: personal store (local SQLite) and org/group store (HTTP client of the org-store, same method names) |
| `EventBus` (writer side) | `RuntimeEventStream` (emit, subscribe), whose subscriber persists events with `append_runtime_event_object` | unchanged inside a worker; the persisting subscriber appends to the personal store and ships to the org-store (see "Ordering and idempotency") |
| `EventSource` (reader side, new) | none: the TUI reads SQLite directly (`RobitsDbReader.get_runtime_events`) | one read interface, `events(after_cursor, filters) -> (events, next_cursor)`; a SQLite implementation (today's polling, cursor = `event_id`) and a remote one (`GET /events?after=` on the org-store) |
| `AgentIdentity` | agent ID, role and metadata rows | adds org ID and the list of scopes the agent may use; derived from the pod's identity (below), never from a request field |
| `ConversationScope` | channel type and visibility on the channel row | maps a channel to the store that owns it (personal, org, group, work) |
| `AgentRuntime` | the headless loop and `RoundRobinScheduler` | a worker entry point that runs the loop for one `AgentIdentity`; it takes its turns when something addressed to it is due, from the follow stream, instead of from a global round-robin |

## Ordering and idempotency

The current APIs do not give either, so this is a contract the org-store adds (additive; local mode keeps
today's behaviour):

- **Who assigns the org order.** The org-store, at commit. The cursor of the org event log is the org
  database's own autoincrement `event_id`; it is monotonic and may have gaps (a rolled-back insert), which
  readers must tolerate. The worker-supplied `sequence` stays what it is today, a source-local counter, so
  order *within one agent and session* is by that sequence and order *across agents* is by org `event_id`
  (commit order). There is no clock-based ordering.
- **Conflict key.** `append_message` and `append_runtime_event` gain an optional `source_key`, and the org
  tables gain nullable `source_agent` and `source_key` columns with a unique index on the pair
  (`WHERE source_key IS NOT NULL`, so existing callers that pass none are unaffected). For events the key
  is `<session>:<local sequence>`; for messages it is a client message ID the worker generates and saves in
  its personal store *before* sending (a counter or UUID), never derived from content.
- **Duplicate handling.** An append with a key that exists returns the stored row's `event_id` and a
  `duplicate: true` flag, with status 200 (idempotent). If the key exists but a hash of the payload differs,
  it answers 409 (a reused key), and nothing is stored.
- **Replay after a restart.** The personal store is the source of truth: the worker records a `shipped_through`
  marker per stream, and after a restart re-sends everything past it. The keys make a re-send harmless, so
  delivery is at least once and storage is exactly once.
- **Cursor semantics.** `GET /events?after=<event_id>&limit=N` returns visible events with `event_id > after`
  in ascending order and `next_cursor` = the last returned `event_id` (or the same `after` when empty).
  Visibility filtering is server-side, so a filtered row leaves a gap the client never sees. A client keeps
  the last cursor (and, like the TUI today, may also dedupe by `event_id`); on a reconnect it resumes from
  that cursor.
- **Turn order across pods.** A worker takes its turn when a message addressed to it, or on a channel it
  belongs to, appears past its cursor. Two agents answering at once is ordinary: both appends commit, in
  some order, and both see the other's on their next read. The deterministic round-robin stays the local,
  single-process mode used by tests.

## Visibility across pods

Enforce at the store boundary, not inside the agent:

- The org-store authenticates each call with a projected service-account token bound to the
  org-store's audience. The token's subject maps to exactly one agent ID. The agent ID is never taken
  from the request body.
- Reads are filtered by channel membership and visibility on the server: an agent sees public org
  channels, channels it is a participant of, and groups it belongs to. `private` rows are not served to
  anyone but the owning agent, and the org store does not hold them in the first place.
- Writes are checked the same way (an agent writes as itself, into scopes it belongs to).
- `NetworkPolicy`: workers may reach the org-store, their group/work stores and the model endpoint; they
  may not reach each other. The org-store accepts only workers and the observer.
- No worker gets Kubernetes API access (`automountServiceAccountToken: false` stays, with a separately
  projected audience-bound token for the store).
- Credentials (MCP servers, OAuth) live in the personal scope or a per-agent Secret. Nothing credential-like goes to the org store.

## Observing a pod-backed runtime

- Today the TUI opens the database file read-only and polls it (see "What exists today"). That cannot work
  across pods: each personal file is on a volume only its agent mounts, and the org file belongs to the
  org-store. The bridge is the `EventSource` read interface above: the TUI's `RobitsDbReader` is wrapped
  as its SQLite implementation (same query, cursor = `event_id`, so no behaviour change), and the remote
  implementation calls the org-store's `GET /events`. `RuntimeEventStream` stays the writer-side bus inside
  a process; the TUI is never a subscriber of it, in either mode.
- The `private` filter moves to the server for the remote source (the TUI's own `public-only` policy stays
  as a display option on top).
- No Kubernetes API access is needed for events. Discovery uses the registry, not the cluster: each worker
  upserts its agent row and refreshes a heartbeat in the org store (the existing agent lifecycle state plus
  a last-seen time); the TUI lists agents from there. Kubernetes labels (`org`, `agent-id`, `role`,
  `lifecycle`) are for operators and for capacity queries, as in `kind-sandbox-runtime.md`.
- Liveness stays the existing heartbeat-file probe inside each worker.

## Minimal kind sketch (illustrative; the environment variables are proposed, not implemented)

One namespace, one org-store with its claim and Service, one worker with its identity token, and the
policies that select them by label. Image names and the entry points are placeholders. The YAML parses
and its Service, NetworkPolicy and volume references were checked against the pod labels and volume
names; it has not been applied to a cluster, so validate it there first.

```yaml
apiVersion: v1
kind: Namespace
metadata: {name: robits-org}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: org, namespace: robits-org}
spec:
  accessModes: [ReadWriteOnce]
  resources: {requests: {storage: 1Gi}}
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: org-store, namespace: robits-org, labels: {app: org-store}}
spec:
  replicas: 1
  strategy: {type: Recreate}              # one writer on the org SQLite file
  selector: {matchLabels: {app: org-store}}
  template:
    metadata: {labels: {app: org-store, robits/role: org-store}}
    spec:
      automountServiceAccountToken: true  # the store validates worker tokens (TokenReview), so it needs the API
      securityContext: {runAsNonRoot: true, runAsUser: 10001, fsGroup: 10001, seccompProfile: {type: RuntimeDefault}}
      containers:
        - name: org-store
          image: robits                    # proposed entry point: python -m robits.org_store
          ports: [{name: http, containerPort: 8080}]
          env:
            - {name: ROBITS_MEMORY_DB, value: /data/org.db}
            - {name: ROBITS_ORG_STORE_AUDIENCE, value: org-store}      # proposed
          securityContext: {allowPrivilegeEscalation: false, readOnlyRootFilesystem: true, capabilities: {drop: [ALL]}}
          resources: {requests: {cpu: 50m, memory: 128Mi}, limits: {memory: 512Mi}}
          volumeMounts: [{name: org, mountPath: /data}, {name: tmp, mountPath: /tmp}]
      volumes:
        - {name: org, persistentVolumeClaim: {claimName: org}}
        - {name: tmp, emptyDir: {sizeLimit: 64Mi}}
---
apiVersion: v1
kind: Service
metadata: {name: org-store, namespace: robits-org}
spec:
  selector: {app: org-store}
  ports: [{name: http, port: 80, targetPort: http}]
---
apiVersion: v1
kind: ServiceAccount
metadata: {name: agent-alex, namespace: robits-org}
automountServiceAccountToken: false
---
apiVersion: v1
kind: Service                              # the StatefulSet's governing (headless) service
metadata: {name: agent-alex, namespace: robits-org}
spec:
  clusterIP: None
  selector: {robits/agent-id: alex}
  ports: [{name: none, port: 1}]
---
apiVersion: apps/v1
kind: StatefulSet
metadata: {name: agent-alex, namespace: robits-org, labels: {robits/agent-id: alex, robits/role: worker}}
spec:
  serviceName: agent-alex
  replicas: 1
  selector: {matchLabels: {robits/agent-id: alex}}
  template:
    metadata: {labels: {robits/agent-id: alex, robits/role: worker}}
    spec:
      serviceAccountName: agent-alex
      automountServiceAccountToken: false  # no API access; the store token below is the only credential
      securityContext: {runAsNonRoot: true, runAsUser: 10001, fsGroup: 10001, seccompProfile: {type: RuntimeDefault}}
      containers:
        - name: agent
          image: robits
          env:
            - {name: ROBITS_AGENT_ID, value: alex}                          # proposed
            - {name: ROBITS_ORG_STORE_URL, value: "http://org-store"}       # proposed
            - {name: ROBITS_ORG_STORE_TOKEN_FILE, value: /var/run/org-store/token}   # proposed
            - {name: ROBITS_MEMORY_DB, value: /data/alex.db}                # the personal store
          securityContext: {allowPrivilegeEscalation: false, readOnlyRootFilesystem: true, capabilities: {drop: [ALL]}}
          resources: {requests: {cpu: 50m, memory: 128Mi}, limits: {memory: 512Mi}}
          volumeMounts:
            - {name: personal, mountPath: /data}
            - {name: org-store-token, mountPath: /var/run/org-store, readOnly: true}
            - {name: tmp, mountPath: /tmp}
      volumes:
        - name: org-store-token            # audience-bound, short-lived, refreshed by the kubelet
          projected:
            sources:
              - serviceAccountToken: {audience: org-store, expirationSeconds: 3600, path: token}
        - {name: tmp, emptyDir: {sizeLimit: 64Mi}}
  volumeClaimTemplates:
    - metadata: {name: personal}
      spec: {accessModes: [ReadWriteOnce], resources: {requests: {storage: 1Gi}}}
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: workers-egress, namespace: robits-org}
spec:
  podSelector: {matchLabels: {robits/role: worker}}
  policyTypes: [Egress]
  egress:
    - to: [{podSelector: {matchLabels: {robits/role: org-store}}}]
      ports: [{port: 8080, protocol: TCP}]
    - ports: [{port: 53, protocol: UDP}, {port: 53, protocol: TCP}]    # DNS
    - ports: [{port: 443, protocol: TCP}]                               # the model endpoint
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: org-store-ingress, namespace: robits-org}
spec:
  podSelector: {matchLabels: {robits/role: org-store}}
  policyTypes: [Ingress]
  ingress:
    - from: [{podSelector: {matchLabels: {robits/role: worker}}}]       # add the observer's label here
      ports: [{port: 8080, protocol: TCP}]
```

The store checks a worker's token with the Kubernetes `TokenReview` API (audience `org-store`) and maps the
service account to one agent ID; that is why the org-store pod, and only that pod, keeps API access. A
complete example would add group stores, the observer, and probes (`python -m robits.health`, as in
`deploy/kubernetes/`). Never run two org-store pods on one claim.

## Phasing

Each step is testable without a cluster, using fake transports and temporary SQLite files.

1. Name the interfaces (`EventSource` over today's SQLite polling, scope routing in a composite store,
   `AgentIdentity`) with the current in-process implementations. No behaviour change.
2. Local async mode: several agents in one process, each with its own personal store and a separate
   org store object. This proves scope routing and visibility filtering.
3. The org-store service and its HTTP client (same method names as the protocol), with the
   authorization rules above and the `source_key` / org `event_id` contract from "Ordering and
   idempotency" (additive columns and a partial unique index); tests use a loopback server, duplicate
   and conflicting keys, and a replay from a `shipped_through` marker.
4. The remote `EventSource` that follows the org log from a cursor, with resume after a reconnect.
5. The worker entry point and registry heartbeat; the TUI over the remote `EventSource`.
6. Manifests under `deploy/` and a kind smoke test.

## Decisions for the maintainer

- Whether org-store is a new entry point in this repository or a separate project.
- Whether group and work scopes are separate store pods per group (isolated, more pods) or scopes
  inside the org-store (fewer pods, enforcement by rows). The sketch assumes separate pods for groups
  with sensitive context; rows are fine for ordinary work scopes.
- Whether to keep local round-robin as the only deterministic mode for tests and make pod mode
  explicitly eventually consistent.
