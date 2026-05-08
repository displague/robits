# Kind Sandbox Runtime Model

Robits should eventually support an optional kind/Kubernetes runtime where each
agent can have an isolated execution environment while still sharing approved
organization state. This document is a target design, not a description of the
current implementation.

## Current State

The current sandbox support is a Python boundary:

- agents carry optional `SandboxMetadata`
- default agents have sandboxing disabled
- `SandboxRuntime` validates that enabled metadata matches a configured backend
- tests use `FakeSandboxBackend` to verify request and result shapes

There is no current kind, Kubernetes, container, or local-process execution
backend. Enabling sandbox metadata without a backend returns an explicit error.
The existing code is useful because it gives future backends a narrow contract:
receive an agent name, trusted tool name, arguments, private workspace, shared
organization workspace, and policy, then return a structured execution result.

## Target Cluster Shape

A local kind cluster can host one Robits organization namespace. The namespace
contains:

- one controller or runtime service that owns scheduling, routing, memory, and
  tool dispatch
- one pod per active agent sandbox when sandboxing is enabled
- an optional shared organization volume mounted read/write where policy permits
- a durable state service or volume for session, memory, lifecycle, and event
  records
- Kubernetes RBAC scoped to the namespace

The human-facing runtime can still run outside the cluster during development,
but the long-term design should allow the runtime service to run inside the
cluster as the namespace controller.

## Role to Pod Mapping

Roles should not map directly to Kubernetes workloads. Agents should.

A role is a policy and prompt definition. An agent is a concrete identity with
lifecycle state, memory, contacts, tools, and sandbox metadata. Therefore:

- each active sandboxed agent maps to one pod
- paused agents keep durable state but may scale to zero pods
- retired agents keep historical state and no running pod
- multiple agents may share the same role type while still having distinct pods
  and state

The pod name should derive from an immutable agent ID, not from a mutable display
name. Labels should expose role, lifecycle state, organization ID, and agent ID
so the runtime can query capacity and health without parsing names.

## Organization Size and Capacity

Organization size can be constrained by declared pod capacity rather than only
by an in-memory HR limit. This creates an observable resource boundary:

- HR proposes or approves agent lifecycle changes
- the COO/operator checks namespace capacity and runtime health
- the runtime admits new active agents only when capacity policy allows another
  pod or when an existing paused pod can be resumed
- Kubernetes limits and requests make CPU, memory, and storage pressure visible

This should complement, not replace, HR policy. HR owns whether an agent should
exist. The COO owns whether the environment can safely run or migrate that
agent.

## COO and Operator Permissions

The COO/operator role can be represented by trusted runtime actions backed by a
namespace-scoped Kubernetes service account. The model should not expose raw
cluster credentials to model text. Instead, trusted tools can provide bounded
verbs such as:

- inspect agent pod health and recent events
- cordon an agent from new work
- restart an agent pod after HR lifecycle coordination
- scale paused agents to zero
- resume an active or paused agent when capacity allows
- snapshot state markers before a risky change
- roll back to a recorded runtime or configuration version

The service account should be namespace-scoped. Cluster-wide permissions should
not be required for normal Robits operation.

## Storage and Reemergence

Pods are replaceable. Agent lives are not. Reemergence depends on externalized
state:

- SQLite can remain the first durable substrate for local development.
- A shared read/write organization volume can hold approved project artifacts,
  tool outputs, logs, and other organization-level files.
- A private per-agent volume can hold workspace artifacts that are not part of
  shared organization state.
- Longer term, session state, memory, thoughts, todos, lifecycle state, and
  tool-call history should be stored in a database service or persistent volume
  that survives pod restarts.

The database should be the source of truth for identity and memory. Volumes are
for workspaces and artifacts. A restarted agent pod should reconstruct its
working context by reading lifecycle state, recent sessions, memory digests, and
retrieved raw records from durable storage.

## Safe Environment Changes

Self-modification should be a governed environment workflow:

1. SE proposes a code, tool, configuration, or runtime change.
2. COO validates operational timing, dependency risk, and namespace impact.
3. HR checks affected agents, active work, memory durability, and commitments.
4. The runtime records state markers for code, tools, database, pods, and active
   sessions.
5. COO applies the change through a trusted executor.
6. Affected pods are restarted or migrated only after state is durable.
7. Runtime health and smoke checks decide whether to continue or roll back.

The kind backend should make these steps observable through runtime events and
Kubernetes events, but Kubernetes should remain an implementation detail behind
trusted tools.

## Implementation Phases

1. Keep the current fakeable `SandboxRuntime` boundary and add no container
   dependency to unit tests.
2. Add a local-process backend for bounded development smoke tests.
3. Add a kind backend that can create, inspect, restart, and remove one
   namespace-scoped pod per active sandboxed agent.
4. Add persistent state integration so pod restarts preserve agent identity,
   memory, lifecycle state, sessions, thoughts, todos, and tool records.
5. Add capacity policy that binds organization active size to pod quotas,
   resource requests, and HR/COO approval.
6. Add TUI observability over runtime events, memory records, and pod health.

Until phase 3 exists, Robits should describe sandboxing as metadata and backend
contract support only.
