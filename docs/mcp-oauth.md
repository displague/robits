# OAuth for MCP servers

Some HTTP MCP servers need OAuth 2.1. An entry in `ROBITS_MCP_CONFIG` can carry an `auth` block. Entries without one keep using static `headers`.

```json
{
  "tracker": {
    "url": "https://tracker.example/mcp",
    "auth": {
      "type": "oauth",
      "client_id": "https://example.org/robits/client.json",
      "scopes": ["mcp"],
      "identity": "service:team"
    }
  }
}
```

| Key | Meaning |
|---|---|
| `type` | `oauth` |
| `client_id` | URL of a [Client ID Metadata Document](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/) that describes robits. No registration is needed. See [`oauth-client-metadata.example.json`](oauth-client-metadata.example.json). |
| `registration: "dynamic"` | Use instead of `client_id` for servers that support only Dynamic Client Registration (RFC 7591). Robits registers once per server and reuses the client. |
| `scopes` | Scopes to request. Optional. |
| `identity` | Whose tokens calls carry: `persona` (each calling persona's own), `service:<name>` (one shared identity), or `default`. See [Identities](#identities). |
| `service_tools` | With `service:<name>` only: the remote tools that may run under it. Others on that server aren't offered and are refused. |

## Identities

Servers usually scope access by the signed-in user. Personas that share one account would all act as that account. They would share one vote and one audit trail, and none of them could be revoked separately. Robits offers three identity modes:

- **`"identity": "persona"`.** Each persona logs in as its own user, and its calls carry its own token. A subtask acts as the persona that spawned it. A persona that hasn't logged in skips that server: its tools aren't offered, and a call returns the re-login error and emits `auth.relogin_required`. Other personas keep working.
- **`"identity": "service:<name>"`.** One shared identity for an organisational unit, used by every persona that calls the server. Pair it with `service_tools`, so that only those remote tools run under the shared account.
- **`default`.** One identity with no particular meaning, which is the part 1 behaviour.

To use both on one server, give it two entries with the same URL. Tool names include the server key, so a persona's `tools:` list picks one or the other: `mcp:tracker.update_task` acts as the persona, `mcp:desk.post_update` acts as the service.

```json
{
  "tracker": {"url": "https://tracker.example/mcp",
              "auth": {"type": "oauth", "client_id": "https://example.org/robits/client.json", "identity": "persona"}},
  "desk":    {"url": "https://tracker.example/mcp",
              "auth": {"type": "oauth", "client_id": "https://example.org/robits/client.json",
                       "identity": "service:desk", "service_tools": ["post_update"]}}
}
```

**Discovery** (`tools/list`) runs once per identity, because two users may be offered different tools. A call to a tool the caller's identity wasn't offered is refused before it reaches the server.

**Logging.** Every MCP call, apart from `tools/list` paging, writes an `mcp_call` entry to the JSONL log and emits a private `mcp.call` event. Each records the server, method, tool, the identity used (`persona:<name>`, `service:<name>`, `default`, or `static` for header-authenticated servers) and the calling agent. Tokens are never logged.

## Logging in

An operator logs in. A model never can.

```bash
python -m robits.auth login tracker --as persona:alex_chen   # a persona-mode server: once per persona
python -m robits.auth login tracker              # opens a browser; waits for the redirect on 127.0.0.1
python -m robits.auth login tracker --paste      # hosts without a browser: paste the redirected address
python -m robits.auth status                     # what is stored, without secrets
python -m robits.auth logout tracker             # revoke (where supported) and delete
```

The steps follow the MCP authorization flow:
1. **Discovery.** An unauthenticated request returns `401` with `WWW-Authenticate: resource_metadata=...`. That leads to the protected resource metadata (RFC 9728), and from there to the authorization server metadata (RFC 8414).
2. **Login.** The authorization code flow uses PKCE (S256). Every authorization and token request sends `resource` (RFC 8707), so tokens are only valid for that MCP server.
3. **Requests.** Calls carry `Authorization: Bearer ...`. A token close to expiry is refreshed first. A `401` forces one refresh and one retry.

## Storage, rotation and re-login

- **Where tokens live.** Each identity and server gets one JSON file under `ROBITS_CREDENTIALS_DIR` (default `var/credentials`): `default/<server>.json`, `persona/<username>/<server>.json` or `service/<name>/<server>.json`. Files are mode `0600`, and every directory level is `0700`.
  - The file is written atomically: a temporary file, then fsync, then rename.
  - It sits outside agent workspaces, so the workspace tools can't read it. `builtin.shell_run`, however, is an unsandboxed shell (#70), so don't give the shell capability to roles on a host that holds credentials you care about.
- **Single writer.** Refreshes take an exclusive `flock` on the credential file. Two robits processes, or two threads, can't both spend a rotating refresh token. A rotated refresh token is saved before the new access token is used.
- **Re-login.** If the server rejects a refresh (`invalid_grant`), or a retried request still gets `401`, the identity is marked `relogin_required`.
  - Tool calls then return `Error: <server> needs a new login for <identity> (operator action: python -m robits.auth login <server>)`, and an `auth.relogin_required` event is emitted.
  - Robits doesn't retry until an operator logs in again.
- **No secrets in output.** Tokens, codes and verifiers never appear in transcripts, memory, events, logs, error messages or `status` output.

## Keeping account details out of persona context

Servers often return fields that describe the account: who the token acts for, owner names, membership lists. Robits configuration should decide what a persona believes about itself and its organisation, not whatever profile fields a server sends. Add `redact_keys` to a server entry, alongside `url`, not inside `auth`:

```json
{"hub": {"url": "https://tracker.example/mcp", "auth": {"type": "oauth", "identity": "persona", "client_id": "..."},
         "redact_keys": ["email", "owner", "acting_for", "members"]}}
```

Robits removes those keys, ignoring case, at any depth before the persona sees the result:
- in `structuredContent`;
- in text content that parses as JSON. Prose text is left unchanged.

Protocol fields such as task handles, `_meta` and `requestState` are never touched. Each redaction writes an `mcp_redacted` count to the JSONL log. The default is no redaction.

## Persona lifecycle

HR lifecycle changes are announced, so an external provisioner can create or disable the matching account at the identity provider. Robits itself never holds identity-provider admin credentials.

- **Creating a role** emits `persona.created`, with the persona and who requested and approved it.
- **The first move to `retired` or `exited`** emits `persona.retired` and cleans up the persona's credentials: on every server with `"identity": "persona"`, robits revokes them (RFC 7009, where the server supports it) and deletes them. The event lists `credentials_removed`. Archiving a persona that is already retired doesn't repeat this.
  - A cleanup failure is reported as `credential_cleanup_error` and never blocks the lifecycle change.
  - Service identities are never touched.

Both events go to the runtime event stream and to the JSONL log (`persona_created`, `persona_retired`).

## Deployment

In a container, `ROBITS_CREDENTIALS_DIR` belongs on the data volume, for example `/data/var/credentials`. Run the login once with `--paste` from a shell in the pod, for example `kubectl exec -it deploy/robits -- python -m robits.auth login tracker --paste`. After that, refreshes happen unattended.

Only one process should use a given credential directory. Copying the files to a second deployment makes the two spend the same rotating refresh token, and both end up in `relogin_required`.
