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
| `identity` | Which identity's tokens this server uses, for example `service:team`. Defaults to `default`. Separate identities per persona are planned for #138 part 2. |

## Logging in

An operator logs in. A model never can.

```bash
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

- **Where tokens live.** Each identity and server gets one JSON file under `ROBITS_CREDENTIALS_DIR` (default `var/credentials`), mode `0600`, in a `0700` directory.
  - The file is written atomically: a temporary file, then fsync, then rename.
  - It sits outside agent workspaces, so the workspace tools can't read it. `builtin.shell_run`, however, is an unsandboxed shell (#70), so don't give the shell capability to roles on a host that holds credentials you care about.
- **Single writer.** Refreshes take an exclusive `flock` on the credential file. Two robits processes, or two threads, can't both spend a rotating refresh token. A rotated refresh token is saved before the new access token is used.
- **Re-login.** If the server rejects a refresh (`invalid_grant`), or a retried request still gets `401`, the identity is marked `relogin_required`.
  - Tool calls then return `Error: <server> needs a new login for <identity> (operator action: python -m robits.auth login <server>)`, and an `auth.relogin_required` event is emitted.
  - Robits doesn't retry until an operator logs in again.
- **No secrets in output.** Tokens, codes and verifiers never appear in transcripts, memory, events, logs, error messages or `status` output.

## Deployment

In a container, `ROBITS_CREDENTIALS_DIR` belongs on the data volume, for example `/data/var/credentials`. Run the login once with `--paste` from a shell in the pod, for example `kubectl exec -it deploy/robits -- python -m robits.auth login tracker --paste`. After that, refreshes happen unattended.

Only one process should use a given credential directory. Copying the files to a second deployment makes the two spend the same rotating refresh token, and both end up in `relogin_required`.
