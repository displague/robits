"""Durable checkpoints for private subtask records (issue #132, first slice).

One SQLite file holds a JSON snapshot per private subtask: scope (owner and root), depth, model,
private message history, events, status, result, outstanding input request and the role's plain
settings. No credentials, no live objects and no model or tool handles are stored; on recovery the
role is rebuilt under the caller's *current* authority (see ``SubtaskManager.recover``).

The file holds private working history, so it is created owner-only (0600). It is opt-in
(``ROBITS_SUBTASK_DB``) and rows expire ``ROBITS_SUBTASK_RETENTION_S`` after their last change.
"""
import json
import os
import sqlite3
import threading
import time

FINISHED = {"completed", "failed", "cancelled"}      # nothing more to do or cancel
TERMINAL = FINISHED | {"interrupted"}                  # cannot run again, but can still be cancelled


def _encode(value):
    """JSON text that keeps sets and tuples recognisable."""
    def default(obj):
        if isinstance(obj, (set, frozenset)):
            return {"__set__": sorted(obj, key=str)}
        if isinstance(obj, tuple):
            return {"__tuple__": list(obj)}
        return str(obj)

    def walk(obj):
        if isinstance(obj, tuple):
            return {"__tuple__": [walk(x) for x in obj]}
        if isinstance(obj, (set, frozenset)):
            return {"__set__": [walk(x) for x in sorted(obj, key=str)]}
        if isinstance(obj, dict):
            return {str(k): walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        return obj
    return json.dumps(walk(value), default=default)


def _decode(text):
    def walk(obj):
        if isinstance(obj, dict):
            if set(obj) == {"__tuple__"}:
                return tuple(walk(x) for x in obj["__tuple__"])
            if set(obj) == {"__set__"}:
                return set(walk(x) for x in obj["__set__"])
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        return obj
    return walk(json.loads(text))


class TaskStore:
    def __init__(self, path, retention_s=7 * 86400, clock=time.time):
        self.path, self.retention_s, self.clock = path, retention_s, clock
        self.lock = threading.Lock()
        # Owner-only, before SQLite opens it: a new file is created 0600, and an existing one (say an
        # empty file made with a looser umask) is tightened, so the history is never world-readable.
        if path != ":memory:":
            os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
            os.chmod(path, 0o600)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS subtasks (
            task_id TEXT PRIMARY KEY, root TEXT NOT NULL, owner TEXT NOT NULL,
            status TEXT NOT NULL, updated REAL NOT NULL, body TEXT NOT NULL)""")
        self.db.commit()

    def save(self, snapshot):
        with self.lock:
            self.db.execute(
                "INSERT INTO subtasks (task_id, root, owner, status, updated, body) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET root=excluded.root, owner=excluded.owner, "
                "status=excluded.status, updated=excluded.updated, body=excluded.body",
                (snapshot["task_id"], _encode(snapshot["root"]), _encode(snapshot["owner"]),
                 snapshot["status"], self.clock(), _encode(snapshot)))
            self.db.commit()

    def purge(self):
        """Drop rows not changed within the retention window; returns how many."""
        with self.lock:
            cursor = self.db.execute("DELETE FROM subtasks WHERE updated < ?", (self.clock() - self.retention_s,))
            self.db.commit()
            return cursor.rowcount

    def load_root(self, root):
        """Snapshots of one root's tasks, oldest first (expired rows are purged first)."""
        self.purge()
        with self.lock:
            rows = self.db.execute("SELECT body FROM subtasks WHERE root = ? ORDER BY rowid", (_encode(root),)).fetchall()
        return [_decode(body) for (body,) in rows]

    def close(self):
        with self.lock:
            try:
                self.db.close()
            except sqlite3.Error:
                pass
