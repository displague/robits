"""Application-wide configuration and mutable runtime state.

All environment-derived settings and runtime singletons live here so that
sub-modules never need to import the top-level ``main`` entry point.
"""
import os
import threading
from pathlib import Path


def _parse_break_schedule(raw: str) -> list:
    """Parse 'HH:MM-HH:MM[,HH:MM-HH:MM...]' into a list of (start, end) string pairs.

    Normalises single-digit hours (9:00 → 09:00). Rejects midnight-wrapping windows
    (start >= end) and silently skips malformed segments.
    """
    import logging
    windows = []
    for segment in (raw or "").split(","):
        segment = segment.strip()
        if not segment:
            continue
        parts = segment.split("-", 1)
        if len(parts) != 2:
            logging.warning("robits: skipping malformed break schedule segment %r", segment)
            continue
        try:
            from datetime import datetime as _dt
            s = _dt.strptime(parts[0].strip(), "%H:%M").time()
            e = _dt.strptime(parts[1].strip(), "%H:%M").time()
        except ValueError:
            logging.warning("robits: skipping unparseable break schedule segment %r", segment)
            continue
        if s >= e:
            logging.warning(
                "robits: skipping midnight-wrapping or zero-length break window %s-%s",
                s.strftime("%H:%M"), e.strftime("%H:%M"),
            )
            continue
        windows.append((s.strftime("%H:%M"), e.strftime("%H:%M")))
    return windows


class Config:
    """All env-derived settings and mutable runtime state in one injectable object.

    Pass an *env* dict to override ``os.environ`` — useful in tests or when
    composing multiple configurations without touching the process environment.
    """

    def __init__(self, env=None):
        if env is None:
            env = os.environ

        # --- OpenAI client ---
        from openai import OpenAI

        client_kwargs = {"api_key": env.get("OPENAI_API_KEY", "not-needed")}
        organization = env.get("OPENAI_ORG")
        if organization:
            client_kwargs["organization"] = organization
        base_url = env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE")
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        # --- model selection ---
        _default = env.get("ROBITS_MODEL") or env.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.costly_model = env.get("ROBITS_COSTLY_MODEL", _default)
        self.cheap_model = env.get("ROBITS_CHEAP_MODEL", _default)
        self.provider_api = env.get("ROBITS_PROVIDER_API", "responses").strip().lower()

        # --- concurrency / retry ---
        self.max_parallelism = max(1, int(env.get("ROBITS_MAX_PARALLELISM", "1")))
        self.max_api_retries = max(0, int(env.get("ROBITS_MAX_API_RETRIES", "3")))
        self.api_retry_base_seconds = max(
            0.0, float(env.get("ROBITS_API_RETRY_BASE_SECONDS", "0.25"))
        )
        self.api_retry_max_seconds = max(
            self.api_retry_base_seconds,
            float(env.get("ROBITS_API_RETRY_MAX_SECONDS", "4.0")),
        )
        self.model_call_gate = threading.BoundedSemaphore(self.max_parallelism)

        # --- memory / storage ---
        from robits.memory.sqlite import SQLiteMemoryStore
        from robits.runtime.workspace import AgentWorkspaceStore

        _default_db = Path.home() / ".local" / "share" / "robits" / "memory.db"
        self.memory_db_path = env.get("ROBITS_MEMORY_DB") or str(_default_db)
        self.memory_store = SQLiteMemoryStore(self.memory_db_path)
        self.memory_max_depth = max(0, int(env.get("ROBITS_MEMORY_MAX_DEPTH", "3")))
        self.memory_max_rows = max(1, int(env.get("ROBITS_MEMORY_MAX_ROWS", "100")))
        self.memory_cache_threshold = max(
            512, int(env.get("ROBITS_MEMORY_CACHE_THRESHOLD", "8192"))
        )
        self.memory_digest_interval = max(
            0, int(env.get("ROBITS_DIGEST_INTERVAL", "0"))
        )
        self.memory_digest_context_chars = max(
            0, int(env.get("ROBITS_DIGEST_CONTEXT_CHARS", "0"))
        )
        self.memory_digest_elapsed_seconds = max(
            0, int(env.get("ROBITS_DIGEST_ELAPSED_SECONDS", "0"))
        )
        self.memory_identity_digest_interval = max(
            0, int(env.get("ROBITS_IDENTITY_DIGEST_INTERVAL", "0"))
        )
        self.memory_goal_digest_interval = max(
            0, int(env.get("ROBITS_GOAL_DIGEST_INTERVAL", "0"))
        )

        self.agent_workspace_root = env.get("ROBITS_AGENT_WORKSPACE_ROOT", "var/agents")
        self.agent_workspace_store = AgentWorkspaceStore(self.agent_workspace_root)
        self._org_workspace = (
            self.agent_workspace_store if self.memory_store is not None else None
        )

        # --- session / org ---
        self.org_chat_context_lines = max(
            0, int(env.get("ROBITS_ORG_CHAT_CONTEXT_LINES", "20"))
        )
        self.org_digest_interval = max(
            0, int(env.get("ROBITS_ORG_DIGEST_INTERVAL", "0"))
        )

        # --- embeddings ---
        self.embedding_model = env.get("ROBITS_EMBEDDING_MODEL", "").strip()
        self.embedding_base_url = (
            env.get("ROBITS_EMBEDDING_BASE_URL")
            or env.get("OPENAI_BASE_URL")
            or env.get("OPENAI_API_BASE")
            or ""
        ).strip() or None
        self.embedding_api_key = env.get("ROBITS_EMBEDDING_API_KEY", "").strip() or None

        # --- personas ---
        self.personas_file = env.get("ROBITS_PERSONAS_FILE", "").strip() or None
        from robits.core.persona import load_personas
        self.persona_entries = load_personas(self.personas_file)

        # --- schedule ---
        self.break_schedule = _parse_break_schedule(env.get("ROBITS_BREAK_SCHEDULE", ""))
        self.tool_proposals_file = env.get("ROBITS_TOOL_PROPOSALS_FILE", "var/tool_proposals.json")

        # --- misc ---
        self.builtin_search_url = env.get("ROBITS_SEARCH_URL", "").strip()
        self._reasoning_effort_env = (
            env.get("ROBITS_REASONING_EFFORT", "").strip().lower() or None
        )
        _raw_clock = env.get("ROBITS_CLOCK_STATE", "on").strip().lower()
        self.clock_state = _raw_clock if _raw_clock in {"on", "off", "break"} else "on"
        self.default_location = env.get("ROBITS_LOCATION", "").strip()
        self.default_timezone = env.get("ROBITS_TIMEZONE", "").strip()

        # --- mutable runtime state (set by main after sub-module imports) ---
        self.tool_registry = None
        self.model_provider = None
        self.tool_proposal_store = None
        self.active_tool_caller = None
        self.active_tool_caller_name = None
        self.active_session_transcript_length = 0


# Module-level singleton; sub-modules import this instead of ``main``.
_config = Config()
