#!/usr/bin/env python3
"""Main entry point and global state for the robits simulation."""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

from openai import OpenAI

from robits.memory.sqlite import SQLiteMemoryStore
from robits.runtime.tool_proposals import ToolProposalStore
from robits.runtime.workspace import AgentWorkspaceStore, WorkspacePathError


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def make_client():
    client_kwargs = {
        "api_key": os.environ.get("OPENAI_API_KEY", "not-needed"),
    }
    organization = os.environ.get("OPENAI_ORG")
    if organization:
        client_kwargs["organization"] = organization
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


# ---------------------------------------------------------------------------
# Module-level globals (tests and sub-modules reference these via _m.X)
# ---------------------------------------------------------------------------

client = make_client()
default_model = os.environ.get("ROBITS_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
costly_model = os.environ.get("ROBITS_COSTLY_MODEL", default_model)
cheap_model = os.environ.get("ROBITS_CHEAP_MODEL", default_model)
provider_api = os.environ.get("ROBITS_PROVIDER_API", "responses").strip().lower()
max_parallelism = max(1, int(os.environ.get("ROBITS_MAX_PARALLELISM", "1")))
max_api_retries = max(0, int(os.environ.get("ROBITS_MAX_API_RETRIES", "3")))
api_retry_base_seconds = max(0.0, float(os.environ.get("ROBITS_API_RETRY_BASE_SECONDS", "0.25")))
api_retry_max_seconds = max(api_retry_base_seconds, float(os.environ.get("ROBITS_API_RETRY_MAX_SECONDS", "4.0")))
model_call_gate = threading.BoundedSemaphore(max_parallelism)
builtin_search_url = os.environ.get("ROBITS_SEARCH_URL", "").strip()
memory_db_path = os.environ.get("ROBITS_MEMORY_DB")
memory_store = SQLiteMemoryStore(memory_db_path) if memory_db_path else None
tool_proposal_store = None
memory_max_depth = max(0, int(os.environ.get("ROBITS_MEMORY_MAX_DEPTH", "3")))
memory_max_rows = max(1, int(os.environ.get("ROBITS_MEMORY_MAX_ROWS", "100")))
memory_cache_threshold = max(512, int(os.environ.get("ROBITS_MEMORY_CACHE_THRESHOLD", "8192")))
memory_digest_interval = max(0, int(os.environ.get("ROBITS_DIGEST_INTERVAL", "0")))
memory_digest_context_chars = max(0, int(os.environ.get("ROBITS_DIGEST_CONTEXT_CHARS", "0")))
memory_digest_elapsed_seconds = max(0, int(os.environ.get("ROBITS_DIGEST_ELAPSED_SECONDS", "0")))
memory_identity_digest_interval = max(0, int(os.environ.get("ROBITS_IDENTITY_DIGEST_INTERVAL", "0")))
memory_goal_digest_interval = max(0, int(os.environ.get("ROBITS_GOAL_DIGEST_INTERVAL", "0")))
org_chat_context_lines = max(0, int(os.environ.get("ROBITS_ORG_CHAT_CONTEXT_LINES", "20")))
org_digest_interval = max(0, int(os.environ.get("ROBITS_ORG_DIGEST_INTERVAL", "0")))
_reasoning_effort_env = os.environ.get("ROBITS_REASONING_EFFORT", "").strip().lower() or None

_raw_clock_state = os.environ.get("ROBITS_CLOCK_STATE", "on").strip().lower()
clock_state = _raw_clock_state if _raw_clock_state in {"on", "off"} else "on"
agent_workspace_store = AgentWorkspaceStore()
_org_workspace = agent_workspace_store if memory_store is not None else None
default_location = os.environ.get("ROBITS_LOCATION", "").strip()
default_timezone = os.environ.get("ROBITS_TIMEZONE", "").strip()

active_tool_caller = None
active_tool_caller_name = None
active_session_transcript_length = 0


# ---------------------------------------------------------------------------
# Sub-module imports (order matters to avoid partial initialization issues)
# ---------------------------------------------------------------------------

from robits.core.io import TeeStream  # noqa: E402
from robits.core.tools import (  # noqa: E402
    SAFE_TOOL_BUILTINS,
    ToolDefinition,
    ToolRegistry,
    role_can_use_tool,
    _normalize_capabilities,
    _normalize_tool_grants,
    _coerce_json_object,
    _normalize_tool_parameters,
)
from robits.core.context import (  # noqa: E402
    agent_runtime_context,
    format_agent_context,
    format_org_chat_context,
    current_agent_context,
    format_verified_tool_results,
    deliver_verified_tool_results,
    prepend_verified_tool_results,
    _get_identity_digests,
)
from robits.core.lifecycle import (  # noqa: E402
    LIFECYCLE_STATES,
    PROTECTED_ROLE_NAMES,
    ALARM_RECURRENCES,
    LifecycleEvent,
    Alarm,
    validate_role_name,
    validate_role_description,
    record_lifecycle_event,
    create_lifecycle_role,
    change_lifecycle_state,
    pause_lifecycle_role,
    retire_lifecycle_role,
    archive_lifecycle_role,
    list_lifecycle_roles,
    create_alarm,
    list_alarms,
    cancel_alarm,
    due_alarm_reminders,
)
from robits.core.tool_functions import (  # noqa: E402
    workspace_list,
    workspace_read,
    workspace_write,
    workspace_delete,
    org_chat_read,
    work_todo_add,
    grant_tool_access,
    revoke_tool_access,
    list_registered_tools,
    propose_tool_change,
    list_tool_proposals,
    approve_tool_proposal,
    reject_tool_proposal,
    rollout_tool_proposal,
    approve_and_rollout_proposal,
    memory_search,
    memory_list_digests,
    memory_expand_digest,
    builtin_web_search,
    builtin_file_search,
    builtin_shell_run,
    builtin_tool_search,
    builtin_mcp_call,
    builtin_computer_use,
    builtin_image_generation,
    agent_think,
    agent_wait,
    _condense_if_large,
)
from robits.core.roles import (  # noqa: E402
    Role,
    System,
    Ops,
    HR,
    Angel,
    SoftwareEngineer,
    Human,
    build_employee_dict,
    parse_tool_instruction,
    parse_agent_action,
    load_tools,
    interact,
    _ON_CLOCK_GUIDANCE,
    _OFF_CLOCK_GUIDANCE,
)
from robits.core.providers import (  # noqa: E402
    ModelProvider,
    ChatCompletionsProvider,
    ResponsesProvider,
    make_model_provider,
    _emit_role_tool_event,
    _with_model_retries,
)
from robits.core.session import (  # noqa: E402
    TranscriptEntry,
    RuntimeEvent,
    RuntimeEventStream,
    RoutedMessage,
    RoundRobinScheduler,
    Session,
)

tool_registry = ToolRegistry()
model_provider = make_model_provider()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_simulation(initial_message=None, max_turns=None):
    session = Session()
    load_tools(session.system)
    return session.run(initial_message=initial_message, max_turns=max_turns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="Initial message to start the simulation.")
    parser.add_argument("--turns", type=int, help="Maximum model turns to run.")
    parser.add_argument("--log", help="Write the console transcript to this file while still printing it.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = TeeStream(original_stdout, log_file)
            sys.stderr = TeeStream(original_stderr, log_file)
            try:
                run_simulation(initial_message=args.prompt, max_turns=args.turns)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    else:
        run_simulation(initial_message=args.prompt, max_turns=args.turns)


if __name__ == "__main__":
    main()
