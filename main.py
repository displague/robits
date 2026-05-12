#!/usr/bin/env python3
"""Main entry point and re-export harness for the robits simulation.

All global state lives in ``robits.core.config._config``.  This module
re-exports every public symbol so that existing ``import main; main.X`` usage
continues to work.  Attribute reads and writes on the ``main`` module are
proxied through to ``_config`` for attributes that are not explicitly bound in
this module's namespace (i.e. classes, functions, and CLI helpers).
"""
import sys
import time  # kept for patch("main.time.sleep") compatibility in tests
import types
from pathlib import Path

from robits.core.config import _config, Config  # noqa: F401


# ---------------------------------------------------------------------------
# Sub-module imports (order matters: config must be fully initialised first)
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
    SANDBOX_GLOBALS,
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

from robits.memory.sqlite import SQLiteMemoryStore  # noqa: F401
from robits.runtime.tool_proposals import ToolProposalStore  # noqa: F401
from robits.runtime.workspace import AgentWorkspaceStore, WorkspacePathError  # noqa: F401


# ---------------------------------------------------------------------------
# Initialise runtime singletons on _config
# ---------------------------------------------------------------------------

_config.tool_registry = ToolRegistry()
_config.model_provider = make_model_provider()


# ---------------------------------------------------------------------------
# Module proxy — forward attribute reads/writes to _config for state attrs
# ---------------------------------------------------------------------------

class _MainModule(types.ModuleType):
    """Proxy so that ``main.X = y`` writes reach ``_config.X`` transparently."""

    def __getattr__(self, name):
        try:
            return getattr(_config, name)
        except AttributeError:
            raise AttributeError(f"module 'main' has no attribute {name!r}") from None

    def __setattr__(self, name, value):
        if name.startswith("__") and name.endswith("__"):
            super().__setattr__(name, value)
        elif name in self.__dict__:
            # Attribute lives in this module's own namespace (re-exported
            # classes/functions defined here).  Keep it there so that code
            # inside main.py that calls these by name still finds the patched
            # version in its globals.
            super().__setattr__(name, value)
        else:
            setattr(_config, name, value)

    def __delattr__(self, name):
        # ``patch.object`` calls delattr on teardown for attributes it found via
        # ``__getattr__`` (i.e. not in ``__dict__``).  Those attributes live on
        # ``_config``, so the delete is a no-op here — the value was already
        # restored via ``__setattr__`` before this call.
        try:
            super().__delattr__(name)
        except AttributeError:
            pass


sys.modules[__name__].__class__ = _MainModule


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_simulation(initial_message=None, max_turns=None):
    """Initialise the session, load tools, and run the simulation loop."""
    session = Session()
    load_tools(session.system)
    return session.run(initial_message=initial_message, max_turns=max_turns)


def parse_args(argv=None):
    """Parse CLI arguments and return the namespace."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="Initial message to start the simulation.")
    parser.add_argument("--turns", type=int, help="Maximum model turns to run.")
    parser.add_argument(
        "--log", help="Write the console transcript to this file while still printing it."
    )
    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry point: parse args, optionally tee stdout/stderr, run simulation."""
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
