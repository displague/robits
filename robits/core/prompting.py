"""System-prompt building blocks, chosen by profile and by the tools a persona is offered.

Two rules keep prompts small enough for small local models:
- A block that tells the model how to use a tool appears only if the persona is offered that
  tool, so a persona with three tools isn't told about delegation, memory recall or tool
  authoring it can't do.
- The "compact" profile replaces the long turn rules with a few direct lines, including an
  explicit instruction to answer from the conversation, and renders runtime state as one short
  line instead of JSON.
"""
from robits.core.config import _config as _m

PROFILES = ("standard", "compact")
REASONING_EFFORTS = ("none", "low", "medium", "high")

_BASE = (
    "On your turn, choose one useful action. You do not need to reply to every message.\n"
    "Only say that you created, changed, or called a tool after the runtime has returned a verified "
    "tool result. Without a verified result, describe the work as a request, proposal, or plan "
    "instead of a completed fact.\n"
)
_DELEGATION = (
    "Stay in your conversational persona while working. Use agent.focus as a private compartment for "
    "focused reflection, inner dialogue, checking details, or mundane work when this helps keep your "
    "conversation context small. Give it only the context it needs and ask for concise findings, "
    "uncertainties, or a question back. Use agent.spawn when you need explicit tool, model, or queue "
    "options. You may also use available tools directly for simple work; you do not need to delegate "
    "every turn. Bring useful findings back into your own response without narrating internal steps or "
    "tool names, unless the person asks or needs that detail to make a decision. Be clear about missing "
    "information and failed work. Internal activity remains available to the runtime observer "
    "separately from your public reply.\n"
)
_MEMORY = (
    "When recalling past events, experiences, or recent work — even if the answer seems available in "
    "context — use memory.search (for specific queries), memory.list_digests, or memory.expand_digest "
    "to retrieve the full picture before answering.\n"
)
_RULE_ROLE = (
    "ROLE BOUNDARIES: Strictly adhere to your own role responsibilities. Do not offer to perform, "
    "describe, or execute actions that belong to other roles (e.g., only HR creates/manages roles, only "
    "SE implements software/tools). Do not copy the options, lists, or text formatting of other roles "
    "in the chat history."
)
_RULE_THINK = (
    "SILENT THINKING: If the CEO or sender instructs you to \"think to yourself\", \"think silently\", "
    "or only think, use the native agent.think tool for a private note and DO NOT output any public "
    "reply or text. A focused private subtask may help with a longer reflection. Use the legacy "
    "{\"action\": \"think\", \"content\": \"...\"} form only when native tool calling is unavailable."
)
_RULE_WAIT = (
    "WAITING, SLEEPING, NAPPING & SKIPPING TURNS: If the CEO or sender instructs you to wait, sleep, "
    "pause, take a break, nap, take a lap/nap, skip turns, go home, or take time off for any number of "
    "minutes/hours, you MUST immediately call the `agent.wait` tool with the requested duration in "
    "minutes (e.g., call `agent.wait` with `minutes=45`). Do not refuse, and do not reply textually "
    "saying you are an AI or cannot sleep. Simply call the tool. If no duration is specified, call "
    "`agent.wait` with `minutes=10` or return {\"action\": \"wait\"}. Do not ignore these commands or "
    "continue replying."
)

_COMPACT_BASE = (
    "Reply to the latest message briefly and directly.\n"
    "Use the facts given in the conversation. If the answer is there, give it; do not ask for more "
    "context. Ask a question only when something you need is truly missing.\n"
    "Write only your reply: no JSON, no tool syntax, no labels such as \"agent_name:\" or "
    "\"response:\". Call tools only through tool calling.\n"
    "Only say you did something after a tool result confirms it.\n"
)
_COMPACT_WAIT = "If asked to wait, pause or take a break, call agent.wait with the minutes (10 if none are given).\n"
_COMPACT_MEMORY = "For past events that are not in this conversation, call memory.search.\n"
_COMPACT_THINK = "If asked to think privately, call agent.think and reply with nothing else.\n"

_GUIDANCE = {
    ("standard", "on"): (
        "\nCommunication style: default to concise work-focused responses — brief status updates, "
        "statements of intent, targeted questions. Reserve longer responses for technical depth "
        "when the context warrants it.\n"
    ),
    ("standard", "off"): (
        "\nCommunication style: you are off the clock. Engage personally and warmly — "
        "short, natural conversation; curiosity; empathy. If work topics surface unexpectedly, "
        "note them briefly and park via the work.todo tool rather than engaging at length.\n"
    ),
    ("compact", "on"): "Keep replies short and work-focused.\n",
    ("compact", "off"): "You are off the clock: keep it short, personal and warm.\n",
}


def prompt_profile(role):
    """The role's profile: its persona's prompt_profile, else ROBITS_PROMPT_PROFILE, else standard."""
    profile = getattr(role, "prompt_profile", None) or getattr(_m, "prompt_profile", None) or "standard"
    return profile if profile in PROFILES else "standard"


def offered_tool_names(role):
    """Names of the tools whose schemas this role is sent, or None when unknown (no registry)."""
    registry = getattr(_m, "tool_registry", None)
    if registry is None or not hasattr(registry, "offered_tools"):
        return None
    try:
        return {tool.name for tool in registry.offered_tools(role)}
    except Exception:
        return None


def _has(names, *tools):
    return names is None or any(tool in names for tool in tools)


def turn_template(role, profile=None, names=None):
    """Turn rules for the role, mentioning only tools it is offered."""
    profile = profile or prompt_profile(role)
    if profile == "compact":
        text = _COMPACT_BASE
        if _has(names, "agent.wait"):
            text += _COMPACT_WAIT
        if _has(names, "memory.search"):
            text += _COMPACT_MEMORY
        if _has(names, "agent.think"):
            text += _COMPACT_THINK
        return "\n" + text
    text = "\n" + _BASE
    if _has(names, "agent.focus", "agent.spawn"):
        text += _DELEGATION
    if _has(names, "memory.search", "memory.list_digests", "memory.expand_digest"):
        text += _MEMORY
    rules = [_RULE_ROLE]
    if _has(names, "agent.think"):
        rules.append(_RULE_THINK)
    if _has(names, "agent.wait"):
        rules.append(_RULE_WAIT)
    text += "\nCRITICAL ROLE & COMPLIANCE RULES:\n" + "".join(f"{i}. {rule}\n" for i, rule in enumerate(rules, 1))
    return text


def guidance(profile, clock_state):
    return _GUIDANCE[(profile, "on" if clock_state == "on" else "off")]


def tool_authoring_notes(role, names=None):
    """The tool-authoring manual, only for a role that can call tools.propose."""
    notes = getattr(role, "tool_authoring_notes", "")
    return f"\n{notes}\n" if notes and _has(names, "tools.propose") else ""


def compact_context(role):
    """Runtime context as one short line (time, clock, identity, location)."""
    from robits.core.context import agent_runtime_context
    context = agent_runtime_context(role)
    local = str(context.get("current_datetime_local", ""))[:16].replace("T", " ")
    utc = str(context.get("current_datetime_utc", ""))[11:16]
    parts = [f"Now: {local} {context.get('timezone', '')} ({utc} UTC)".strip()]
    if context.get("location"):
        parts.append(f"Location: {context['location']}")
    if context.get("agent_name"):
        parts.append(f"You are {context['agent_name']}")
    return ". ".join(parts) + ".\n"


def unavailable_tools_note(role):
    """Name the tools in a persona's list that aren't registered right now.

    MCP tools only exist once their server has been discovered, which needs a login and
    sometimes the server's approval. Without this note a small model may claim results from
    tools it was never given.
    """
    ceiling = getattr(role, "tool_ceiling", None)
    registry = getattr(_m, "tool_registry", None)
    if not ceiling or registry is None:
        return ""
    labels = getattr(role, "tool_labels", {}) or {}
    missing = [labels.get(name, name) for name in ceiling
               if "*" not in name and name not in registry]
    if not missing:
        return ""
    return ("Unavailable right now: " + ", ".join(missing) + ". If a task needs them, say you could not "
            "do it and why; do not describe results you did not get.\n")
