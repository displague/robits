"""Model provider implementations and API helpers."""
import json
import random
import re
import time
from uuid import uuid4

from robits.core.config import _config as _m

# Compiled once at module load — used by _extract_xml_tool_calls
_XML_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
_XML_QWEN_FN_RE = re.compile(
    r"<function=(?P<name>[^>]+)>\s*(?P<body>.*?)\s*</function>",
    flags=re.DOTALL,
)
_XML_QWEN_PARAM_RE = re.compile(
    r"<parameter=(?P<key>[^>]+)>\s*(?P<val>.*?)\s*</parameter>",
    flags=re.DOTALL,
)
# Used by _extract_thinking_chat
_INLINE_THINK_RE = re.compile(
    r"<(?P<tag>think|thinking)>(?P<body>.*?)</(?P=tag)>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _response_text(response):
    """Extract concatenated text content from an OpenAI Responses API response object."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    chunks = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        elif getattr(item, "type", None) in {"output_text", "text"}:
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def _response_function_calls(response):
    """Return all function_call output items from a Responses API response."""
    calls = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        calls.append(item)
    return calls


def _is_retryable_api_error(error):
    """Return True for transient API errors (rate-limits, timeouts, 5xx) that warrant a retry."""
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = error.__class__.__name__.lower()
    return any(token in name for token in ("ratelimit", "timeout", "connection"))


def _with_model_retries(operation):
    """Run operation() under the concurrency gate, retrying on transient API errors."""
    attempt = 0
    while True:
        try:
            with _m.model_call_gate:
                return operation()
        except Exception as error:
            if attempt >= _m.max_api_retries or not _is_retryable_api_error(error):
                raise
            delay = min(_m.api_retry_max_seconds, _m.api_retry_base_seconds * (2**attempt))
            if delay:
                time.sleep(delay + random.uniform(0, delay / 4))
            attempt += 1


def _normalize_xml_tool_name(registry, name):
    """Map a local-model tool name to its registry canonical form.

    Local models (Qwen, Granite) emit snake_case names such as
    ``memory_list_digests`` while the registry stores dotted names
    (``memory.list_digests``) with double-underscore OpenAI aliases
    (``memory__list_digests``).  This function tries three lookups in order:

    1. Direct resolve (handles dotted names and ``__`` aliases).
    2. Dot-substitution scan: find a canonical name whose dots-replaced-by-
       underscores form matches the input.
    3. Fallback to the original name so ``execute`` can return a clear error.
    """
    resolved = registry.resolve_name(name)
    if registry.has(resolved):
        return resolved
    for canonical in registry._tools:
        if canonical.replace(".", "_") == name:
            return canonical
    return name


def _extract_xml_tool_calls(text):
    """Extract <tool_call> blocks from model content and return (clean_text, tool_call_list).

    Handles two sub-formats emitted by local models when LM Studio does not
    translate them to the OpenAI tool_calls array:

    Qwen XML:
        <tool_call>
        <function=NAME>
        <parameter=KEY>VALUE</parameter>
        </function>
        </tool_call>

    Granite / generic JSON-in-tag:
        <tool_call>
        {"name": "TOOL", "arguments": "{...}"}
        </tool_call>

    Returns a list of dicts: [{"name": str, "arguments": dict}, ...].
    Unrecognised blocks are left in the cleaned text.
    """
    extracted = []

    def _handle_block(inner):
        fn_match = _XML_QWEN_FN_RE.search(inner)
        if fn_match:
            name = fn_match.group("name").strip()
            body = fn_match.group("body")
            args = {}
            for pm in _XML_QWEN_PARAM_RE.finditer(body):
                args[pm.group("key").strip()] = pm.group("val").strip()
            extracted.append({"name": name, "arguments": args})
            return True
        inner_stripped = inner.strip()
        if inner_stripped.startswith("{"):
            try:
                obj = json.loads(inner_stripped)
                name = obj.get("name") or obj.get("function")
                raw_args = obj.get("arguments") or obj.get("args") or {}
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}
                if name and isinstance(raw_args, dict):
                    extracted.append({"name": name, "arguments": raw_args})
                    return True
            except json.JSONDecodeError:
                pass
        return False

    def _replace(match):
        inner = match.group(1)
        return "" if _handle_block(inner) else match.group(0)

    clean = _XML_BLOCK_RE.sub(_replace, text).strip()
    return clean, extracted


def _execute_tool_call(registry, role, tool_name, call_id, args, employee_dict):
    """Execute one tool call and return the tool-result message dict.

    Handles event emission, result recording (skipped for agent.think), and
    formats the OpenAI-style tool result message.  ``tool_name`` must already
    be the registry canonical form.
    """
    payload = {
        "agent": getattr(role, "name", None),
        "role_name": getattr(role, "runtime_role_name", None),
        "call_id": call_id,
        "tool_name": tool_name,
        "arguments": args,
    }
    _emit_role_tool_event(role, "tool_call.requested", payload)
    result = registry.execute(tool_name, args, employee_dict, caller=role)
    event_type = "tool_call.failed" if _tool_result_is_error(result) else "tool_call.executed"
    _emit_role_tool_event(role, event_type, {**payload, "result": result})
    if registry.resolve_name(tool_name) != "agent.think":
        _record_role_tool_result(
            role,
            f"{event_type}: {tool_name}({json.dumps(args, sort_keys=True)}) -> {result}",
        )
    return {"role": "tool", "tool_call_id": call_id, "content": str(result)}


def _extract_thinking_chat(message):
    """Extract thinking/reasoning from a ChatCompletions message.

    Returns (content_text, thinking_text | None). content_text has inline
    thinking tags stripped so only the final expressed output is returned.
    """
    def _strip_inline_thinking(text):
        if not text:
            return "", None
        blocks = []
        for match in _INLINE_THINK_RE.finditer(text):
            body = (match.group("body") or "").strip()
            if body:
                blocks.append(body)
        clean = _INLINE_THINK_RE.sub("", text).strip()
        return clean, ("\n\n".join(blocks) if blocks else None)

    def _join_thinking(*parts):
        chunks = []
        for part in parts:
            if part and isinstance(part, str) and part.strip():
                chunks.append(part.strip())
        return "\n\n".join(chunks) if chunks else None

    # Anthropic-compat via OpenAI endpoint: content is a list of typed blocks
    content = getattr(message, "content", None)
    if isinstance(content, list):
        text_parts = []
        thinking_parts = []
        for block in content:
            if hasattr(block, "type"):
                btype, btext = block.type, getattr(block, "text", None)
                bthink = getattr(block, "thinking", None)
            elif isinstance(block, dict):
                btype = block.get("type")
                btext = block.get("text")
                bthink = block.get("thinking")
            else:
                continue
            if btype == "thinking":
                if bthink:
                    thinking_parts.append(bthink)
                elif btext:
                    thinking_parts.append(btext)
            elif btext:
                text_parts.append(btext)
        content_text, inline_thinking = _strip_inline_thinking("".join(text_parts))
        typed_thinking = "\n\n".join([str(t).strip() for t in thinking_parts if str(t).strip()]) or None
        reasoning = getattr(message, "reasoning_content", None)
        thinking_attr = getattr(message, "thinking", None)
        provider_thinking = None
        if reasoning and isinstance(reasoning, str):
            provider_thinking = reasoning
        elif thinking_attr and isinstance(thinking_attr, str):
            provider_thinking = thinking_attr
        return content_text, _join_thinking(provider_thinking, typed_thinking, inline_thinking)

    # String content: always strip inline tags, then merge with any provider-specific fields.
    content_str = str(content) if content is not None else ""
    content_text, inline_thinking = _strip_inline_thinking(content_str)

    # DeepSeek-R1 / QwQ via OpenAI-compat: dedicated reasoning_content field
    reasoning = getattr(message, "reasoning_content", None)
    # Some providers expose a top-level thinking field
    thinking_attr = getattr(message, "thinking", None)
    provider_thinking = None
    if reasoning and isinstance(reasoning, str):
        provider_thinking = reasoning
    elif thinking_attr and isinstance(thinking_attr, str):
        provider_thinking = thinking_attr

    return content_text, _join_thinking(provider_thinking, inline_thinking)


def _extract_thinking_responses(response):
    """Extract reasoning text from a Responses API response; return text or None.

    Handles three shapes:
    - item.summary[{type:"summary_text", text:...}]  (reasoning.summary:"auto")
    - item.content[{text:...}]                        (raw content blocks)
    - item.text                                        (direct text field)
    """
    chunks = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        # OpenAI Responses API with reasoning.summary:"auto"
        for part in getattr(item, "summary", []) or []:
            if getattr(part, "type", None) == "summary_text":
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        # Raw content blocks
        for part in getattr(item, "content", []) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
        # Direct text field (some providers)
        text = getattr(item, "text", None)
        if text:
            chunks.append(text)
    return "\n\n".join(chunks) if chunks else None


def _emit_role_tool_event(role, event_type, payload):
    """Emit a tool event to the role's event stream and persist it to the memory store."""
    event_stream = getattr(role, "runtime_event_stream", None)
    session_id = getattr(role, "runtime_session_id", None)
    if event_stream is not None and session_id is not None:
        event_stream.emit(event_type, session_id, payload)
    if _m.memory_store is not None and event_type in {"tool_call.executed", "tool_call.failed"}:
        agent_id = getattr(role, "runtime_role_name", None) or getattr(role, "name", None)
        if agent_id:
            try:
                _m.memory_store.append_tool_call(
                    tool_call_id=payload.get("call_id") or str(uuid4()),
                    agent_id=agent_id,
                    tool_name=payload.get("tool_name", ""),
                    arguments=payload.get("arguments"),
                    result_content=str(payload.get("result", "")),
                    status="executed" if event_type == "tool_call.executed" else "failed",
                    session_id=session_id,
                )
            except Exception:
                pass


def _record_role_tool_result(role, result):
    """Append a tool-result string to the role's runtime_tool_results list."""
    if not hasattr(role, "runtime_tool_results"):
        role.runtime_tool_results = []
    role.runtime_tool_results.append(result)


def _tool_result_is_error(raw_result):
    """Return True if the raw execute() output represents an error."""
    return str(raw_result).startswith("Error:")


class ModelProvider:
    """Abstract base for model provider implementations."""

    def generate(self, role, model, sender, messages):
        """Generate a response for role given messages; return the response text."""
        raise NotImplementedError


class ChatCompletionsProvider(ModelProvider):
    """ModelProvider backed by the OpenAI Chat Completions API."""

    def __init__(self, api_client, registry=None):
        self.client = api_client
        self.registry = registry if registry is not None else _m.tool_registry

    def generate(self, role, model, sender, messages):
        """Run a chat-completions agentic loop, executing tool calls until a text reply is produced."""
        employee_dict = getattr(role, "employee_dict", None)
        tools = self.registry.as_chat_completion_tools(role)
        kwargs = {
            "model": model,
            "messages": list(messages),
            "max_tokens": role.max_tokens,
            "n": 1,
            "temperature": role.temperature,
            "user": f"robits_{role.name}",
        }
        top_p = getattr(role, "top_p", None)
        if top_p is not None:
            kwargs["top_p"] = top_p
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        for _ in range(8):
            response = _with_model_retries(lambda: self.client.chat.completions.create(**kwargs))
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            content, thinking = _extract_thinking_chat(message)
            if thinking:
                current = getattr(role, "runtime_thinking", None)
                role.runtime_thinking = (current + "\n\n" + thinking) if current else thinking
            if not tool_calls:
                # Check for XML-format tool calls emitted by Qwen/Granite when LM
                # Studio doesn't translate them to the OpenAI tool_calls array.
                if tools and "<tool_call>" in (content or ""):
                    clean_content, xml_calls = _extract_xml_tool_calls(content or "")
                    if xml_calls and employee_dict is not None:
                        # Synthesise an assistant message with fabricated tool_calls so
                        # the model sees proper tool results on the next round.
                        synth_calls = []
                        for xc in xml_calls:
                            synth_calls.append({
                                "id": f"call_{uuid4().hex[:16]}",
                                "type": "function",
                                "function": {
                                    "name": xc["name"],
                                    "arguments": json.dumps(xc["arguments"]),
                                },
                            })
                        kwargs["messages"] = list(kwargs["messages"]) + [{
                            "role": "assistant",
                            "content": clean_content or None,
                            "tool_calls": synth_calls,
                        }]
                        tool_results = [
                            _execute_tool_call(
                                self.registry,
                                role,
                                _normalize_xml_tool_name(self.registry, xc["name"]),
                                call_dict["id"],
                                xc["arguments"],
                                employee_dict,
                            )
                            for call_dict, xc in zip(synth_calls, xml_calls)
                        ]
                        kwargs["messages"] = kwargs["messages"] + tool_results
                        continue
                return content
            if employee_dict is None:
                return "Error: Tool call requested but no employee dictionary is available."
            kwargs["messages"] = list(kwargs["messages"]) + [message.model_dump()]
            tool_results = []
            for call in tool_calls:
                tool_name = call.function.name
                call_id = call.id
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as error:
                    payload = {
                        "agent": getattr(role, "name", None),
                        "role_name": getattr(role, "runtime_role_name", None),
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "arguments": call.function.arguments,
                        "error": f"Invalid tool arguments JSON: {error}",
                    }
                    _emit_role_tool_event(role, "tool_call.requested", payload)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"Error: Invalid tool arguments JSON: {error}",
                    })
                    continue
                tool_results.append(
                    _execute_tool_call(self.registry, role, tool_name, call_id, args, employee_dict)
                )
            kwargs["messages"] = kwargs["messages"] + tool_results
        return "Error: Too many consecutive tool-call rounds."


class ResponsesProvider(ModelProvider):
    """ModelProvider backed by the OpenAI Responses API (stateful multi-turn)."""

    def __init__(self, api_client, registry=None):
        self.client = api_client
        self.registry = registry if registry is not None else _m.tool_registry

    def generate(self, role, model, sender, messages):
        """Run a responses agentic loop, chaining previous_response_id for state continuity."""
        employee_dict = getattr(role, "employee_dict", None)
        tools = self.registry.as_responses_tools(role)
        kwargs = {
            "model": model,
            "input": messages,
            "max_output_tokens": role.max_tokens,
            "temperature": role.temperature,
            "user": f"robits_{role.name}",
        }
        top_p = getattr(role, "top_p", None)
        if top_p is not None:
            kwargs["top_p"] = top_p
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        effort = getattr(role, "reasoning_effort", None) or _m._reasoning_effort_env
        if effort and effort in {"low", "medium", "high"}:
            kwargs["reasoning"] = {"effort": effort}

        response = _with_model_retries(lambda: self.client.responses.create(**kwargs))
        for _ in range(8):
            function_calls = _response_function_calls(response)
            thinking = _extract_thinking_responses(response)
            if thinking:
                current = getattr(role, "runtime_thinking", None)
                role.runtime_thinking = (current + "\n\n" + thinking) if current else thinking
            if not function_calls:
                return _response_text(response)
            if employee_dict is None:
                return "Error: Tool call requested but no employee dictionary is available."
            outputs = []
            for call in function_calls:
                call_id = getattr(call, "call_id", getattr(call, "id", ""))
                tool_name = getattr(call, "name", "")
                payload = {
                    "agent": getattr(role, "name", None),
                    "role_name": getattr(role, "runtime_role_name", None),
                    "provider_response_id": getattr(response, "id", None),
                    "call_id": call_id,
                    "tool_name": tool_name,
                }
                try:
                    args = json.loads(getattr(call, "arguments", "") or "{}")
                except json.JSONDecodeError as error:
                    payload["arguments"] = getattr(call, "arguments", "")
                    payload["error"] = f"Invalid tool arguments JSON: {error}"
                    _emit_role_tool_event(role, "tool_call.requested", payload)
                    result = f"Error: Invalid tool arguments JSON: {error}"
                else:
                    payload["arguments"] = args
                    _emit_role_tool_event(role, "tool_call.requested", payload)
                    result = self.registry.execute(
                        tool_name,
                        args,
                        employee_dict,
                        caller=role,
                    )
                status_payload = {**payload, "result": result}
                event_type = "tool_call.failed" if _tool_result_is_error(result) else "tool_call.executed"
                _emit_role_tool_event(role, event_type, status_payload)
                resolved_name = self.registry.resolve_name(tool_name)
                if resolved_name != "agent.think":
                    _record_role_tool_result(
                        role,
                        f"{event_type}: {tool_name}({json.dumps(payload.get('arguments'), sort_keys=True)}) -> {result}",
                    )
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }
                )
            previous_response_id = getattr(response, "id", None)
            if previous_response_id:
                response = _with_model_retries(
                    lambda: self.client.responses.create(
                        model=model,
                        input=outputs,
                        previous_response_id=previous_response_id,
                    )
                )
            else:
                kwargs["input"] = messages + outputs
                response = _with_model_retries(lambda: self.client.responses.create(**kwargs))
        return "Error: Too many consecutive tool-call rounds."


def make_model_provider():
    """Instantiate the configured ModelProvider from ROBITS_PROVIDER_API."""
    if _m.provider_api in {"chat", "chat_completions", "chat-completions"}:
        return ChatCompletionsProvider(_m.client)
    return ResponsesProvider(_m.client)
