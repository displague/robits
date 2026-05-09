import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple


class LLMClient:
    """Thin wrapper around a Responses-compatible endpoint with async helpers."""

    def __init__(self) -> None:
        base = os.getenv("ROBOTS_RESPONSES_BASE_URL")
        if not base:
            base = os.getenv("OPENAI_BASE_URL_INTERNAL")
        if not base:
            base = "http://localhost:8080"
        self.base_url = base.rstrip("/")
        self.default_model = os.getenv("ROBOTS_DEFAULT_MODEL", "gpt-oss")
        self.dummy_mode = os.getenv("ROBOTS_DUMMY_MODE", "0") in {"1", "true", "TRUE"}
        self.timeout = float(os.getenv("ROBOTS_RESPONSES_TIMEOUT", "120"))

    async def generate(
        self,
        agent_name: str,
        instructions: str,
        user_message: str,
        sender: str,
        conversation_snippet: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 512,
        previous_response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
        if self.dummy_mode:
            simulated = self._simulate_response(agent_name, sender, user_message)
            return simulated, None, []
        return await asyncio.to_thread(
            self._stream_response,
            agent_name,
            instructions,
            user_message,
            sender,
            conversation_snippet,
            tools,
            model or self.default_model,
            temperature,
            max_output_tokens,
            previous_response_id,
            metadata or {},
        )

    def _simulate_response(self, agent_name: str, sender: str, user_message: str) -> str:
        _ = user_message  # placeholder to mirror real signature usage
        return f"Understood, {sender}. Logging the update and continuing with my agenda."

    def _stream_response(
        self,
        agent_name: str,
        instructions: str,
        user_message: str,
        sender: str,
        conversation_snippet: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        model: str,
        temperature: float,
        max_output_tokens: int,
        previous_response_id: Optional[str],
        metadata: Dict[str, Any],
    ) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
        request_body: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "instructions": instructions,
            "input": [],
            "tools": tools or [],
            "tool_choice": "auto",
            "user": f"robits_{agent_name}",
        }
        if conversation_snippet:
            request_body["input"].append(
                {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Recent context:\n{conversation_snippet}",
                        }
                    ],
                }
            )
        request_body["input"].append(
            {
                "type": "message",
                "role": "user",
                "name": sender,
                "content": [
                    {
                        "type": "input_text",
                        "text": user_message,
                    }
                ],
            }
        )
        headers: Dict[str, str] = {}
        api_key = os.getenv("OPENAI_API_KEY")
        if "api.openai.com" in self.base_url and api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["OpenAI-Beta"] = "assistants=v2"
        if previous_response_id:
            request_body["previous_response_id"] = previous_response_id
        if metadata:
            request_body["metadata"] = metadata

        response_text = ""
        response_id: Optional[str] = None
        tool_calls: List[Dict[str, Any]] = []
        import httpx

        with httpx.Client(timeout=self.timeout) as client:
            url = f"{self.base_url}/responses"
            with client.stream("POST", url, json=request_body) as stream:
                for raw_chunk in stream.iter_bytes():
                    if not raw_chunk:
                        continue
                    chunk = raw_chunk.decode("utf-8")
                    if chunk.startswith("data: "):
                        chunk = chunk[6:]
                    chunk = chunk.strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    event_type = data.get("type")
                    if event_type == "response.created":
                        response_payload = data.get("response") or {}
                        if isinstance(response_payload, dict):
                            response_id = response_payload.get("id", response_id)
                    elif event_type == "response.completed":
                        response_payload = data.get("response") or {}
                        outputs = []
                        if isinstance(response_payload, dict):
                            outputs = response_payload.get("output", []) or []
                        for output in outputs:
                            if output.get("type") == "message":
                                for part in output.get("content", []):
                                    if part.get("type") == "output_text":
                                        response_text += part.get("text", "")
                    elif event_type == "response.output_text.delta":
                        delta = data.get("delta") or {}
                        if isinstance(delta, dict):
                            response_text += delta.get("text", "") or ""
                    elif event_type == "response.error":
                        error = data.get("error") or {}
                        if isinstance(error, dict):
                            response_text += f"\n[error] {error.get('message', 'Unknown error')}"
                    elif event_type == "response.failed":
                        error = data.get("error") or {}
                        if isinstance(error, dict):
                            response_text += f"\n[failed] {error.get('message', 'Unknown failure')}"
                    elif event_type == "response.tool_calls.created":
                        tool_call = data.get("tool_call")
                        if tool_call is None:
                            tool_call = data.get("tool_calls")
                        if isinstance(tool_call, dict):
                            tool_calls.append(tool_call)
                        else:
                            # Some runtimes send a list under tool_calls; normalize to dict entries
                            for entry in (tool_call or []):
                                if isinstance(entry, dict):
                                    tool_calls.append(entry)
                    elif event_type == "response.function_call_arguments.done":
                        identifier = data.get("id")
                        for call in tool_calls:
                            if not isinstance(call, dict):
                                continue
                            if call.get("id") == identifier:
                                try:
                                    call["arguments"] = json.loads(data.get("arguments", "{}"))
                                except json.JSONDecodeError:
                                    call["arguments"] = {}
                                break
        return response_text.strip(), response_id, tool_calls
