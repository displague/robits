from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Message:
    sender: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    id: Optional[int] = None


class ConversationState:
    """Track the shared conversation timeline for all agents."""

    def __init__(self) -> None:
        self._messages: List[Message] = []
        self._lock = asyncio.Lock()
        self._next_id = 1

    async def append(self, message: Message) -> Message:
        async with self._lock:
            message.id = self._next_id
            self._next_id += 1
            self._messages.append(message)
            return message

    async def extend(self, messages: Iterable[Message]) -> List[Message]:
        appended: List[Message] = []
        for message in messages:
            appended.append(await self.append(message))
        return appended

    async def history(self, limit: Optional[int] = None) -> List[Message]:
        async with self._lock:
            if limit is None or limit >= len(self._messages):
                return list(self._messages)
            return self._messages[-limit:]

    async def tail_text(self, limit: int = 8) -> str:
        entries = await self.history(limit)
        lines = []
        for message in entries:
            prefix = message.metadata.get("internal_prefix", message.sender)
            lines.append(f"{prefix}: {message.content}")
        return "\n".join(lines)

    async def clear(self) -> None:
        async with self._lock:
            self._messages.clear()
            self._next_id = 1
