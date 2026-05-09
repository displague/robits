from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import dataclass
from typing import Optional

try:
    from termcolor import colored
except ModuleNotFoundError:  # pragma: no cover - cosmetic fallback
    def colored(text: str, *_args: object, **_kwargs: object) -> str:
        return text


@dataclass
class ManagedProcess:
    name: str
    process: asyncio.subprocess.Process

    async def stop(self) -> None:
        if self.process.returncode is not None:
            return
        try:
            self.process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()


async def maybe_launch_from_env(prefix: str) -> Optional[ManagedProcess]:
    command = os.getenv(f"{prefix}_COMMAND")
    if not command:
        return None
    args = shlex.split(command)
    if not args:
        return None
    cwd = os.getenv(f"{prefix}_CWD") or None
    env = os.environ.copy()
    delay = float(os.getenv(f"{prefix}_READY_DELAY", "0"))
    name = os.getenv(f"{prefix}_NAME", prefix.lower())

    print(colored(f"Starting {name} via: {' '.join(args)}", "magenta"))
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env=env,
        )
    except FileNotFoundError:
        print(colored(f"Failed to launch {name}: command not found", "red"))
        return None

    if delay > 0:
        await asyncio.sleep(delay)
    return ManagedProcess(name=name, process=process)
