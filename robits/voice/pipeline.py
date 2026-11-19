"""Pipecat pieces for a robits call: the brain (a ``Room``), per-persona voices, a thinking cue.

    transport in -> STT -> user turns (VAD) -> Brain -> PersonaTTS -> transport out

There is no Pipecat LLM service: robits personas are the model. The user aggregator ends the
caller's turn and passes its context on; ``Brain`` takes the caller's newest words from it, runs
``Room.say`` in a thread, and speaks each reply as a ``PersonaSpeakFrame`` in that persona's
voice, sentence by sentence as the model writes it (``stream``, on by default), so the caller
hears the first sentence while the rest is still being written. While the personas think (a
model call, maybe tool calls) the caller hears a soft blip every ``CUE_EVERY_S`` seconds instead of silence.
"""
import asyncio
import math
import struct
import time
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (BotStartedSpeakingFrame, CancelFrame, EndFrame, Frame, LLMContextFrame,
                                   OutputAudioRawFrame, TTSAudioRawFrame, TTSSpeakFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.tts import OpenAITTSService

CUE_RATE = 24000
CUE_FIRST_S = 0.8
CUE_EVERY_S = 1.6
CUE_MAX_S = 90
MERGE_S = 0.8          # caller turns this close together are one question


@dataclass
class PersonaSpeakFrame(TTSSpeakFrame):
    """Text to speak in one persona's voice (None: the TTS service's own voice)."""

    voice: str | None = None
    persona: str = ""


class PersonaTTS(OpenAITTSService):
    """An OpenAI-compatible TTS that speaks each ``PersonaSpeakFrame`` in its own voice.

    Voices are passed through as given (a local server's catalogue, not only OpenAI's names).
    The voice is bound to the utterance's TTS context, so queued replies keep their voices.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._next_voice = None
        self._context_voices = {}

    async def process_frame(self, frame, direction):
        if isinstance(frame, PersonaSpeakFrame):
            self._next_voice = frame.voice
        await super().process_frame(frame, direction)

    async def on_turn_context_created(self, context_id):
        await super().on_turn_context_created(context_id)
        if self._next_voice:
            self._context_voices[context_id] = self._next_voice
        self._next_voice = None
        while len(self._context_voices) > 64:
            self._context_voices.pop(next(iter(self._context_voices)))

    async def run_tts(self, text, context_id):
        voice = self._context_voices.get(context_id) or self._settings.voice
        params = {"input": text, "model": self._settings.model, "voice": voice, "response_format": "pcm"}
        if self._settings.speed:
            params["speed"] = self._settings.speed
        try:
            async with self._client.audio.speech.with_streaming_response.create(**params) as r:
                if r.status_code != 200:
                    logger.error(f"{self}: TTS answered {r.status_code} for voice {voice!r}")
                    return
                await self.start_tts_usage_metrics(text)
                async for chunk in r.iter_bytes(self.chunk_size):
                    if chunk:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
        except Exception as exc:   # one failed sentence must not end the call
            logger.error(f"{self}: TTS failed ({type(exc).__name__})")


def blip(rate=CUE_RATE, level=0.07):
    """Two soft tones, 0.22 s in all: the 'still thinking' cue."""
    out = bytearray()
    for freq, secs in ((880, 0.09), (0, 0.04), (660, 0.09)):
        n = int(rate * secs)
        for i in range(n):
            fade = min(1.0, i / (rate * 0.01), (n - i) / (rate * 0.01))
            v = level * fade * math.sin(2 * math.pi * freq * i / rate) if freq else 0.0
            out += struct.pack("<h", int(v * 32767))
    return bytes(out)


def last_user_text(context):
    """The newest user message's text in an LLM context ('' if none)."""
    for m in reversed(context.get_messages() if hasattr(context, "get_messages") else context.messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return (c or "").strip()
    return ""


class Brain(FrameProcessor):
    """Answers each caller turn from a ``Room``; plays the thinking cue while it waits."""

    def __init__(self, room, cue=True, run=None, stream=True, **kwargs):
        """``run(fn, *args)`` runs a blocking call off the event loop (default: asyncio.to_thread).
        ``stream``: speak each sentence as soon as it is written rather than whole replies."""
        super().__init__(**kwargs)
        self.room = room
        self.run = run or asyncio.to_thread
        self.cue = cue
        self.stream = stream
        self._thinking = 0
        self._cue_task = None
        self._tasks = set()
        self._seen = 0          # user messages already answered
        self._pending = []      # caller turns not answered yet
        self._heard = asyncio.Event()
        self._worker = None

    async def greet(self):
        line = self.room.line
        if line.greeting:
            first = line.personas[0]
            await self.push_frame(PersonaSpeakFrame(text=line.greeting, voice=line.voices.get(first), persona=first))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            if getattr(frame, "speculation", False):
                return
            users = [m for m in frame.context.get_messages() if isinstance(m, dict) and m.get("role") == "user"]
            if len(users) <= self._seen:
                return
            self._seen = len(users)
            text = last_user_text(frame.context)
            if text:
                self._pending.append(text)
                self._heard.set()
                if self._worker is None or self._worker.done():
                    self._worker = asyncio.create_task(self._work())
                    self._tasks.add(self._worker)
                    self._worker.add_done_callback(self._tasks.discard)
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            self._stop_cue()
        if isinstance(frame, (EndFrame, CancelFrame)):
            self._stop_cue()
            for task in list(self._tasks):
                task.cancel()
        await self.push_frame(frame, direction)

    async def _work(self):
        """Answer what the caller said, one turn at a time. Turns that arrive within ``MERGE_S`` of each
        other, or while the personas are still thinking, are answered together: speech detection
        often splits one question at a pause ("Hi Alex," ... "what are you working on?")."""
        while self._pending:
            while True:
                self._heard.clear()
                try:
                    await asyncio.wait_for(self._heard.wait(), MERGE_S)
                except asyncio.TimeoutError:
                    break
            text = " ".join(self._pending)
            self._pending.clear()
            await self._answer(text)

    async def _answer(self, text):
        self._thinking += 1
        self._start_cue()
        started = time.monotonic()
        first = []
        loop = asyncio.get_running_loop()

        def speak(reply):
            # Called from the thread running Room.say: wait until the frame is queued, so sentences
            # keep their order.
            if not first:
                first.append(time.monotonic() - started)
            asyncio.run_coroutine_threadsafe(self.push_frame(PersonaSpeakFrame(
                text=reply.text, voice=reply.voice, persona=reply.persona)), loop).result()

        try:
            replies = await (self.run(self.room.say, text, speak) if self.stream else self.run(self.room.say, text))
            # Who answered and how long it took; never the words.
            logger.info(f"robits voice: {len(text.split())} words -> {', '.join(r.persona for r in replies) or 'no reply'} "
                        f"in {time.monotonic() - started:.1f} s"
                        + (f" (first words after {first[0]:.1f} s)" if first else ""))
        except Exception as exc:
            logger.error(f"robits voice: the personas failed to answer ({type(exc).__name__}: {exc})")
            replies = []
            first = self.room.line.personas[0]
            await self.push_frame(PersonaSpeakFrame(text="Sorry, I lost my train of thought. Could you say that again?",
                                                    voice=self.room.line.voices.get(first), persona=first))
        finally:
            self._thinking -= 1
            if not self._thinking:
                self._stop_cue()
        for reply in replies:
            if not reply.spoken:
                await self.push_frame(PersonaSpeakFrame(text=reply.text, voice=reply.voice, persona=reply.persona))

    def _start_cue(self):
        if self.cue and (self._cue_task is None or self._cue_task.done()):
            self._cue_task = asyncio.create_task(self._play_cue())

    def _stop_cue(self):
        if self._cue_task is not None:
            self._cue_task.cancel()
            self._cue_task = None

    async def _play_cue(self):
        sound, start = blip(), time.monotonic()
        try:
            await asyncio.sleep(CUE_FIRST_S)
            while self._thinking and time.monotonic() - start < CUE_MAX_S:
                await self.push_frame(OutputAudioRawFrame(audio=sound, sample_rate=CUE_RATE, num_channels=1))
                await asyncio.sleep(CUE_EVERY_S)
        except asyncio.CancelledError:
            pass
