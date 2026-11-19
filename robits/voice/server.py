"""HTTP server for robits phone lines: WebRTC offers, hangups and texts from a telephony connector.

    python -m robits.voice            (ROBITS_VOICE_PORT, default 7860)

Routes (the connector picks the line with ``?line=<name>``; one connector per line):

- ``POST /api/offer?line=``  {sdp, type: "offer", request_data: {ice_servers, ice_transport_policy}}
  -> {sdp, type: "answer"}; 429 when the server already has ``ROBITS_VOICE_MAX_CALLS`` calls, so the
  connector can say busy. With ``ice_transport_policy: "relay"`` the answer gathers relay
  candidates only, on the TURN servers given, and an answer with any other candidate is refused (503).
- ``POST /api/hangup``  {type, call_id}: ends the call (connectors don't name the call in their
  offers, so with one call at a time it is that call; with several, all of them).
- ``POST /api/text?line=``  {text_id, conversation, from, body} -> {body}: a text answered by the
  line's personas (in memory per conversation, forgotten after ``ROBITS_VOICE_TEXT_FORGET_S``).
- ``GET /health``.

Listens on 127.0.0.1 unless ``ROBITS_VOICE_HOST`` says otherwise: put the connector next to it.
Never logged: SDP, TURN credentials, or what callers and personas say.
"""
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

HOST = os.environ.get("ROBITS_VOICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("ROBITS_VOICE_PORT", "7860"))
LINES_FILE = os.environ.get("ROBITS_VOICE_LINES", "lines.yaml")
MAX_CALLS = int(os.environ.get("ROBITS_VOICE_MAX_CALLS", "1"))
STT_URL = os.environ.get("ROBITS_VOICE_STT_URL") or os.environ.get("OPENAI_BASE_URL")
STT_MODEL = os.environ.get("ROBITS_VOICE_STT_MODEL", "whisper-1")
STT_TERMS = os.environ.get("ROBITS_VOICE_STT_TERMS", "")   # extra words for speech recognition, comma-separated
TTS_URL = os.environ.get("ROBITS_VOICE_TTS_URL") or os.environ.get("OPENAI_BASE_URL")
TTS_MODEL = os.environ.get("ROBITS_VOICE_TTS_MODEL", "tts-1")
TTS_VOICE = os.environ.get("ROBITS_VOICE_TTS_VOICE", "alloy")
TTS_RATE = int(os.environ.get("ROBITS_VOICE_TTS_SAMPLE_RATE", "24000"))
API_KEY = os.environ.get("ROBITS_VOICE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "unused"
# Silence that ends the caller's turn: Pipecat's 0.2 s cuts "Hi Alex, ... what's new?" at the comma,
# and each piece then costs a persona's whole turn.
PAUSE_S = float(os.environ.get("ROBITS_VOICE_PAUSE_S", "0.7"))
TEXT_FORGET_S = float(os.environ.get("ROBITS_VOICE_TEXT_FORGET_S", "1800"))
TEXT_SECONDS = float(os.environ.get("ROBITS_VOICE_TEXT_SECONDS", "45"))
TEXT_CHARS = 1000
OFFER_SECONDS = 8

# robits' SQLite memory store belongs to the thread that opened it, so everything robits does
# (configuration, sessions, turns) runs on this one thread; it also keeps one model call at a time.
ROBITS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robits")


async def in_robits(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(ROBITS, fn, *args)


CALLS = {}       # pc id -> {"conn", "task", "room", "line"}
TEXTS = {}       # (line, conversation) -> [last used, Room]
LINES = {}


def configure():
    """Configure robits (personas, memory, model) and load the line file."""
    import main  # noqa: F401  (robits' configuration is set up on import)
    from robits.core.config import _config
    from robits.voice import lines
    LINES.clear()
    LINES.update(lines.load(LINES_FILE, _config.persona_entries or {}))
    logger.info(f"robits voice: lines {', '.join(f'{n} ({l.mode})' for n, l in LINES.items())}")


# --- WebRTC ------------------------------------------------------------------------------------

def turn_servers(ice_servers):
    import pipecat.transports.smallwebrtc.connection as sw
    servers = []
    for s in ice_servers if isinstance(ice_servers, list) else []:
        urls = s.get("urls") if isinstance(s, dict) else None
        urls = [u for u in ([urls] if isinstance(urls, str) else urls or []) if isinstance(u, str)
                and u.split(":", 1)[0] in ("turn", "turns")]
        if urls and isinstance(s.get("username"), str) and isinstance(s.get("credential"), str):
            servers.append(sw.IceServer(urls=urls, username=s["username"], credential=s["credential"]))
    return servers


def connection(request_data):
    """A SmallWebRTCConnection for this offer: relay-only on the offer's TURN servers when asked."""
    import pipecat.transports.smallwebrtc.connection as sw
    relay = request_data.get("ice_transport_policy") == "relay"
    if not relay:
        return sw.SmallWebRTCConnection(), False
    servers = turn_servers(request_data.get("ice_servers"))
    if not servers:
        raise ValueError("relay-only was asked for, but the offer has no TURN servers")
    from aiortc import RTCIceTransportPolicy

    class RelayOnly(sw.SmallWebRTCConnection):
        def _initialize(self):
            # _initialize builds RTCConfiguration(...) and the peer connection synchronously;
            # add the relay policy for that one construction.
            original = sw.RTCConfiguration
            sw.RTCConfiguration = lambda **kw: original(iceTransportPolicy=RTCIceTransportPolicy.RELAY, **kw)
            try:
                super()._initialize()
            finally:
                sw.RTCConfiguration = original

    return RelayOnly(ice_servers=servers), True


def only_relay(sdp):
    kinds = [line.split(" typ ", 1)[1].split()[0] for line in sdp.splitlines()
             if line.startswith("a=candidate:") and " typ " in line]
    return bool(kinds) and all(k == "relay" for k in kinds)


async def run_call(key, conn, line):
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (LLMContextAggregatorPair,
                                                                       LLMUserAggregatorParams)
    from pipecat.services.openai.stt import OpenAISTTService
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
    from pipecat.workers.runner import WorkerRunner

    from robits.voice.pipeline import Brain, PersonaTTS
    from robits.voice.room import Room
    from robits.voice.speech import parse_terms, stt_prompt

    room = await in_robits(Room, line)
    CALLS[key]["room"] = room
    # Speech recognition hints: the personas' names, full and first (a short first name alone is easy to
    # mishear), then ROBITS_VOICE_STT_TERMS.
    full = [getattr(room.session.participants[k], "full_name", None) or k for k in line.personas]
    names = sorted(set(full) | {n.split()[0] for n in full})
    transport = SmallWebRTCTransport(webrtc_connection=conn,
                                     params=TransportParams(audio_in_enabled=True, audio_out_enabled=True))
    stt = OpenAISTTService(base_url=STT_URL, api_key=API_KEY,
                           settings=OpenAISTTService.Settings(model=STT_MODEL, prompt=stt_prompt(names, parse_terms(STT_TERMS))))
    tts = PersonaTTS(base_url=TTS_URL, api_key=API_KEY, sample_rate=TTS_RATE,
                     settings=PersonaTTS.Settings(model=TTS_MODEL, voice=TTS_VOICE))
    user_aggregator, _ = LLMContextAggregatorPair(
        LLMContext(), user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=PAUSE_S))))
    brain = Brain(room, run=in_robits)
    worker = PipelineWorker(Pipeline([transport.input(), stt, user_aggregator, brain, tts, transport.output()]),
                            params=PipelineParams(enable_metrics=False))
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    started = time.monotonic()

    @transport.event_handler("on_client_connected")
    async def connected(_transport, _client):
        logger.info(f"robits voice: call on line {line.name} connected")
        await brain.greet()

    @transport.event_handler("on_client_disconnected")
    async def disconnected(_transport, _client):
        await runner.cancel()

    async def limit():
        if line.max_seconds:
            await asyncio.sleep(line.max_seconds)
            logger.info(f"robits voice: call on line {line.name} reached {line.max_seconds} s")
            await runner.cancel()

    limiter = asyncio.create_task(limit())
    try:
        await runner.run()
    finally:
        limiter.cancel()
        room.close()
        CALLS.pop(key, None)
        try:
            await conn.disconnect()
        except Exception:
            pass
        logger.info(f"robits voice: call on line {line.name} ended after {time.monotonic() - started:.0f} s")


# --- texts -------------------------------------------------------------------------------------

def text_reply(line, conversation, sender, body):
    """A text answered by the line's personas; a group line signs each reply with its persona."""
    from robits.voice.room import Room
    now = time.monotonic()
    for k in [k for k, (used, _) in TEXTS.items() if now - used > TEXT_FORGET_S]:
        TEXTS.pop(k)[1].close()
    entry = TEXTS.get((line.name, conversation))
    if entry is None:
        entry = TEXTS[(line.name, conversation)] = [now, Room(line, caller=(sender or "caller")[:80])]
    entry[0] = now
    replies = entry[1].say(body)
    if line.group:
        text = " ".join(f"{r.name.split()[0]}: {r.text}" for r in replies)
    else:
        text = " ".join(r.text for r in replies)
    return text[:TEXT_CHARS]


# --- app ---------------------------------------------------------------------------------------

def app():
    from fastapi import FastAPI, HTTPException, Request
    api = FastAPI(title="robits voice")
    text_lock = asyncio.Lock()

    def line_for(request):
        name = request.query_params.get("line") or (next(iter(LINES)) if len(LINES) == 1 else None)
        if name not in LINES:
            raise HTTPException(404, "no such line")
        return LINES[name]

    @api.get("/health")
    async def health():
        return {"lines": sorted(LINES), "calls": len(CALLS)}

    @api.post("/api/offer")
    async def offer(request: Request):
        line = line_for(request)
        if len(CALLS) >= MAX_CALLS:
            raise HTTPException(429, "busy")
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("sdp"), str) or body.get("type") != "offer":
            raise HTTPException(400, "expected {sdp, type: offer}")
        request_data = body.get("request_data") if isinstance(body.get("request_data"), dict) else {}
        try:
            conn, relay = connection(request_data)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        try:
            await asyncio.wait_for(conn.initialize(sdp=body["sdp"], type="offer"), OFFER_SECONDS)
            answer = conn.get_answer()
            if not answer or (relay and not only_relay(answer["sdp"])):
                raise RuntimeError("no relay-only answer (TURN unreachable?)")
        except Exception as exc:
            logger.info(f"robits voice: offer on line {line.name} refused ({type(exc).__name__})")
            try:
                await conn.disconnect()
            except Exception:
                pass
            raise HTTPException(503, "no answer")
        key = answer.get("pc_id") or id(conn)
        CALLS[key] = {"conn": conn, "line": line.name}
        CALLS[key]["task"] = asyncio.create_task(run_call(key, conn, line))
        return {"sdp": answer["sdp"], "type": answer["type"]}

    @api.post("/api/hangup")
    async def hangup(request: Request):
        for call in list(CALLS.values()):
            task = call.get("task")
            if task:
                task.cancel()
        return {"ok": True}

    @api.post("/api/text")
    async def text(request: Request):
        line = line_for(request)
        body = await request.json()
        words = body.get("body") if isinstance(body, dict) else None
        conversation = body.get("conversation") if isinstance(body, dict) else None
        if not isinstance(words, str) or not words.strip() or not isinstance(conversation, str):
            raise HTTPException(400, "expected {conversation, body}")
        sender = body.get("from") if isinstance(body.get("from"), str) else ""
        async with text_lock:   # one text at a time on the shared model
            try:
                reply = await asyncio.wait_for(
                    in_robits(text_reply, line, conversation, sender, words), TEXT_SECONDS)
            except Exception as exc:
                logger.info(f"robits voice: text on line {line.name} unanswered ({type(exc).__name__})")
                raise HTTPException(503, "no reply")
        if not reply:
            raise HTTPException(503, "no reply")
        return {"body": reply}

    return api


def main():
    import uvicorn
    ROBITS.submit(configure).result()
    uvicorn.run(app(), host=HOST, port=PORT, log_level="warning")
