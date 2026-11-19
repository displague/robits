# Phone lines (`robits.voice`)

People can call an organisation's personas. A **line** is either one persona or a group call
with several. Each persona speaks in its own voice. The call is an ordinary robits session holding
the line's personas, so they answer with their own prompts, memories, briefings and tools, and the
call is recorded in memory like any other session.

```
caller ── telephony connector ── robits.voice server ── Pipecat: STT → turns (VAD) → Brain → TTS
                                                                            │
                                                               robits Session (the line's personas)
```

Robits doesn't do telephony. A connector (a SIP or WebRTC gateway's client) posts each call's
WebRTC offer to this server and plays the answer back to the caller. One connector per line.

## Install

```bash
pip install -r requirements-voice.txt     # Pipecat (WebRTC, Silero VAD), FastAPI, uvicorn
```

Container: `docker build --build-arg EXTRAS="-r requirements-voice.txt" -t robits-voice .`, run with
`python -m robits.voice` as the command.

## Personas and lines

Give each persona a voice from your TTS server's catalogue:

```yaml
- username: alex_chen
  full_name: Alex Chen
  role: SE
  voice: nova
```

The line file (`ROBITS_VOICE_LINES`, default `lines.yaml`):

```yaml
lines:
  team:
    mode: group                  # several personas on one call
    personas: [alex_chen, sam_ortiz]
    greeting: "Team, Alex and Sam here."
    max_seconds: 900             # optional: end the call after this long
    handoffs: 0                  # optional: times a persona may pass a question on (0 to 3)
    heard_as:                    # optional: what speech recognition writes for a name
      sam_ortiz: [Psalm, Sam Morty's]
  editor:
    mode: persona                # one persona
    persona: sam_ortiz
default_voice: alloy             # personas without a voice of their own
```

Who answers on a group line:

1. the persona the caller names first ("Sam, ...", "@sam_ortiz ...");
2. else the only persona the caller mentions ("what does Sam think?");
3. else whoever spoke last (a call is a conversation), and the line's first persona at the start.

Personas speak only to the caller, and a name a reply opens with ("Alex Chen, ...") is dropped.
With `handoffs` above 0, a persona that starts its reply with another persona's name and a
question ("Sam, can you take this one?") hands the question on, and the other persona answers in
the same turn. It's off by default: small models pass questions back and forth instead of
answering.

Speech recognition often mishears an unusual name, so naming a persona doesn't reach them. List
what it writes instead under `heard_as`: those words are replaced by the persona's first name,
whole words and any case, before choosing who answers. Find them by playing the name through
your text-to-speech and speech-to-text, and leave out ordinary words ("some" for "Sam"), since
they are replaced wherever they appear.

Each reply costs a model call on the whole prompt, so keep calls short with
`ROBITS_MAX_CONTEXT_TOKENS` (robits' own limit: older turns are left out of the prompt) and, for
small local models, `ROBITS_PROMPT_PROFILE=compact`.

Every prompt on a call starts with a short note that the persona is on a phone call. It asks for
one to three spoken sentences and, on a group line, names the others. Replies are cleaned for
speech: no markdown, links or code, and at most 600 characters.

### Spoken as it is written

On a call, a persona's reply is spoken sentence by sentence while the model is still writing it, so
the caller hears the first sentence after its first full stop instead of after the whole reply. This
needs a model server that streams chat completions; roles without a call listener never stream.

- A sentence ends at `.`, `!` or `?` followed by a space, except after a short abbreviation ("Gov.
  Ruiz", "Dr. Lee") or an initial ("U.S."). The unfinished tail is spoken when the reply, or the text
  before a tool call, ends.
- When the persona runs a tool before it has said anything, the caller hears "Let me check." once
  for that turn, in the persona's voice. After a tool call the persona goes on in the same turn, and
  its next round is streamed the same way.
- A reply that starts like a structured action, markup or code (`{`, `[`, `<` or a backtick) is not
  streamed: it is handled whole at the end of the turn, as before. Inline `<tool_call>` text from
  models that write calls in their reply is never spoken, though the sentences before it are.
- An opening name on the first sentence is dropped (as in "Alex Chen, ..."), and the whole spoken
  reply stays within the 600-character limit; a sentence that would pass it is not spoken.
- A hand-off to another persona (`handoffs`) still happens after the first persona's turn; each
  persona's sentences are spoken in its own voice, in order.
- Texts (`/api/text`) are not streamed: the answer is returned whole.
- The server logs how long the personas took and, when it streamed, how soon the first words came:
  `robits voice: 12 words -> alex_chen in 4.2 s (first words after 1.9 s)`. Words and names of
  callers are not logged.

## Server

`python -m robits.voice` listens on `ROBITS_VOICE_HOST:ROBITS_VOICE_PORT` (default
`127.0.0.1:7860`; run the connector beside it).

| Route | Body | Answer |
|---|---|---|
| `POST /api/offer?line=<name>` | `{sdp, type: "offer", request_data: {ice_servers, ice_transport_policy}}` | `{sdp, type: "answer"}`; `429` when busy, `503` when no acceptable answer |
| `POST /api/hangup` | `{type, call_id}` | ends the call |
| `POST /api/text?line=<name>` | `{text_id, conversation, from, body}` | `{body}` (at most 1,000 characters); a group line signs each reply with the persona's first name |
| `GET /health` | | `{lines, calls}` |

`?line=` can be left out when there is only one line.

With `ice_transport_policy: "relay"`, the answer gathers relay candidates only, on the TURN servers
in `ice_servers`. An answer that has any other kind of candidate is refused. This keeps the server's
own addresses away from callers.

Connectors don't name the call in an offer. So `/api/hangup` ends every call the server has, which
is the one call at the default `ROBITS_VOICE_MAX_CALLS=1`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ROBITS_VOICE_LINES` | `lines.yaml` | the line file |
| `ROBITS_VOICE_HOST` / `ROBITS_VOICE_PORT` | `127.0.0.1` / `7860` | where the server listens |
| `ROBITS_VOICE_MAX_CALLS` | `1` | calls at once; more get `429` |
| `ROBITS_VOICE_STT_URL` | `OPENAI_BASE_URL` | OpenAI-compatible `/v1/audio/transcriptions` |
| `ROBITS_VOICE_STT_MODEL` | `whisper-1` | its model |
| `ROBITS_VOICE_TTS_URL` | `OPENAI_BASE_URL` | OpenAI-compatible `/v1/audio/speech` (raw PCM) |
| `ROBITS_VOICE_TTS_MODEL` / `ROBITS_VOICE_TTS_VOICE` | `tts-1` / `alloy` | its model and default voice |
| `ROBITS_VOICE_TTS_SAMPLE_RATE` | `24000` | the PCM rate the TTS returns |
| `ROBITS_VOICE_API_KEY` | `OPENAI_API_KEY` | key for both speech endpoints |
| `ROBITS_VOICE_PAUSE_S` | `0.7` | silence that ends the caller's turn (shorter cuts questions at commas) |
| `ROBITS_VOICE_TEXT_FORGET_S` | `1800` | a text conversation is forgotten after this many idle seconds |

Everything robits does (opening memory, sessions, model turns) runs on one thread, because the
SQLite memory store belongs to the thread that opened it. Calls and texts therefore take turns
with the model, which also suits a single local model.

Caller turns that arrive within 0.8 s of each other, or while the personas are still thinking, are
answered together. While the personas think, the caller hears a soft two-tone blip every 1.6 s instead of silence.

## Not logged

SDP, TURN credentials, and what callers and personas say. The words are kept in robits' memory,
like any other session.
