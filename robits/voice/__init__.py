"""Phone calls with robits personas (optional; ``pip install -r requirements-voice.txt``).

A *line* is a phone number for an organisation's personas: either one persona
(``mode: persona``) or a group call where the caller talks to several personas
at once (``mode: group``) and addresses one by name ("Alex, what's the lead?").
Each persona speaks with its own ``voice`` from the persona file.

Layers, each usable without the next:

- ``lines``: the line file (which personas answer which line).
- ``room``: a call's conversation as a robits ``Session`` (no audio, no Pipecat):
  the caller's words in, persona replies out. Unit-testable with fake roles.
- ``speech``: making a model reply fit to be spoken.
- ``pipeline``: Pipecat pieces: a brain processor that runs the room, and a TTS
  service that speaks each reply in its persona's voice.
- ``server``: an HTTP server a telephony connector posts WebRTC offers, hangups
  and texts to (``python -m robits.voice``).

Speech services are any OpenAI-compatible ``/v1/audio/transcriptions`` and
``/v1/audio/speech`` endpoints; see docs/voice.md.
"""
