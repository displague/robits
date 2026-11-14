"""The line file: which personas answer which line.

YAML (path in ``ROBITS_VOICE_LINES``)::

    lines:
      team:                  # the line's name, chosen by the connector (?line=team)
        mode: group              # several personas on one call
        personas: [alex_chen, sam_ortiz]
        greeting: "Team, Alex and Sam here."
        handoffs: 0              # optional: times a persona may pass a question on (default 0)
        heard_as:                # optional: what speech recognition writes for a persona's name
          sam_ortiz: [Psalm, Sam Morty's]
      editor:
        mode: persona            # one persona
        persona: sam_ortiz
    default_voice: alloy         # for personas without a voice of their own

A persona's own voice comes from its persona file (``voice: <name>``).
"""
from dataclasses import dataclass, field

MODES = ("persona", "group")


class LineError(ValueError):
    pass


@dataclass(frozen=True)
class Line:
    name: str
    mode: str
    personas: tuple
    greeting: str = ""
    max_seconds: int = 0          # 0: no limit of our own
    voices: dict = field(default_factory=dict)   # username -> voice, after defaults
    handoffs: int = 0             # group lines: times per caller turn a persona may pass a question on
    heard_as: tuple = ()          # (words, username) pairs, longest words first: mishearings of a name

    @property
    def group(self):
        return self.mode == "group"


def parse(data, persona_map, default_voice=None):
    """{name: Line} from the parsed YAML; raise LineError for anything malformed.

    ``persona_map`` is robits' loaded personas ({username: {..., voice}}); every
    persona a line names must be in it.
    """
    if not isinstance(data, dict) or not isinstance(data.get("lines"), dict) or not data["lines"]:
        raise LineError("the line file needs a non-empty 'lines' mapping")
    default_voice = data.get("default_voice", default_voice)
    if default_voice is not None and not isinstance(default_voice, str):
        raise LineError("default_voice must be a string")
    lines = {}
    for name, spec in data["lines"].items():
        if not isinstance(name, str) or not name.replace("-", "").replace("_", "").isalnum():
            raise LineError(f"line name {name!r}: letters, digits, '-' and '_' only")
        if not isinstance(spec, dict):
            raise LineError(f"line {name}: expected a mapping")
        mode = spec.get("mode", "persona")
        if mode not in MODES:
            raise LineError(f"line {name}: mode must be one of {', '.join(MODES)}")
        if mode == "persona":
            personas = [spec.get("persona")]
        else:
            personas = spec.get("personas")
            if not isinstance(personas, list) or not personas:
                raise LineError(f"line {name}: a group line needs a list of personas")
        for p in personas:
            if not isinstance(p, str) or p not in persona_map:
                raise LineError(f"line {name}: unknown persona {p!r}")
        if len(set(personas)) != len(personas):
            raise LineError(f"line {name}: a persona is listed twice")
        greeting = spec.get("greeting", "")
        if not isinstance(greeting, str):
            raise LineError(f"line {name}: greeting must be a string")
        max_seconds = spec.get("max_seconds", 0)
        if not isinstance(max_seconds, int) or isinstance(max_seconds, bool) or max_seconds < 0:
            raise LineError(f"line {name}: max_seconds must be a whole number of seconds")
        handoffs = spec.get("handoffs", 0)
        if not isinstance(handoffs, int) or isinstance(handoffs, bool) or not 0 <= handoffs <= 3:
            raise LineError(f"line {name}: handoffs must be 0 to 3")
        heard_as = _heard_as(name, spec.get("heard_as", {}), personas)
        voices = {}
        for p in personas:
            voice = persona_map[p].get("voice") or default_voice
            if voice is not None and not isinstance(voice, str):
                raise LineError(f"persona {p}: voice must be a string")
            voices[p] = voice
        lines[name] = Line(name, mode, tuple(personas), greeting, max_seconds, voices,
                           handoffs if mode == "group" else 0, heard_as)
    return lines


def _heard_as(line, spec, personas):
    """((words, username), ...) from {username: [words, ...]}, longest words first."""
    if not isinstance(spec, dict):
        raise LineError(f"line {line}: heard_as maps a persona to the words heard for its name")
    pairs = []
    for persona, heard in spec.items():
        if persona not in personas:
            raise LineError(f"line {line}: heard_as names {persona!r}, who isn't on the line")
        if not isinstance(heard, list) or not all(isinstance(h, str) and h.strip() and len(h) <= 40 for h in heard):
            raise LineError(f"line {line}: heard_as {persona} must be a list of words (40 characters at most)")
        pairs += [(" ".join(h.split()), persona) for h in heard]
    words = [w.lower() for w, _ in pairs]
    if len(set(words)) != len(words):
        raise LineError(f"line {line}: heard_as lists the same words twice")
    return tuple(sorted(pairs, key=lambda pair: -len(pair[0])))


def load(path, persona_map):
    import yaml
    with open(path, encoding="utf-8") as f:
        return parse(yaml.safe_load(f), persona_map)
