"""Making a persona's reply fit to be spoken, and the call note its prompts carry."""
import re

MAX_CHARS = 600          # a spoken reply longer than this is cut at a sentence end

CALL_NOTE = (
    "[You are on a phone call{who}. Everything you write is spoken aloud: reply in one to three short "
    "sentences of plain speech, with no lists, markdown, links or emoji. If you don't know, say so.]"
)
GROUP_NOTE = (
    " Others on the call: {others}. Speak only as yourself, and only to the caller: never address "
    "the others or start with anyone's name."
)
HANDOFF_NOTE = (
    " Others on the call: {others}. Speak only as yourself, to the caller. Only to ask one of them a "
    "question, start your reply with their name and a comma."
)


def call_note(caller="", others=(), handoffs=False):
    who = f" with {caller}" if caller else ""
    note = CALL_NOTE.format(who=who)
    if others:
        note = note[:-1] + (HANDOFF_NOTE if handoffs else GROUP_NOTE).format(others=", ".join(others)) + "]"
    return note


_FENCE = re.compile(r"```.*?```", re.S)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_SEG = r"\w[\w.@+%-]*"
_PATH = re.compile(rf"(?<![\w/.@:-])(~?/|\.{{1,2}}/)?({_SEG}(?:/ ?/?{_SEG})+)")
_KIND = {"drafts": "draft", "draft": "draft", "records": "record", "record": "record"}
_DATE = re.compile(r"(?:^|(?<=[-_ ]))(?:\d{4}-\d{2}-\d{2}|\d{8})(?=$|[-_ ])")
_EXTENSION = re.compile(r"\.[A-Za-z]\w{0,4}$")
_DOMAIN = re.compile(r"\.[A-Za-z]{2,}$")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
_MARKS = re.compile(r"[*_`#>|~]+")
_TAG = re.compile(r"<[^>]{1,40}>")
_SPACE = re.compile(r"\s+")


def _path_words(match):
    """A file path or record key as words: "drafts/2026-11-18-vote.md" is "the vote draft". Ordinary
    slashes ("and/or", "24/7", "11/18/2026") are left alone."""
    prefix, body = match.group(1) or "", match.group(2)
    tail = re.search(r"[.]+$", body)
    tail = tail.group(0) if tail else ""
    body = body[:len(body) - len(tail)]
    parts = [x for x in re.split(r"/ ?/?", body)]
    named = prefix or _EXTENSION.search(parts[-1]) or parts[0].lower() in _KIND
    if not named:
        return match.group(0)
    if not prefix and _DOMAIN.search(parts[0]) and not _KIND.get(parts[0].lower()):
        return "a link" + tail
    kind = next((_KIND[x.lower()] for x in parts[:-1] if x.lower() in _KIND), None)
    kind = kind or ("record" if parts[0].lower() == "records" else "file")
    stem = _EXTENSION.sub("", parts[-1]) if _EXTENSION.search(parts[-1]) else parts[-1]
    if stem.lower() in _KIND and len(parts) > 1 and not _EXTENSION.search(parts[-1]):
        stem = ""
    words = _DATE.sub(" ", stem)
    words = re.sub(r"(?<=[-_])\d{6,}$|[-_.]+", " ", words)
    words = " ".join(w for w in words.split() if not w.isdigit() or len(w) < 6)
    words = words if sum(c.isalpha() for c in words) >= 3 else ""
    return (f"the {words} {kind}" if words else f"a {kind}") + tail


def speakable(text, limit=MAX_CHARS):
    """Plain sentences from a model reply: no code, markdown, links or tags; at most ``limit``
    characters, cut at the last sentence end that fits."""
    if not text:
        return ""
    text = _FENCE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _PATH.sub(_path_words, text)
    text = _BULLET.sub("", text)
    text = _TAG.sub(" ", text)
    text = _MARKS.sub("", text)
    text = _SPACE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > limit // 3 else cut.rsplit(" ", 1)[0] + "."
