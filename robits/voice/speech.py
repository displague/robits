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
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
_MARKS = re.compile(r"[*_`#>|~]+")
_TAG = re.compile(r"<[^>]{1,40}>")
_SPACE = re.compile(r"\s+")


def speakable(text, limit=MAX_CHARS):
    """Plain sentences from a model reply: no code, markdown, links or tags; at most ``limit``
    characters, cut at the last sentence end that fits."""
    if not text:
        return ""
    text = _FENCE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _BULLET.sub("", text)
    text = _TAG.sub(" ", text)
    text = _MARKS.sub("", text)
    text = _SPACE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > limit // 3 else cut.rsplit(" ", 1)[0] + "."
