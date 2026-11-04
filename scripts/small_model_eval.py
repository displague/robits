"""Small-model eval for robits personas.

    OPENAI_BASE_URL=http://127.0.0.1:11434/v1/ python scripts/small_model_eval.py . <model> [reps] \
        ["profile=compact,effort=none,tools=agent.think|agent.wait|memory.search"] [scenario,...]

Runs four fixed scenarios (answer from given facts, take a break with agent.wait, a short
status line, recall the previous turn) through robits' real interact() and Chat Completions
path, and scores each reply with deterministic checks. "Clean" means no JSON/tool syntax or
labels in the reply, not empty (except after agent.wait), and no request for more context.
"""
import os
import json
import re
import sys
import time

sys.path.insert(0, sys.argv[1])
MODEL = sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
# variant: comma list of profile=..., effort=..., tools=a|b|c
VARIANT = dict(kv.split("=", 1) for kv in (sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else []))
PROFILE = VARIANT.get("profile", "")

from contextlib import redirect_stdout
from io import StringIO

from openai import OpenAI

import main
from robits.core.config import _config as config
from robits.core.providers import ChatCompletionsProvider

config.memory_store = None
config.max_context_tokens = 0
config.cheap_model = config.costly_model = MODEL
config.model_provider = ChatCompletionsProvider(
    OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1/"),
           api_key=os.environ.get("OPENAI_API_KEY", "not-needed"), timeout=600), main.tool_registry)
if PROFILE:
    config.prompt_profile = PROFILE
config._reasoning_effort_env = VARIANT.get("effort")

BAD = [r"agent_name", r'\{\s*"action"', r"(need|provide) (more|a) context", r"^\s*$"]


def fresh_se():
    with redirect_stdout(StringIO()):
        org = main.build_employee_dict()
        main.tool_registry.clear()
        main.load_tools(main.System(org))
    se = org["SE"]
    se.runtime_clock_state = "on"
    if VARIANT.get("tools"):
        from robits.core.roles import _apply_persona_tools
        _apply_persona_tools(se, {"tools": VARIANT["tools"].split("|")})
    from robits.core.session import RuntimeEventStream
    se.runtime_event_stream = RuntimeEventStream()
    se.runtime_session_id = "eval"
    return se


def tools_called(se):
    return [e.payload.get("tool_name") for e in se.runtime_event_stream.events()
            if e.event_type == "tool_call.requested"]


def say(se, text):
    with redirect_stdout(StringIO()):
        return se.interact("CEO", text) or ""


def hygiene(reply):
    return not any(re.search(p, reply, re.I | re.M) for p in BAD)


def scenario_fact(se):
    r = say(se, "Status from the test lab: the payments suite is flaky on ARM runners; "
                "search and auth suites are green. Which suite is flaky, and on which runners?")
    return ("payment" in r.lower() and "arm" in r.lower()), r


def scenario_wait(se):
    r = say(se, "Please take a 20 minute break now.")
    return se.waiting_until is not None, r


def scenario_status(se):
    r = say(se, "Give me a one-line status update on the API migration. Facts: 3 of 5 services "
                "have moved; the blocker is token refresh in the auth service.")
    words = len(r.split())
    return (re.search(r"\b(3|three)\b", r, re.I) is not None and ("token" in r.lower() or "auth" in r.lower()) and words <= 45), r


def scenario_recall(se):
    say(se, "Heads up: the release is moving to Thursday because the database migration needs a second review.")
    r = say(se, "Remind me: why did the release move?")
    return ("migration" in r.lower() or "review" in r.lower()), r


SCENARIOS = [scenario_fact, scenario_wait, scenario_status, scenario_recall]
if len(sys.argv) > 5:
    SCENARIOS = [f for f in SCENARIOS if f.__name__ in sys.argv[5].split(",")]
totals, clean, n_replies, started = {}, 0, 0, time.time()
for scenario in SCENARIOS:
    for _ in range(REPS):
        se = fresh_se()
        ok, reply = scenario(se)
        totals.setdefault(scenario.__name__, []).append(ok)
        n_replies += 1
        clean += hygiene(reply) or (scenario is scenario_wait and ok and not reply.strip())
        print(f"  {scenario.__name__:16} {'PASS' if ok else 'fail'}  tools={tools_called(se)}  {reply[:70]!r}", flush=True)

score = {k: f"{sum(v)}/{len(v)}" for k, v in totals.items()}
passed = sum(sum(v) for v in totals.values())
se = fresh_se()
cap = {}
real = config.model_provider
config.model_provider = type("P", (), {"generate": lambda self, role, model, sender, messages: cap.update(m=messages) or "ok"})()
say(se, "size check")
config.model_provider = real
tools_json = json.dumps(main.tool_registry.as_chat_completion_tools(se))
print(json.dumps({"variant": sys.argv[4] if len(sys.argv) > 4 else "default",
                  "prompt_tokens_est": (len(cap["m"][0]["content"]) + len(tools_json)) // 4,
                  "tool_schemas": len(main.tool_registry.as_chat_completion_tools(se)), "task_pass": f"{passed}/{len(SCENARIOS) * REPS}",
                  "clean_replies": f"{clean}/{n_replies}", **score,
                  "minutes": round((time.time() - started) / 60, 1)}))
