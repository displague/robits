"""Seed a fresh robits SQLite memory DB for single-persona memory testing.

Usage:
    python scripts/seed_test_db.py [db_path]

Default db_path: var/test-memory.db

Creates:
  - Session record "seed-session"
  - alex_chen identity digest (via persona seeding)
  - Five granular thought records covering specific past work
  - One episodic digest that summarises those thoughts, with each thought
    as a distinct source ref — enabling memory.expand_digest to reveal the
    granular detail behind the summary

The DB is fully self-contained. Point ROBITS_MEMORY_DB at it and run with
ROBITS_PERSONAS_FILE=personas-test-alex.yaml to skip re-seeding at startup
(seed_persona skips if any memory already exists for the agent).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robits.memory.sqlite import SQLiteMemoryStore

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "var/test-memory.db"
AGENT = "alex_chen"
SESSION = "seed-session"

os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)

store = SQLiteMemoryStore(DB_PATH)
store.upsert_agent(AGENT, role="SE", full_name="Alex Chen", username=AGENT)
store.create_session(SESSION)

print(f"Seeding {DB_PATH} for agent {AGENT!r} ...")

# --- identity digest ---
identity_content = (
    "Alex Chen is a pragmatic backend engineer with deep Python experience who values "
    "clean APIs, strong test coverage, and minimal abstractions. Has led the "
    "payment-gateway integration project and billing-service reliability work. "
    "Enjoys cooking Italian food and outdoor running outside of work hours. "
    "Current focus: delivering the Stripe integration performance report."
)
store.seed_memory_digest(
    "identity",
    identity_content,
    agent_id=AGENT,
    session_id=SESSION,
    relationship_type="personal",
)
print("  identity digest seeded")

# --- granular work thoughts ---
thoughts = [
    (
        "Completed Stripe payment-gateway integration on 2026-04-10. "
        "Used idempotency keys (header: Idempotency-Key) on POST /v1/charges to make "
        "retries safe under network failures. Latency p99 is 320 ms."
    ),
    (
        "Fixed webhook HMAC validation bug in billing-service (commit a3f9b1c, 2026-04-08). "
        "Root cause: STRIPE_WEBHOOK_SECRET env var was missing in staging, so the signature "
        "check was silently skipped. Added a startup assertion to fail fast if the secret is absent."
    ),
    (
        "CEO requested a performance report on the payment system by 2026-04-18 (end of sprint). "
        "Key metrics to include: throughput (req/s), p50/p95/p99 latency, error rate, "
        "and webhook delivery success rate."
    ),
    (
        "Jordan (HR) mentioned new engineer Priya Patel starts 2026-04-14. "
        "Need to set up her dev environment: clone payment-gateway repo, provision a Stripe test account, "
        "and add her to the #payments Slack channel."
    ),
    (
        "Investigated memory leak in the billing-worker process (2026-04-05). "
        "Found that SQLAlchemy sessions were not being closed after each job. "
        "Fixed with a context manager; worker RSS now stays flat over 24 h."
    ),
]

thought_ids = []
for content in thoughts:
    tid = store.append_thought(
        agent_id=AGENT,
        content=content,
        session_id=SESSION,
        visibility="private",
        source="seed_script",
        relationship_type="work",
    )
    thought_ids.append(tid)
    print(f"  thought {tid}: {content[:60]}...")

# --- episodic digest referencing all five thoughts ---
episodic_summary = (
    "Work summary — April 2026:\n"
    "• Completed Stripe payment-gateway integration (2026-04-10): idempotency keys, p99 latency 320 ms.\n"
    "• Fixed billing-service webhook HMAC validation bug (commit a3f9b1c, 2026-04-08).\n"
    "• CEO performance report due 2026-04-18: throughput, latency p50/p95/p99, error rate, webhook delivery.\n"
    "• Priya Patel joins 2026-04-14 — needs dev env, Stripe test account, #payments Slack access.\n"
    "• Fixed billing-worker memory leak (2026-04-05): SQLAlchemy sessions not closed; resolved with context manager."
)
source_refs = [{"source_table": "thoughts", "source_id": tid} for tid in thought_ids]
digest_id = store.append_memory_digest(
    content=episodic_summary,
    source_refs=source_refs,
    agent_id=AGENT,
    session_id=SESSION,
    digest_type="episodic",
    accessibility="agent",
    system_only=False,
    source="seed_script",
    relationship_type="work",
)
print(f"  episodic digest {digest_id} created, sources: {thought_ids}")

print(f"\nDone. Run with:")
print(f"  ROBITS_MEMORY_DB={DB_PATH} ROBITS_PERSONAS_FILE=personas-test-alex.yaml \\")
print(f"  ROBITS_SPAWN_MODEL=functiongemma-270m-it ROBITS_PROVIDER_API=chat \\")
print(f"  ROBITS_LOOP_DETECT_THRESHOLD=3 ROBITS_DIGEST_INTERVAL=2 \\")
print(f"  PYTHONUTF8=1 python main.py")
