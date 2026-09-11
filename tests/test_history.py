"""Checks for src/history.py, offline against a throwaway database file.
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src import config, history

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)


tmp = tempfile.TemporaryDirectory()
config.HISTORY_DB = str(pathlib.Path(tmp.name) / "test.db")
ALICE, BOB = "+15550000001", "+15550000002"

# ── storing and reading back ─────────────────────────────────────────────────────────
first = history.record(ALICE, "in", "how much is a drop in", message_sid="SM1")
history.record(ALICE, "out", "A single drop-in class is $28.")
latest = history.record(ALICE, "in", "what about a 10 pack")

rows = history.recent(ALICE)
check([r["body"] for r in rows] == ["how much is a drop in",
                                    "A single drop-in class is $28.",
                                    "what about a 10 pack"],
      f"messages should come back oldest first, got {[r['body'] for r in rows]}")
check([r["direction"] for r in rows] == ["in", "out", "in"], "direction should be preserved")

# The message being answered must not appear inside its own history.
before = history.recent(ALICE, before_id=latest)
check([r["body"] for r in before] == ["how much is a drop in", "A single drop-in class is $28."],
      f"before_id should exclude the current message, got {[r['body'] for r in before]}")
check(history.recent(ALICE, before_id=first) == [], "nothing precedes the first message")

# ── bounds ───────────────────────────────────────────────────────────────────────────
check(len(history.recent(ALICE, limit=2)) == 2, "limit should cap how many come back")
check(history.recent(ALICE, limit=2)[-1]["body"] == "what about a 10 pack",
      "a capped read should keep the NEWEST messages, not the oldest")
# Messages older than the window drop out. Written with an explicit past timestamp rather
# than by sleeping, so the test stays fast.
with history.connect() as conn:
    conn.execute("INSERT INTO messages (phone, direction, body, created_at) VALUES (?,?,?,?)",
                 (ALICE, "in", "a question from last week", "2020-01-01T00:00:00+00:00"))
check(all("last week" not in r["body"] for r in history.recent(ALICE, hours=24)),
      "a message older than the window should not come back")
check(any("last week" in r["body"] for r in history.recent(ALICE, hours=24 * 365 * 50)),
      "the same message should come back with a wide enough window")

# ── customers are kept apart ─────────────────────────────────────────────────────────
history.record(BOB, "in", "do you have parking")
check(len(history.recent(BOB)) == 1, "one customer's history must not include another's")
check(len(history.recent(ALICE)) == 3, "Alice's history should be untouched by Bob")

# ── names ────────────────────────────────────────────────────────────────────────────
check(history.customer(ALICE)["name"] is None, "no name until the customer gives one")
history.record(ALICE, "in", "my name is Sara and i want to book")
check(history.customer(ALICE)["name"] == "Sara",
      f"an explicit introduction should be picked up, got {history.customer(ALICE)['name']!r}")

# A later guess must not overwrite a name we already hold.
history.record(ALICE, "in", "this is Dave asking for a friend")
check(history.customer(ALICE)["name"] == "Sara", "a stored name must not be overwritten by a guess")

# A correction by hand always wins.
history.set_name(ALICE, "Sarah")
check(history.customer(ALICE)["name"] == "Sarah", "set_name should overwrite")

# Ambiguous phrasing is skipped rather than guessed at.
for text in ("im new here", "i'm interested in barre", "this is ridiculous", "hey whats the price"):
    check(history.detect_name(text) is None, f"should not read a name out of {text!r}")
for text, want in (("my name is dev", "Dev"), ("this is maya", "Maya"), ("i'm kai", "Kai")):
    check(history.detect_name(text) == want, f"{text!r} should give {want}")

# ── the roster ───────────────────────────────────────────────────────────────────────
everyone = {c["phone"]: c for c in history.customers()}
check(set(everyone) == {ALICE, BOB}, f"both customers should be listed, got {set(everyone)}")
check(everyone[ALICE]["messages"] == 6, f"Alice's message count is wrong: {everyone[ALICE]['messages']}")
check(everyone[BOB]["name"] is None, "Bob never gave a name")

print("FAILURES:\n" + "\n".join("  ! " + f for f in fail) if fail else "history: ALL CHECKS PASS")
tmp.cleanup()
sys.exit(1 if fail else 0)
