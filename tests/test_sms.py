"""Checks for src/sms.py, offline with the pipeline and sender stubbed.

Signature validation is pinned to Twilio's published test vector, not one this code
generated.
"""
import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src import config, history, sms

# Point conversation storage at a throwaway file BEFORE any app is built: these tests must
# never write to the real history.db.
_tmp = tempfile.TemporaryDirectory()
config.HISTORY_DB = str(pathlib.Path(_tmp.name) / "sms-test.db")

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)


# ── 1. signature: Twilio's published example ─────────────────────────────────────────
VECTOR_TOKEN = "12345"
VECTOR_URL = "https://example.com/myapp.php?foo=1&bar=2"
VECTOR_PARAMS = {"CallSid": "CA1234567890ABCDE", "Caller": "+14158675310",
                 "Digits": "1234", "From": "+14158675310", "To": "+18005551212"}
VECTOR_SIGNATURE = "L/OH5YylLD5NRKLltdqwSvS0BnU="

check(sms.expected_signature(VECTOR_TOKEN, VECTOR_URL, VECTOR_PARAMS) == VECTOR_SIGNATURE,
      "signature does not match Twilio's published test vector")

# Param order must not matter (Twilio sorts), but content must.
shuffled = dict(reversed(list(VECTOR_PARAMS.items())))
check(sms.expected_signature(VECTOR_TOKEN, VECTOR_URL, shuffled) == VECTOR_SIGNATURE,
      "signature should be independent of dict ordering")
tampered = dict(VECTOR_PARAMS, Digits="9999")
check(sms.expected_signature(VECTOR_TOKEN, VECTOR_URL, tampered) != VECTOR_SIGNATURE,
      "a tampered parameter must change the signature")

# ── 2. GSM-7 transliteration and segment counting ────────────────────────────────────
from src.answer import FALLBACK

check(sms.segments(FALLBACK)[0] == "UCS-2", "the raw fallback should be UCS-2 (it has an em dash)")
check(sms.segments(sms.to_gsm7(FALLBACK)) == ("GSM-7", 1),
      f"transliterated fallback should be 1 GSM-7 segment, got {sms.segments(sms.to_gsm7(FALLBACK))}")
check(sms.to_gsm7("it’s 9–5 … ok") == "it's 9-5 ... ok",
      f"transliteration wrong: {sms.to_gsm7('it’s 9–5 … ok')!r}")
# Anything with no safe equivalent is left alone rather than mangled.
check(sms.to_gsm7("café 🎉") == "café 🎉", "unmapped characters must pass through untouched")
check(sms.segments("a" * 160) == ("GSM-7", 1), "160 GSM chars is one segment")
check(sms.segments("a" * 161) == ("GSM-7", 2), "161 GSM chars splits into two")
# Extended-table characters cost two septets, so half as many fit: 80 euro signs are
# exactly 160 septets (still one segment), 81 tips into two.
check(sms.segments("€" * 80) == ("GSM-7", 1), "80 euro signs is exactly 160 septets: one segment")
check(sms.segments("€" * 81)[1] == 2, "81 euro signs exceeds 160 septets: two segments")

# ── 3. the webhook, with the pipeline and the sender stubbed ─────────────────────────
os.environ["TWILIO_AUTH_TOKEN"] = VECTOR_TOKEN
os.environ["TWILIO_ACCOUNT_SID"] = "ACtest"
os.environ["TWILIO_FROM_NUMBER"] = "+15550000000"
WEBHOOK_URL = "https://example.com/sms"
config.TWILIO_WEBHOOK_URL = WEBHOOK_URL

asked, sent, seen_history = [], [], []


class FakeAnswer:
    text = "A single drop-in class is $28 — bookable without a membership."
    seconds, gated = 1.0, False


sms.answer_mod.answer = lambda idx, question, history=None: (asked.append(question),
                                                             seen_history.append(history),
                                                             FakeAnswer())[2]
sms.send_sms = lambda to, body: sent.append((to, body))

app = sms.create_app(index="fake-index")
client = app.test_client()


def post(params, signature=None, url=WEBHOOK_URL):
    sig = signature if signature is not None else sms.expected_signature(VECTOR_TOKEN, url, params)
    return client.post("/sms", data=params, headers={"X-Twilio-Signature": sig})


def wait_for(predicate, timeout=2.0):
    """Poll for work the pool does on another thread, instead of guessing at a sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    return predicate()


INBOUND = {"Body": "how much is a drop in", "From": "+15551234567", "MessageSid": "SM1"}

check(post(INBOUND, signature="wrong").status_code == 403, "an invalid signature must be rejected")
check(post(INBOUND, signature="").status_code == 403, "a missing signature must be rejected")
check(not asked, "no request should have reached the pipeline yet")

r = post(INBOUND)
check(r.status_code == 200, f"a signed request should be accepted, got {r.status_code}")
check(b"<Response></Response>" in r.data, "should acknowledge with empty TwiML, not a TwiML reply")

# The answer is produced on a worker thread, so wait for it rather than assuming.
wait_for(lambda: bool(sent))

check(asked == ["how much is a drop in"], f"pipeline should have been called once, got {asked}")
check(len(sent) == 1 and sent[0][0] == "+15551234567", f"one reply to the sender, got {sent}")
check("—" not in sent[0][1] and "-" in sent[0][1], "outbound text should be transliterated to GSM-7")

# the transport must store both sides of the exchange
check(wait_for(lambda: len(history.recent("+15551234567", limit=10)) == 2),
      "the exchange should reach the database")
stored = history.recent("+15551234567", limit=10)
check([m["direction"] for m in stored] == ["in", "out"],
      f"the webhook should store the question and the reply, got {[m['direction'] for m in stored]}")
check(stored[0]["body"] == "how much is a drop in", "the inbound text should be stored verbatim")
check(seen_history == [[]], f"the first message has no prior history, got {seen_history}")

# a redelivered webhook must not answer twice
post(INBOUND)
time.sleep(0.2)
check(len(asked) == 1, f"duplicate MessageSid should be ignored, pipeline calls: {len(asked)}")
check(len(sent) == 1, f"duplicate MessageSid should not send a second text, sent: {len(sent)}")

# an empty body is acknowledged but not answered
post({"Body": "   ", "From": "+15551234567", "MessageSid": "SM2"})
time.sleep(0.1)
check(len(asked) == 1, "an empty message body should not reach the pipeline")

# ── 4. per-sender rate limiting ──────────────────────────────────────────────────────
# A fresh app with a tiny limit and a 1-second window, so the sliding window is exercised
# for real rather than by mocking the clock. Keep the assertions fast: this test's own
# elapsed time has to stay well inside the window it configures.
config.SMS_RATE_LIMIT = 3
config.SMS_RATE_WINDOW = 1.0
rl_app = sms.create_app(index="fake-index")
rl_client = rl_app.test_client()
asked.clear()
sent.clear()


def rl_post(body, sender, sid):
    params = {"Body": body, "From": sender, "MessageSid": sid}
    sig = sms.expected_signature(VECTOR_TOKEN, WEBHOOK_URL, params)
    return rl_client.post("/sms", data=params, headers={"X-Twilio-Signature": sig})


SPAMMER = "+15559999999"
for i in range(3):
    check(rl_post(f"question {i}", SPAMMER, f"RL{i}").status_code == 200,
          "requests under the limit should be accepted")
check(wait_for(lambda: len(asked) == 3), f"3 messages under the limit should all be answered, got {len(asked)}")

# The 4th crosses the limit: no pipeline call, exactly one notice.
rl_post("question 4", SPAMMER, "RL4")
check(wait_for(lambda: len(sent) == 4), f"expected 3 answers + 1 notice, got {len(sent)}")
check(len(asked) == 3, f"an over-limit message must not reach the pipeline, got {len(asked)}")
check("paused replies" in sent[-1][1], f"the 4th reply should be the rate-limit notice: {sent[-1][1]!r}")

# Further messages in the same window are silent — the notice is NOT resent, or an abuser
# would earn a free outbound SMS per text.
for i in range(5, 9):
    rl_post(f"question {i}", SPAMMER, f"RL{i}")
time.sleep(0.2)
check(len(asked) == 3, f"still no pipeline calls while rate limited, got {len(asked)}")
check(len(sent) == 4, f"the notice must be sent only once per window, got {len(sent)} sends")

# A different sender is unaffected.
rl_post("innocent question", "+15558888888", "RLX")
check(wait_for(lambda: len(asked) == 4), f"a different sender should not be blocked, got {len(asked)}")

# Once the window rolls off, the original sender is allowed again.
time.sleep(config.SMS_RATE_WINDOW + 0.2)
rl_post("later question", SPAMMER, "RLY")
check(wait_for(lambda: len(asked) == 5), f"sender should be allowed again after the window, got {len(asked)}")

# a second question from the same number should carry the first exchange as history
asked.clear()
sent.clear()
seen_history.clear()
post({"Body": "what about the 10 pack", "From": "+15551234567", "MessageSid": "SM3"})
check(wait_for(lambda: len(sent) == 1), "the follow-up should be answered")
check(len(seen_history) == 1 and len(seen_history[0]) == 2,
      f"the follow-up should see the 2 earlier messages, got {seen_history}")
check(seen_history[0][0]["body"] == "how much is a drop in",
      "history passed to the pipeline should start with the earlier question")
check(all("what about the 10 pack" != m["body"] for m in seen_history[0]),
      "a question must not appear inside its own history")

# ── 5. hardening: nothing an outsider can reach should leak or execute ──────────────
# Flask's debugger can run arbitrary code, and Flask obeys FLASK_DEBUG from the
# environment unless run() is told otherwise. Capture what `rag serve` passes to run().
import argparse
import flask
from src import cli

captured = {}
real_run, real_open, real_client = flask.Flask.run, sms.store.open_index, sms.store.client
flask.Flask.run = lambda self, **kw: captured.update(kw)
sms.store.open_index = lambda pc: "fake-index"
sms.store.client = lambda: None
os.environ["FLASK_DEBUG"] = "1"
config.ENABLE_UI = False  # the web chat has its own checks in test_ui.py
try:
    cli.cmd_serve(argparse.Namespace(host="127.0.0.1", port=5099, ui_port=7999))
finally:
    flask.Flask.run, sms.store.open_index, sms.store.client = real_run, real_open, real_client
    del os.environ["FLASK_DEBUG"]
check(captured.get("debug") is False,
      f"rag serve must force debug off even with FLASK_DEBUG=1, passed debug={captured.get('debug')!r}")
check(captured.get("load_dotenv") is False, "rag serve must not let Flask read .env itself")
check(captured.get("host") == "127.0.0.1", "rag serve should bind to this machine only by default")

hard = sms.create_app(index="fake-index").test_client()
check(hard.get("/health").get_json() == {"ok": True},
      f"/health is unauthenticated and must reveal nothing, got {hard.get('/health').get_json()}")
big = hard.post("/sms", data=b"Body=" + b"A" * (100 * 1024),
                content_type="application/x-www-form-urlencoded",
                headers={"X-Twilio-Signature": "forged"})
check(big.status_code == 413, f"an oversized unsigned request should be refused, got {big.status_code}")

_tmp.cleanup()
print("FAILURES:\n" + "\n".join("  ! " + f for f in fail) if fail else "sms transport: ALL CHECKS PASS")
sys.exit(1 if fail else 0)
