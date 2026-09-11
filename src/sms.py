"""The Twilio webhook, a thin shell over answer.answer().

Replies are asynchronous because the pipeline's median is 10.8s against Twilio's 15s
ceiling, which is also its maximum (PLAN §13).
"""
import base64
import hashlib
import hmac
import logging
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx
from flask import Flask, Response, request

from . import answer as answer_mod
from . import config, history, store

log = logging.getLogger("rag.sms")

# Sent when the pipeline itself fails. Deliberately distinct from answer.FALLBACK, which
# claims we looked and found nothing — a lie after a crash.
ERROR_REPLY = ("Sorry - something went wrong on our end and I couldn't look that up. "
               "Please try again in a minute, or call the studio.")

# Sent ONCE when a sender crosses the rate limit, then silence for the rest of the window.
# Sending it per over-limit message would hand an abuser a free outbound SMS per text,
# which is the cost this limit exists to avoid.
RATE_LIMIT_REPLY = ("You've hit the limit for now, so I've paused replies for a bit. "
                    "Please call the studio if it's urgent.")

# An empty TwiML document: HTTP 200, no synchronous reply. This is what makes the webhook
# return in milliseconds instead of waiting on the LLM.
EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


# ── SMS character encoding ───────────────────────────────────────────────────────────
# SMS bills per 160-char GSM-7 segment, but one character outside that alphabet forces the
# whole message into UCS-2 at 70 chars — turning a typical answer from 1 segment into 3.
GSM7 = set("@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡"
           "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà")
GSM7_EXTENDED = set("^{}\\[~]|€")  # legal, but each costs two septets

TRANSLITERATE = {"—": "-", "–": "-", "―": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
                 "…": "...", " ": " ", "•": "*", "→": "->", "×": "x"}


def to_gsm7(text):
    """Replace common non-GSM characters with GSM-safe equivalents. Anything unmapped is
    left alone, so the message falls back to UCS-2 rather than being mangled."""
    return "".join(TRANSLITERATE.get(c, c) for c in text)


def segments(text):
    """(encoding, segment count) under the SMS concatenation rules. Used for cost logging."""
    if all(c in GSM7 or c in GSM7_EXTENDED for c in text):
        septets = sum(2 if c in GSM7_EXTENDED else 1 for c in text)
        return "GSM-7", 1 if septets <= 160 else -(-septets // 153)
    return "UCS-2", 1 if len(text) <= 70 else -(-len(text) // 67)


# ── Request authentication ───────────────────────────────────────────────────────────
def expected_signature(auth_token, url, params):
    """Twilio's scheme: the exact URL Twilio called, then every POST field appended as
    name+value in case-sensitive sorted order, HMAC-SHA1 with the auth token, base64.

    Verified against Twilio's own published test vector, which tests/test_sms.py pins —
    that external vector is what makes it safe to implement this in stdlib rather than
    pulling in the twilio SDK for one function.
    """
    payload = url + "".join(k + params[k] for k in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def signature_ok(url, params, header):
    """Constant-time comparison, so the check cannot be probed byte by byte."""
    if not header:
        return False
    return hmac.compare_digest(
        expected_signature(config.require("TWILIO_AUTH_TOKEN"), url, params), header)


# ── Outbound ─────────────────────────────────────────────────────────────────────────
def send_sms(to, body):
    """Deliver one message through Twilio's REST API (the async half of the reply)."""
    account = config.require("TWILIO_ACCOUNT_SID")
    r = httpx.post(f"https://api.twilio.com/2010-04-01/Accounts/{account}/Messages.json",
                   auth=(account, config.require("TWILIO_AUTH_TOKEN")),
                   data={"To": to, "From": config.require("TWILIO_FROM_NUMBER"), "Body": body},
                   timeout=30)
    r.raise_for_status()
    return r.json().get("sid")


# ── The app ──────────────────────────────────────────────────────────────────────────
def create_app(index=None):
    """Build the Flask app. `index` is injectable so tests never touch Pinecone."""
    app = Flask(__name__)
    # Twilio's webhook payloads are a few KB. Anything larger is refused before the body is
    # read, rather than relying on Flask's default limit.
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    # Bounded pool: a burst of texts queues instead of spawning unlimited threads and
    # hammering OpenRouter into rate limits. Replies are delayed, never dropped.
    pool = ThreadPoolExecutor(max_workers=config.SMS_WORKERS, thread_name_prefix="sms")
    # Twilio can redeliver a webhook. Without this, one duplicate means a second LLM call
    # and a second text to the customer — paid for twice, and visibly wrong.
    seen, seen_lock = OrderedDict(), Lock()
    # Sliding-window rate limit, per sender. Checked BEFORE anything expensive happens,
    # so a rejected message costs nothing at all.
    recent, rate_lock = OrderedDict(), Lock()
    limit, window = config.SMS_RATE_LIMIT, config.SMS_RATE_WINDOW
    idx = index if index is not None else store.open_index(store.client())

    def already_handled(message_sid):
        if not message_sid:
            return False
        with seen_lock:
            if message_sid in seen:
                return True
            seen[message_sid] = None
            while len(seen) > 500:
                seen.popitem(last=False)
        return False

    def rate_state(sender):
        """'allow' | 'notify' (first rejection: send one notice) | 'drop' (stay quiet).

        Rejected messages deliberately do NOT count towards the window, so a sender who
        keeps texting is not punished with an ever-extending block — they simply get their
        next slot as the oldest timestamp ages out.
        """
        now = time.monotonic()
        with rate_lock:
            entry = recent.setdefault(sender, {"stamps": deque(), "notified": False})
            recent.move_to_end(sender)
            stamps = entry["stamps"]
            while stamps and now - stamps[0] > window:
                stamps.popleft()
            if len(stamps) < limit:
                stamps.append(now)
                entry["notified"] = False
                return "allow"
            if entry["notified"]:
                return "drop"
            entry["notified"] = True
            return "notify"

    def prune_senders():
        """Bound memory: a long-running process must not accumulate a row per number seen."""
        with rate_lock:
            while len(recent) > 2000:
                recent.popitem(last=False)

    def send_quietly(to, body):
        """Deliver a message we generated ourselves, logging rather than raising."""
        try:
            send_sms(to, body)
        except Exception:
            log.exception("could not deliver notice to %s", to)

    def answer_and_send(to, question, row_id):
        # Everything before this message, so the question is not shown as its own history.
        prior = history.recent(to, before_id=row_id) if config.USE_HISTORY else None
        try:
            result = answer_mod.answer(idx, question, history=prior)
            body = to_gsm7(result.text)
            encoding, n = segments(body)
            log.info("answered %s in %.1fs | %d chars, %s, %d segment(s) | gated=%s",
                     to, result.seconds, len(body), encoding, n, result.gated)
        except Exception:
            log.exception("pipeline failed for %s", to)
            body = to_gsm7(ERROR_REPLY)
        send_quietly(to, body)
        history.record(to, "out", body)

    @app.post("/sms")
    def inbound():
        # The URL must be byte-identical to what Twilio called or the signature will not
        # match — behind a tunnel or proxy, request.url is often http:// or an internal
        # host. Hence TWILIO_WEBHOOK_URL: paste the exact URL from the Twilio console.
        url = config.TWILIO_WEBHOOK_URL or request.url
        if not signature_ok(url, request.form.to_dict(), request.headers.get("X-Twilio-Signature")):
            # This endpoint is public and every accepted request spends money on an LLM
            # call and an outbound SMS. Rejecting unsigned requests is the only thing
            # standing between a leaked URL and someone else's bill.
            log.warning("rejected unsigned request from %s", request.remote_addr)
            return Response("signature check failed", status=403)

        question = (request.form.get("Body") or "").strip()
        sender = request.form.get("From", "")
        if question and sender and not already_handled(request.form.get("MessageSid", "")):
            # Stored before the rate-limit check, so the record is of every message the
            # customer actually sent, not only the ones we chose to answer.
            row_id = history.record(sender, "in", question, request.form.get("MessageSid"))
            # Dedup first: a webhook Twilio redelivered is not the sender's fault and must
            # not eat their quota.
            state = rate_state(sender)
            if state == "allow":
                pool.submit(answer_and_send, sender, question, row_id)
            else:
                log.warning("rate limited %s (%d/%.0fs) -> %s", sender, limit, window, state)
                if state == "notify":
                    notice = to_gsm7(RATE_LIMIT_REPLY)
                    pool.submit(send_quietly, sender, notice)
                    history.record(sender, "out", notice)
            prune_senders()
        # Always 200, always instantly. The real reply arrives over the REST API later.
        return Response(EMPTY_TWIML, mimetype="text/xml")

    @app.get("/health")
    def health():
        # Unauthenticated, so it says nothing about the index or model behind it.
        return {"ok": True}

    return app
