"""Every tunable number and secret, loaded from .env.

Secrets go through require() at call time, so importing config never needs keys present.
"""
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
CONTENT_DIR = ROOT / "content"

# A six-line .env reader, chosen over adding the python-dotenv dependency.
# setdefault() means a real environment variable WINS over the file — that is what makes
# a one-off override like `TOP_K=8 python3 eval/run.py answers` work, which is exactly how
# k=5 and k=8 were compared before k=8 was locked in.
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        # Trailing comments are stripped only when preceded by whitespace: an API key can
        # legitimately contain '#', and silently truncating a key is a miserable bug.
        os.environ.setdefault(_k.strip(), re.sub(r"\s+#.*$", "", _v).strip())


def require(name):
    """Fetch a secret at call time, so importing config never needs keys present."""
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} is not set — add it to .env (see .env.example)")
    return value


def openrouter_key(override=None):
    """The OpenRouter key to bill: `override` (a web visitor's own) when given, else ours.
    Checked with `is None`, not truthiness, so an empty visitor key fails instead of quietly
    spending OPENROUTER_API_KEY."""
    return override if override is not None else require("OPENROUTER_API_KEY")


# ── Models ───────────────────────────────────────────────────────────────────────────
# All three run through OpenRouter, so one API key covers embeddings, generation and
# judging. EMBED_MODEL must be identical at ingest time and at query time, or the question
# vector and the stored vectors live in different spaces and cosine similarity is noise.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "openai/text-embedding-3-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", 1536))
GEN_MODEL = os.environ.get("GEN_MODEL", "google/gemini-3.8-flash")
# The judge MUST differ from the generator — a model grading its own output has a
# documented self-preference bias (PLAN §8).
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "z-ai/glm-5.3-flash")

# ── Pinecone ─────────────────────────────────────────────────────────────────────────
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "studio-kb")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "__default__")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")

# ── Retrieval ────────────────────────────────────────────────────────────────────────
# These defaults are the CALIBRATED values, not the original guesses, so a checkout with
# no .env still runs the configuration the eval measured. tests/test_config.py keeps them
# in step with .env.example.
TOP_K = int(os.environ.get("TOP_K", 8))              # chunks handed to the generator
CANDIDATES = int(os.environ.get("CANDIDATES", 20))   # depth of each arm BEFORE fusion
RRF_K = int(os.environ.get("RRF_K", 60))             # RRF constant: score = 1/(RRF_K + rank)
RRF_TEXT_WEIGHT = float(os.environ.get("RRF_TEXT_WEIGHT", 0.1))  # BM25 arm; dense is 1.0
RRF_TEXT_DEPTH = int(os.environ.get("RRF_TEXT_DEPTH", 5))        # drop the BM25 noise tail

# The abstention gate. Read against the RAW DENSE score, never the RRF score — see the
# long warning in fuse.py about why gating on a fused score is a no-op that looks like
# it works.
SCORE_FLOOR = float(os.environ.get("SCORE_FLOOR", 0.20))

# ── Answers ──────────────────────────────────────────────────────────────────────────
# 320 rather than a single 160-char SMS segment: multi-part answers (three pricing tiers,
# five Saturday classes) cannot fit in 160 without dropping facts, and a dropped fact just
# triggers a second inbound text (PLAN decision #11).
ANSWER_TARGET_CHARS = int(os.environ.get("ANSWER_TARGET_CHARS", 320))
ANSWER_MAX_CHARS = int(os.environ.get("ANSWER_MAX_CHARS", 480))

# ── Twilio SMS transport (src/sms.py) ────────────────────────────────────────────────
# TWILIO_WEBHOOK_URL must be the EXACT url set in the Twilio console, because signature
# validation hashes that string. Behind a tunnel, request.url is usually wrong.
TWILIO_WEBHOOK_URL = os.environ.get("TWILIO_WEBHOOK_URL", "")
SMS_WORKERS = int(os.environ.get("SMS_WORKERS", 4))

# Per-sender rate limit: the webhook is public and every accepted message costs an LLM
# call plus a 2-3 segment reply. The window is a float so tests can use a fraction of a second.
SMS_RATE_LIMIT = int(os.environ.get("SMS_RATE_LIMIT", 15))
SMS_RATE_WINDOW = float(os.environ.get("SMS_RATE_WINDOW", 3600))

# ── Conversation history (src/history.py) ────────────────────────────────────────────
# Every text in and out is stored in this SQLite file. USE_HISTORY controls only whether
# past messages are shown to the generator; storage happens either way, so turning it off
# does not lose data.
HISTORY_DB = os.environ.get("HISTORY_DB", str(ROOT / "history.db"))
USE_HISTORY = os.environ.get("USE_HISTORY", "true").lower() not in ("false", "0", "no")
HISTORY_MESSAGES = int(os.environ.get("HISTORY_MESSAGES", 6))   # how many past texts to show
HISTORY_HOURS = float(os.environ.get("HISTORY_HOURS", 24))      # and how far back to look

# Rewrites a follow-up into a standalone search query before retrieval, costing one extra
# LLM call per message that has history. On by measurement: retrieval 10/12 -> 12/12 (§14a).
USE_QUERY_REWRITE = os.environ.get("USE_QUERY_REWRITE", "true").lower() not in ("false", "0", "no")
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "z-ai/glm-5.3-flash")

# ── Web chat (src/ui.py) ─────────────────────────────────────────────────────────────
# Off by default, and it needs `pip install -r requirements-ui.txt`. When on, `rag serve`
# starts the web chat, where each visitor pays with their own OpenRouter key.
ENABLE_UI = os.environ.get("ENABLE_UI", "false").lower() in ("true", "1", "yes")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
