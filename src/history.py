"""Every text in and out, stored in a SQLite file, plus a row per phone number.

Each call opens its own connection, which keeps it safe across the webhook's threads.
"""
import datetime
import pathlib
import re
import sqlite3

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    phone      TEXT PRIMARY KEY,
    name       TEXT,
    first_seen TEXT NOT NULL,
    last_seen  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    phone       TEXT NOT NULL,
    direction   TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    message_sid TEXT
);
CREATE INDEX IF NOT EXISTS messages_by_phone ON messages (phone, created_at);
"""


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def connect(path=None):
    """Open the database, creating it and its tables on first use."""
    db = pathlib.Path(path or config.HISTORY_DB)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL lets a read happen while another thread is writing, which is the normal case
    # when two texts arrive close together.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


# ── Names ────────────────────────────────────────────────────────────────────────────
# Only explicit self-introductions. Deliberately narrow: calling a customer by the wrong
# name is worse than not using one, so anything uncertain is skipped.
NAME_PATTERNS = [r"\bmy name(?:'s| is)\s+([a-z][a-z'\-]{1,19})\b",
                 r"\bthis is\s+([a-z][a-z'\-]{1,19})\b",
                 r"\bi'?m\s+([a-z][a-z'\-]{1,19})\b"]

# Words that follow "i'm" or "this is" far more often than a name does.
NOT_NAMES = {"a", "an", "the", "new", "just", "not", "in", "on", "at", "so", "very",
             "really", "sorry", "good", "ok", "okay", "fine", "free", "here", "back",
             "still", "about", "only", "also", "already", "almost", "from", "with",
             "for", "interested", "looking", "trying", "wondering", "coming", "going",
             "gonna", "thinking", "asking", "after", "before", "unsure", "confused",
             "ridiculous", "annoyed", "sure", "keen", "happy", "hoping", "booked"}


def detect_name(text):
    """Return a name if the message clearly introduces one, else None."""
    for pattern in NAME_PATTERNS:
        m = re.search(pattern, text.lower())
        if m and m.group(1) not in NOT_NAMES:
            return m.group(1).capitalize()
    return None


# ── Writing ──────────────────────────────────────────────────────────────────────────
def record(phone, direction, body, message_sid=None, conn=None):
    """Store one message and keep the customer row current. direction is 'in' or 'out'.

    Returns the new row id, so the caller can ask recent() for everything BEFORE it — the
    question being answered right now should not appear in its own history.
    """
    own = conn is None
    conn = conn or connect()
    try:
        now = _now()
        with conn:
            cur = conn.execute(
                "INSERT INTO messages (phone, direction, body, created_at, message_sid) "
                "VALUES (?, ?, ?, ?, ?)", (phone, direction, body, now, message_sid))
            row_id = cur.lastrowid
            conn.execute(
                "INSERT INTO customers (phone, name, first_seen, last_seen) "
                "VALUES (?, NULL, ?, ?) "
                "ON CONFLICT(phone) DO UPDATE SET last_seen = excluded.last_seen",
                (phone, now, now))
            # Never overwrite a name we already have — a manually corrected name must not
            # be clobbered by a later guess.
            if direction == "in":
                name = detect_name(body)
                if name:
                    conn.execute(
                        "UPDATE customers SET name = ? WHERE phone = ? AND name IS NULL",
                        (name, phone))
        return row_id
    finally:
        if own:
            conn.close()


def set_name(phone, name, conn=None):
    """Set or correct a customer's name by hand."""
    own = conn is None
    conn = conn or connect()
    try:
        now = _now()
        with conn:
            conn.execute(
                "INSERT INTO customers (phone, name, first_seen, last_seen) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(phone) DO UPDATE SET name = excluded.name", (phone, name, now, now))
    finally:
        if own:
            conn.close()


# ── Reading ──────────────────────────────────────────────────────────────────────────
def recent(phone, limit=None, hours=None, before_id=None, conn=None):
    """The last few messages with this number, oldest first.

    Bounded on both axes on purpose. `limit` keeps the prompt small; `hours` keeps a
    conversation from three months ago out of today's answer, where it would be noise.
    """
    own = conn is None
    conn = conn or connect()
    try:
        limit = config.HISTORY_MESSAGES if limit is None else limit
        hours = config.HISTORY_HOURS if hours is None else hours
        cutoff = (datetime.datetime.now(datetime.timezone.utc)
                  - datetime.timedelta(hours=hours)).isoformat(timespec="seconds")
        rows = conn.execute(
            "SELECT direction, body, created_at FROM messages "
            "WHERE phone = ? AND created_at >= ? AND id < ? ORDER BY id DESC LIMIT ?",
            (phone, cutoff, before_id if before_id is not None else 2 ** 62, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        if own:
            conn.close()


def customer(phone, conn=None):
    own = conn is None
    conn = conn or connect()
    try:
        row = conn.execute("SELECT * FROM customers WHERE phone = ?", (phone,)).fetchone()
        return dict(row) if row else None
    finally:
        if own:
            conn.close()


def customers(conn=None):
    """Everyone we have heard from, busiest conversation last."""
    own = conn is None
    conn = conn or connect()
    try:
        rows = conn.execute(
            "SELECT c.phone, c.name, c.first_seen, c.last_seen, "
            "       COUNT(m.id) AS messages "
            "FROM customers c LEFT JOIN messages m ON m.phone = c.phone "
            "GROUP BY c.phone ORDER BY c.last_seen").fetchall()
        return [dict(r) for r in rows]
    finally:
        if own:
            conn.close()
