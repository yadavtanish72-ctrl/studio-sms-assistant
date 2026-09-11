"""Checks for src/ui.py, offline: OpenRouter and Pinecone are faked, and the one real server
listens on this machine only.

The property that matters most: a visitor's message is paid for by the visitor's key on
every call, and never by ours.
"""
import argparse
import os
import pathlib
import socket
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src import answer, config, store

try:
    import gradio as gr
except ModuleNotFoundError:
    print("ui: SKIPPED (Gradio is optional: .venv/bin/pip install -r requirements-ui.txt)")
    sys.exit(0)

import httpx

from src import ui

# Set after config has read .env, so this fake replaces the real key for the whole run.
OWNER, VISITOR = "sk-or-OWNER", "sk-or-VISITOR"
os.environ["OPENROUTER_API_KEY"] = OWNER
_tmp = tempfile.TemporaryDirectory()
config.HISTORY_DB = str(pathlib.Path(_tmp.name) / "ui-test.db")
config.USE_HISTORY = config.USE_QUERY_REWRITE = True  # so a follow-up exercises every call

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)


# ── fakes: every OpenRouter request is recorded with the key it carried ─────────────
calls, models, status = [], [], {"code": 200, "finish": "stop"}
REWRITTEN = "how long does the 10-class pack last"


def reply(url, payload):
    return httpx.Response(status["code"], json=payload, request=httpx.Request("POST", url))


def fake_post(url, headers=None, json=None, timeout=None):
    """answer.chat() posts with httpx.post: the rewrite and the answer both come here."""
    kind = "rewrite" if json["messages"][0]["content"] == answer.REWRITE_SYSTEM else "answer"
    calls.append((kind, headers["Authorization"]))
    models.append((kind, json["model"]))
    text = REWRITTEN if kind == "rewrite" else "A drop-in class is $28.\nSOURCES: 1"
    return reply(url, {"choices": [{"message": {"content": text}, "finish_reason": status["finish"]}]})


class FakeClient:
    """embed.embed() posts through an httpx.Client."""
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        calls.append(("embed", headers["Authorization"]))
        return reply(url, {"data": [{"index": i, "embedding": [0.1] * config.EMBED_DIM}
                                    for i in range(len(json["input"]))]})


real_post, real_client = httpx.post, httpx.Client
httpx.post, httpx.Client = fake_post, FakeClient
HIT = {"id": "pricing.md#drop-in-and-class-packs", "content": "Pricing > Drop-in: $28.", "score": 0.55}
store.dense_search = lambda idx, qv: [HIT]
store.text_search = lambda idx, q: [HIT]

bot = gr.Chatbot()


def as_browser_sends(chat):
    """The conversation as it comes back from the page, after Gradio's own round trip."""
    return bot.preprocess(bot.postprocess(chat))


def error_of(question, chat, key, model=config.GEN_MODEL):
    """The gr.Error message a turn raises, or None if it went through."""
    try:
        ui.respond("fake-index", question, chat, key, model)
    except gr.Error as e:
        return e.message
    return None


# ── 1. the visitor's key pays for every call; ours never does ────────────────────────
check(config.openrouter_key("") == "", "an empty override must not fall back to OPENROUTER_API_KEY")
check(config.openrouter_key(None) == OWNER, "no override means the .env key (the CLI and SMS paths)")

# A typed model is tidied up; an empty box means the default.
box, chat, panel = ui.respond("fake-index", "how much is a drop in class", [], VISITOR,
                              "  Anthropic/Claude-Haiku-4.5 ")
check("`anthropic/claude-haiku-4.5`" in panel, "the panel should name the model that answered")
box, chat, panel = ui.respond("fake-index", "what about the 10 pack", as_browser_sends(chat), VISITOR, "")
check(models == [("answer", "anthropic/claude-haiku-4.5"), ("rewrite", config.REWRITE_MODEL),
                 ("answer", config.GEN_MODEL)],
      f"the chosen model should write the answer and only the answer, got {models}")
check(f"`{config.GEN_MODEL}` (the one the evaluation measured)" in panel,
      "an empty model box should fall back to the measured default, and say so")
kinds = [kind for kind, _ in calls]
check(kinds == ["embed", "answer", "rewrite", "embed", "answer"],
      f"a first message and a follow-up should make exactly these calls, got {kinds}")
check(all(auth == f"Bearer {VISITOR}" for _, auth in calls),
      f"every call must carry the visitor's key, got {sorted({a for _, a in calls})}")
check(box == "", "the message box should be cleared after a turn")
check([m["role"] for m in chat] == ["user", "assistant"] * 2, "both turns should be in the conversation")
check(f'"{REWRITTEN}"' in panel, "the panel should show the rewritten search for a follow-up")
check("`pricing.md#drop-in-and-class-packs`" in panel, "the panel should list the source used")

# ── 2. no key, no call of any kind ───────────────────────────────────────────────────
calls.clear()
for blank in ("", "   ", None):
    msg = error_of("how much is a drop in class", [], blank)
    check(msg is not None and "key" in msg.lower(), f"a missing key must stop the turn, key={blank!r}")
check(error_of("x" * (ui.MAX_QUESTION + 1), [], VISITOR) is not None, "an over-long message must be refused")
for bad in ("gpt", "not a model", "a/b/c", "google/" + "x" * 80, "google/gemini;rm -rf"):
    msg = error_of("how much is a drop in class", [], VISITOR, bad)
    check(msg is not None and "model ID" in msg, f"{bad!r} is not a model ID and must be refused, got {msg!r}")
check(not calls, f"refused turns must make no request at all, made {calls}")

# ── 3. a refused key is explained, never echoed back ─────────────────────────────────
for code, words in ((400, "rejected that model"), (401, "didn't accept"), (402, "out of credit"),
                     (500, "went wrong")):
    status["code"] = code
    msg = error_of("how much is a drop in class", [], VISITOR)
    check(msg is not None and words in msg, f"HTTP {code} should say '{words}', got {msg!r}")
    check(msg is None or VISITOR not in msg, "an error message must never contain the key")
status["code"] = 200
# A model that runs out of tokens mid-answer gets told to try another, not to "try again".
status["finish"] = "length"
msg = error_of("how much is a drop in class", [], VISITOR)
check(msg is not None and "didn't finish" in msg, f"a truncated answer should suggest another model, got {msg!r}")
status["finish"] = "stop"

# ── 4. nothing from the web chat is stored ───────────────────────────────────────────
check(not pathlib.Path(config.HISTORY_DB).exists(), "the web chat must not write to the history database")

# ── 5. launch settings win over hostile environment variables ────────────────────────
# Each of these would weaken the page if Gradio were left to read it. Captured, not run, so a
# regression here can never actually open a public tunnel.
HOSTILE = {"GRADIO_SHARE": "True", "GRADIO_MCP_SERVER": "True", "GRADIO_SSR_MODE": "True",
           "GRADIO_VIBE_MODE": "1", "GRADIO_ALLOWED_PATHS": str(config.ROOT)}
os.environ.update(HOSTILE)
captured = {}
real_launch = gr.Blocks.launch
gr.Blocks.launch = lambda self, **kw: captured.update(kw)
try:
    demo = ui.launch("fake-index", "127.0.0.1", 7999)
finally:
    gr.Blocks.launch = real_launch
for name, want in (("share", False), ("run_history", False), ("enable_monitoring", False),
                   ("mcp_server", False), ("ssr_mode", False), ("server_name", "127.0.0.1")):
    check(captured.get(name) == want, f"launch({name}=...) should be {want!r}, got {captured.get(name)!r}")
check(str(config.ROOT) in captured.get("blocked_paths", []), "the project folder must be blocked")
check(demo.vibe_mode is False, "GRADIO_VIBE_MODE must not switch on the in-browser code editor")

# Then for real, still with the hostile variables that are safe to run: a file inside the project
# must not be downloadable, and the code editor must refuse.
httpx.post, httpx.Client = real_post, real_client
os.environ.pop("GRADIO_SHARE"), os.environ.pop("GRADIO_SSR_MODE"), os.environ.pop("GRADIO_MCP_SERVER")
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
demo = ui.launch("fake-index", "127.0.0.1", port)
try:
    base = f"http://127.0.0.1:{port}"
    check(httpx.get(base).status_code == 200, "the page itself should load")
    canary = config.ROOT / "README.md"  # always present; stands in for .env, which may not be
    r = httpx.get(f"{base}/gradio_api/file={canary}")
    check(r.status_code != 200 and "Studio SMS Assistant" not in r.text,
          f"a project file was downloadable through Gradio (HTTP {r.status_code})")
    r = httpx.post(f"{base}/gradio_api/upload", files={"files": ("x.txt", b"hello")})
    check(r.status_code == 413, f"uploads would let anyone fill this server's disk, got HTTP {r.status_code}")
    # Valid bodies, so a refusal comes from the vibe_mode check and not from validation.
    for route, body in (("vibe-edit", {"prompt": "x"}), ("vibe-code", {"code": "x"})):
        r = httpx.post(f"{base}/gradio_api/{route}", json=body)
        check(r.status_code == 403, f"/{route} rewrites the app's code and must be refused, got HTTP {r.status_code}")
finally:
    demo.close()
    for name in HOSTILE:
        os.environ.pop(name, None)

# ── 6. `rag serve` runs whatever is configured, and nothing when nothing is ─────────
import flask

from src import cli


class DemoStub:
    blocked = False

    def block_thread(self):
        self.blocked = True


def fake_launch(idx, host, port):
    started["ui"] = (idx, host, port)
    return stub


TWILIO = {"TWILIO_ACCOUNT_SID": "ACtest", "TWILIO_AUTH_TOKEN": "t", "TWILIO_FROM_NUMBER": "+15550000000"}
real_twilio = {name: os.environ.pop(name, None) for name in TWILIO}
started = {}
saved = flask.Flask.run, ui.launch, store.open_index, store.client
flask.Flask.run = lambda self, **kw: started.setdefault("sms", kw)
ui.launch = fake_launch
store.client = lambda: None
store.open_index = lambda pc: (started.setdefault("pinecone", True), "fake-index")[1]
try:
    for sms_on, ui_on in ((True, False), (True, True), (False, True)):
        started.clear()
        stub = DemoStub()
        config.ENABLE_UI = ui_on
        for name, value in TWILIO.items():
            if sms_on:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)
        cli.cmd_serve(argparse.Namespace(host="127.0.0.1", port=5099, ui_port=7999))
        label = f"Twilio set={sms_on}, ENABLE_UI={ui_on}"
        check(("sms" in started) == sms_on, f"{label}: SMS webhook started={'sms' in started}")
        check(("ui" in started) == ui_on, f"{label}: web chat started={'ui' in started}")
        check(stub.blocked == (ui_on and not sms_on),
              f"{label}: a web-chat-only server must keep running in the foreground")
    check(started.get("ui") == ("fake-index", "127.0.0.1", 7999),
          f"the web chat should get the index and --host/--ui-port, got {started.get('ui')}")

    started.clear()
    config.ENABLE_UI = False
    for name in TWILIO:
        os.environ.pop(name, None)
    try:
        cli.cmd_serve(argparse.Namespace(host="127.0.0.1", port=5099, ui_port=7999))
        check(False, "with nothing configured, rag serve should refuse to start")
    except SystemExit as e:
        check("Nothing to serve" in str(e), f"unexpected message: {e}")
    check("pinecone" not in started, "with nothing to serve, Pinecone should not be opened")
finally:
    flask.Flask.run, ui.launch, store.open_index, store.client = saved
    for name, value in real_twilio.items():
        if value is not None:
            os.environ[name] = value

_tmp.cleanup()
print("FAILURES:\n" + "\n".join("  ! " + f for f in fail) if fail else "ui: ALL CHECKS PASS")
sys.exit(1 if fail else 0)
