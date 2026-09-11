"""The web chat: an optional Gradio page over answer.answer(), started by `rag serve` when
ENABLE_UI is on.

Each visitor's own OpenRouter key pays for their messages and is never stored. Search runs on
the Pinecone index of whoever runs `rag serve`.
"""
import logging
import re

import gradio as gr
import httpx

from . import answer as answer_mod
from . import config, sms

log = logging.getLogger("rag.ui")

MAX_QUESTION = 1600  # the longest message Twilio accepts, so the SMS path's own cap
# The shape of every OpenRouter model ID, e.g. google/gemini-3.8-flash or google/gemma-4-31b-it:free.
MODEL_ID = re.compile(r"[a-z0-9.:~-]{1,60}/[a-z0-9.:~-]{1,60}")

INTRO = """## Northline Studio · text assistant

Text a question the way you'd text a gym. It answers from the studio's own documents, or \
says it doesn't know rather than guessing. Northline Studio is fictional: its prices, \
classes and trainers were invented for this demo.

**This page runs on your OpenRouter key, not the site owner's.** With the default model a \
message costs about $0.004 on your account. Your key is sent to this server to make your requests and is never \
saved, and the conversation is kept only in this browser tab. Only paste a key into a site \
you trust; safest is a new key with a $1 limit, from \
[openrouter.ai/keys](https://openrouter.ai/keys)."""

HINT = ("Send a message and this panel shows how the answer was found: what it searched for, "
        "how close the best match was, and which documents it used.")

EXAMPLES = ["how much is a drop in class", "do you have a sauna", "whats on saturday",
            "its 24 hrs notice to cancel right"]

# What a visitor sees when OpenRouter refuses a request. None of these echo the key back.
REFUSED = {400: "OpenRouter rejected that model. Check the ID at openrouter.ai/models, or try another.",
           401: "OpenRouter didn't accept that key. Check you copied all of it.",
           402: "That key is out of credit. Add some on openrouter.ai, or raise the key's limit."}
UNFINISHED = "That model didn't finish a usable answer. Try another one, or the default."
FAILED = "Something went wrong looking that up. Please try again in a minute."


def _text(message):
    """Gradio hands chat messages back with their content split into parts."""
    return "".join(part.get("text", "") for part in message["content"])


def describe(question, a, model):
    """The 'what happened' panel: the same facts `rag ask --debug` prints."""
    body = sms.to_gsm7(a.text)
    _, parts = sms.segments(body)
    if a.gated:
        decision = "refused before asking the AI, because nothing was close enough"
    elif a.abstained:
        decision = "refused, because the AI found no answer in the pieces it was given"
    else:
        decision = "answered"
    searched = "your message, as typed" if a.search_query == question else f'"{a.search_query}"'
    return "\n".join([
        "### What happened",
        f"- **Model:** `{model}`" + (" (the one the evaluation measured)" if model == config.GEN_MODEL else ""),
        f"- **Searched for:** {searched}",
        f"- **Best match:** {a.dense_top1:.2f} (it refuses below {config.SCORE_FLOOR:.2f})",
        f"- **Decision:** {decision}",
        "- **Sources:** " + (", ".join(f"`{c}`" for c in a.citations) or "none"),
        f"- **As a text:** {len(body)} characters, {parts} SMS segment{'' if parts == 1 else 's'}",
        "",
        f"**All {len(a.chunks)} pieces it considered, best first:**",
        *(f"{n}. `{c['id']}`" for n, c in enumerate(a.chunks, 1)),
    ])


def respond(idx, question, chat, key, model):
    """One turn, returning (message box, conversation, panel). Problems raise gr.Error, which
    Gradio shows as a popup and which leaves the typed message where it was."""
    question, key = (question or "").strip(), (key or "").strip()
    model = (model or "").strip().lower() or config.GEN_MODEL
    # Checked before anything else: with no key of theirs there is nothing to bill but ours.
    if not key:
        raise gr.Error("Paste your OpenRouter key first. This page only runs on your own key.")
    if not question:
        return "", chat, gr.skip()
    if len(question) > MAX_QUESTION:
        raise gr.Error(f"That's {len(question)} characters. The limit is {MAX_QUESTION}, "
                       "the most one text message can hold.")
    if not MODEL_ID.fullmatch(model):
        raise gr.Error(f"That doesn't look like an OpenRouter model ID. They're written like "
                       f"{config.GEN_MODEL}; see openrouter.ai/models.")
    # The conversation comes from this browser tab and nothing goes into history.db, which
    # has no way to delete a stranger's messages.
    prior = ([{"direction": "in" if m["role"] == "user" else "out", "body": _text(m)}
              for m in chat][-config.HISTORY_MESSAGES:] if config.USE_HISTORY else None)
    try:
        a = answer_mod.answer(idx, question, history=prior, api_key=key, model=model)
    except httpx.HTTPStatusError as e:
        raise gr.Error(REFUSED.get(e.response.status_code, FAILED)) from None
    except RuntimeError:
        # answer.chat() raises this when a model hits its token cap or returns nothing.
        raise gr.Error(UNFINISHED) from None
    except Exception:
        log.exception("web chat turn failed")
        raise gr.Error(FAILED) from None
    chat = chat + [{"role": "user", "content": question}, {"role": "assistant", "content": a.text}]
    return "", chat, describe(question, a, model)


def build(idx):
    """The page. `idx` is the Pinecone index every visitor's search runs against."""
    with gr.Blocks(title="Northline Studio text assistant", analytics_enabled=False) as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            key = gr.Textbox(label="Your OpenRouter API key", type="password",
                             placeholder="sk-or-v1-...", scale=3)
            model = gr.Textbox(label="Model", value=config.GEN_MODEL, scale=2,
                               info="Any model ID from openrouter.ai/models. The default is the "
                                    "one the evaluation measured.")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversation", height=420)
                msg = gr.Textbox(label="Your text", placeholder="how much is a drop in class")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    new = gr.Button("New conversation")
                gr.Examples(EXAMPLES, inputs=msg)
            with gr.Column(scale=2):
                panel = gr.Markdown(HINT)
        gr.on([msg.submit, send.click], lambda q, c, k, m: respond(idx, q, c, k, m),
              inputs=[msg, chat, key, model], outputs=[msg, chat, panel], api_name="ask")
        new.click(lambda: ("", [], HINT), outputs=[msg, chat, panel], api_visibility="private")
    # GRADIO_VIBE_MODE would give every visitor an AI editor for this app's code, which is
    # remote code execution, and launch() has no option to turn it off.
    demo.vibe_mode = False
    return demo


def launch(idx, host, port):
    """Start the web chat on a background thread and return it, so `rag serve` can run the
    SMS webhook in the foreground or wait on the page itself."""
    demo = build(idx)
    # Four answers in flight at once and twenty waiting; past that, visitors are told it's busy.
    demo.queue(default_concurrency_limit=4, max_size=20)
    demo.launch(
        server_name=host, server_port=port, prevent_thread_lock=True,
        # Each of these is explicit because Gradio would otherwise take it from an environment
        # variable, or from a default meant for notebooks.
        share=False,               # GRADIO_SHARE would open a public tunnel to this machine
        run_history=False,         # saves every input, the visitor's key included, in their browser
        enable_monitoring=False,   # a usage dashboard, on by default
        mcp_server=False,          # GRADIO_MCP_SERVER would expose the chat as an AI tool
        ssr_mode=False,            # GRADIO_SSR_MODE would start a second, Node.js server
        max_file_size=0,           # Gradio's upload route is open even with no upload button
        # .env and history.db live under ROOT. A blocked path wins over GRADIO_ALLOWED_PATHS.
        blocked_paths=[str(config.ROOT)],
    )
    return demo
