"""The command line: ingest, ask, eval, serve, and the history commands.

`./rag` is a wrapper that runs this module with the project's own Python.
"""
import argparse
import subprocess
import sys

from . import answer as answer_mod
from . import chunk, config, embed, history, store


def cmd_ingest(args):
    """Rebuild the index from content/, idempotently. Embedding runs before the wipe, so a
    failure there leaves the existing index intact and serving."""
    pc = store.client()
    if args.init:
        print(f"index '{config.PINECONE_INDEX}': "
              + ("created" if store.create_index(pc) else "already exists"))
    idx = store.open_index(pc)
    chunks = chunk.all_chunks()
    print(f"chunked {len(chunks)} sections from {len({c.source_file for c in chunks})} files")
    vectors = embed.embed([c.content for c in chunks])
    store.wipe(idx)
    store.upsert(idx, chunks, vectors)
    # Reading the count back is the verification step: it proves the write landed, and
    # proves re-ingesting did not duplicate anything.
    print(f"upserted {len(chunks)} documents; index now holds {store.count(idx)}")


def cmd_ask(args):
    """Answer one question. With --as, read and write that number's history like a texter."""
    """One question, end to end. The first line is what a customer would receive; the
    bracketed line and below are diagnostics that would never go out over SMS."""
    idx = store.open_index(store.client())
    prior = None
    if args.as_number:
        # Recorded before the answer so a sequence of asks builds a real conversation, and
        # before_id keeps this question out of its own history.
        row_id = history.record(args.as_number, "in", args.question)
        prior = history.recent(args.as_number, before_id=row_id) if config.USE_HISTORY else None
    a = answer_mod.answer(idx, args.question, history=prior)
    if args.as_number:
        history.record(args.as_number, "out", a.text)
    print(a.text)
    # dense_top1 is the gate's input — printing it is how you tell a refusal caused by the
    # score floor (GATED) from one the model itself chose.
    print(f"\n[{len(a.text)} chars | dense_top1 {a.dense_top1:.3f}"
          f"{' | GATED' if a.gated else ''}{' | OVER CAP' if a.over_cap else ''}]")
    if prior:
        print(f"history: {len(prior)} earlier message(s) from {args.as_number}")
    if a.search_query and a.search_query != args.question:
        print(f"searched for: {a.search_query!r}")
    print("citations: " + (", ".join(a.citations) or "none"))
    if args.debug:
        print("retrieved:")
        for n, c in enumerate(a.chunks, 1):
            print(f"  [{n}] {c['id']:45} rrf={c['rrf_score']:.5f}")


def cmd_eval(args):
    """Delegate to eval/run.py: rag eval retrieval | answers | control | sweep"""
    # A subprocess rather than an import: eval/ is a directory of scripts, not a package,
    # and this keeps the eval runnable directly as `python3 eval/run.py ...` too.
    raise SystemExit(subprocess.call(
        [sys.executable, str(config.ROOT / "eval" / "run.py"), *args.rest]))


def cmd_serve(args):
    """Run the Twilio webhook. Flask's dev server is fine for a studio's traffic and for
    testing through a tunnel; put gunicorn in front of it if this ever matters."""
    from . import sms  # imported lazily so `rag ask` does not require flask installed
    print(f"POST /sms  ->  http://{args.host}:{args.port}/sms")
    if not config.TWILIO_WEBHOOK_URL:
        print("WARNING: TWILIO_WEBHOOK_URL is unset — signature validation will hash the "
              "reconstructed request URL, which is usually wrong behind a tunnel.")
    # debug=False is passed explicitly because Flask's run() otherwise obeys FLASK_DEBUG
    # from the environment, and its debugger exposed through a tunnel can run code on this
    # machine. load_dotenv=False stops Flask reading .env on its own.
    sms.create_app().run(host=args.host, port=args.port, threaded=True,
                         debug=False, load_dotenv=False)


def cmd_customers(args):
    from . import history
    rows = history.customers()
    if not rows:
        print("nobody has texted yet")
        return
    print(f"{'phone':18} {'name':14} {'msgs':>5}  last heard from")
    for r in rows:
        print(f"{r['phone']:18} {(r['name'] or '-'):14} {r['messages']:>5}  {r['last_seen']}")


def cmd_history(args):
    from . import history
    who = history.customer(args.phone)
    if not who:
        print(f"no record of {args.phone}")
        return
    print(f"{args.phone}  {who['name'] or '(name unknown)'}  since {who['first_seen']}\n")
    # limit=0/hours=0 would hide everything, so ask for the whole conversation explicitly.
    for m in history.recent(args.phone, limit=args.limit, hours=args.hours):
        speaker = "customer" if m["direction"] == "in" else "bot"
        print(f"[{m['created_at']}] {speaker:8} {m['body']}")


def cmd_name(args):
    from . import history
    history.set_name(args.phone, args.name)
    print(f"{args.phone} is now {args.name}")


def main():
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="rebuild the index from content/")
    ing.add_argument("--init", action="store_true", help="create the index first if missing")
    ing.set_defaults(func=cmd_ingest)
    ask = sub.add_parser("ask", help="answer one question end to end")
    ask.add_argument("question")
    ask.add_argument("--debug", action="store_true", help="show the fused chunk list")
    ask.add_argument("--as", dest="as_number", metavar="PHONE",
                     help="answer as if texted from this number: uses and stores its history")
    ask.set_defaults(func=cmd_ask)
    ev = sub.add_parser("eval", help="retrieval | answers | control | sweep")
    # REMAINDER passes everything after `eval` through to eval/run.py untouched, so
    # `rag eval retrieval --k 3` works without redeclaring the eval's own flags here.
    ev.add_argument("rest", nargs=argparse.REMAINDER)
    ev.set_defaults(func=cmd_eval)
    srv = sub.add_parser("serve", help="run the Twilio SMS webhook")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=5000)
    srv.set_defaults(func=cmd_serve)
    cus = sub.add_parser("customers", help="who has texted, and how much")
    cus.set_defaults(func=cmd_customers)
    his = sub.add_parser("history", help="one customer's conversation")
    his.add_argument("phone")
    his.add_argument("--limit", type=int, default=200)
    his.add_argument("--hours", type=float, default=24 * 365)
    his.set_defaults(func=cmd_history)
    nm = sub.add_parser("name", help="set or correct a customer's name")
    nm.add_argument("phone")
    nm.add_argument("name")
    nm.set_defaults(func=cmd_name)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
