"""The measurement harness: retrieval, sweep, answers, control, multiturn.

Tier 1 makes no LLM calls and is where tuning happens; the rest cost money per run.
"""
import argparse
import concurrent.futures as futures
import hashlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import yaml

from src import answer as answer_mod
from src import chunk, config, embed, fuse, store

EVAL_DIR = pathlib.Path(__file__).parent
CACHE = EVAL_DIR / ".cache"
# 55 questions: exact-fact lookups, paraphrases sharing no vocabulary with the source,
# enumerations, multi-hop, 8 out-of-scope, plus injection and personal-data adversarials.
# Each entry carries `sources` (gold chunk ids), `must_include` (literal facts) and
# `expect` (answerable | abstain). eval/audit_golden.py validates it against content/.
GOLDEN = yaml.safe_load(open(EVAL_DIR / "golden.yaml"))
# The multi-turn set lives in its own file so the 55 single-turn questions stay a stable
# ruler — §12's numbers are only comparable over time if that set does not move.
MULTITURN = yaml.safe_load(open(EVAL_DIR / "multiturn.yaml"))


def question_vectors(questions):
    """Embed questions, caching per (model, question) so Tier-1 reruns are free."""
    CACHE.mkdir(exist_ok=True)
    paths, missing = {}, []
    for q in questions:
        key = hashlib.sha1(f"{config.EMBED_MODEL}|{q}".encode()).hexdigest()
        paths[q] = CACHE / f"{key}.json"
        if not paths[q].exists():
            missing.append(q)
    if missing:
        for q, v in zip(missing, embed.embed(missing)):
            paths[q].write_text(json.dumps(v))
        print(f"embedded {len(missing)} new questions ({len(questions) - len(missing)} cached)")
    return {q: json.loads(p.read_text()) for q, p in paths.items()}


def retrieve_all():
    """Run every golden question through retrieval once, keeping the full 20-deep list so a
    miss can be reported as a rank rather than just "not found"."""
    idx = store.open_index(store.client())
    vectors = question_vectors([e["question"] for e in GOLDEN])
    out = []
    for e in GOLDEN:
        r = fuse.retrieve(idx, e["question"], query_vector=vectors[e["question"]],
                          top_k=config.CANDIDATES)
        out.append((e, r))
    return out


def _rank(hits, wanted):
    """1-based rank of the best-placed wanted id, or None if none of them appear."""
    ids = [h["id"] for h in hits]
    return min((ids.index(w) + 1 for w in wanted if w in ids), default=None)


def _recall_at_k(results, k, arm):
    """Fraction of answerable questions whose gold sources are all in the arm's top k."""
    hit = full = 0
    for e, r in results:
        if e["expect"] != "answerable":
            continue
        full += 1
        ids = {h["id"] for h in getattr(r, arm)[:k]}
        hit += set(e["sources"]).issubset(ids)
    return hit / full, full


def cmd_retrieval(args):
    """Tier 1: recall@k, hit@k and MRR per arm, plus the SCORE_FLOOR calibration table.
    Makes no generation calls."""
    results = retrieve_all()
    k = args.k
    answerable = [(e, r) for e, r in results if e["expect"] == "answerable"]

    print(f"\n=== Tier 1: retrieval (k={k}, candidates={config.CANDIDATES}) ===")
    print(f"{len(answerable)} answerable / {len(results) - len(answerable)} abstain questions\n")

    print(f"{'arm':10} {'recall@k (all sources)':>24} {'hit@k (any source)':>20} {'MRR':>8}")
    for arm in ("chunks", "dense", "text"):
        recall, n = _recall_at_k(results, k, arm)
        ranks = [_rank(getattr(r, arm), e["sources"]) for e, r in answerable]
        hit = sum(1 for x in ranks if x and x <= k) / n
        mrr = sum(1 / x for x in ranks if x) / n
        label = {"chunks": "fused", "dense": "dense", "text": "bm25"}[arm]
        print(f"{label:10} {recall:>23.1%} {hit:>19.1%} {mrr:>8.3f}")

    print("\n--- retrieval misses (gold not fully in fused top-k) ---")
    misses = 0
    for e, r in answerable:
        if set(e["sources"]).issubset({h["id"] for h in r.chunks[:k]}):
            continue
        misses += 1
        print(f"  {e['id']} [{e['category']}] {e['question'][:58]!r}")
        for src in e["sources"]:
            fr, dr, tr = (_rank(r.chunks, [src]), _rank(r.dense, [src]), _rank(r.text, [src]))
            print(f"      {src:45} fused={fr} dense={dr} bm25={tr}")
    if not misses:
        print("  none")

    print(f"\n--- abstention gate (SCORE_FLOOR={config.SCORE_FLOOR}) ---")
    gated = [(e, r) for e, r in results if r.dense_top1 < config.SCORE_FLOOR]
    should = [(e, r) for e, r in results if e["expect"] == "abstain"]
    tp = [e for e, _ in gated if e["expect"] == "abstain"]
    precision = len(tp) / len(gated) if gated else 1.0
    recall = len(tp) / len(should) if should else 1.0
    print(f"  precision {precision:.2f}  recall {recall:.2f}  "
          f"({len(gated)} gated, {len(should)} should abstain)")
    for e, r in gated:
        if e["expect"] != "abstain":
            print(f"  ! FALSE ABSTENTION {e['id']} score={r.dense_top1:.3f} {e['question'][:50]!r}")

    # SCORE_FLOOR calibration: how many answerable questions a floor would wrongly refuse,
    # against how many out-of-scope ones it catches. Precision 1.0 first, then best recall.
    print("\n--- threshold sweep (raw dense top-1) ---")
    ans_scores = sorted(r.dense_top1 for e, r in results if e["expect"] == "answerable")
    oos_scores = sorted(r.dense_top1 for e, r in results if e["expect"] == "abstain")
    print(f"  answerable   min={ans_scores[0]:.3f}  p10={ans_scores[len(ans_scores)//10]:.3f}")
    print(f"  out-of-scope max={oos_scores[-1]:.3f}  p90={oos_scores[int(len(oos_scores)*.9)]:.3f}")
    print(f"  {'floor':>7} {'false abstentions':>18} {'correct abstentions':>21}")
    for t in [x / 100 for x in range(15, 51, 2)]:
        fa = sum(1 for s in ans_scores if s < t)
        ca = sum(1 for s in oos_scores if s < t)
        flag = "  <- precision 1.0" if fa == 0 else ""
        print(f"  {t:>7.2f} {fa:>18} {ca:>16}/{len(oos_scores)}{flag}")


def cmd_sweep(args):
    """Score RRF weight and BM25 depth against the golden set. Re-fuses already-retrieved
    lists, so 28 configurations cost one pass over Pinecone."""
    results = retrieve_all()
    answerable = [(e, r) for e, r in results if e["expect"] == "answerable"]
    k = args.k
    print(f"\n=== fusion sweep (recall@{k} = all gold sources in top {k}) ===")
    print(f"{'text_weight':>12} {'bm25_depth':>11} {'recall@k':>10} {'hit@k':>8} {'MRR':>8}")
    for weight in (0.0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0):
        for depth in (3, 5, 10, 20):
            recall = hit = mrr = 0
            for e, r in answerable:
                fused = fuse.rrf(r.dense, r.text, top_k=config.CANDIDATES,
                                 text_weight=weight, text_depth=depth)
                ids = {h["id"] for h in fused[:k]}
                recall += set(e["sources"]).issubset(ids)
                rank = _rank(fused, e["sources"])
                hit += bool(rank and rank <= k)
                mrr += 1 / rank if rank else 0
            n = len(answerable)
            print(f"{weight:>12} {depth:>11} {recall/n:>9.1%} {hit/n:>7.1%} {mrr/n:>8.3f}")


# LLM-as-judge, two verdicts because they fail independently: an answer can be grounded and
# still miss the question, or state every fact while bolting on an invented one.
JUDGE = """You grade one answer from a fitness studio's SMS assistant. Judge only what is \
written; do not use outside knowledge.

faithful: true only if EVERY factual claim in the ANSWER is supported by CONTEXT. An answer \
that says it does not have the information is faithful.
correct: true only if the ANSWER conveys the required facts. Grade on facts present, not \
wording — a terser answer that states the facts is correct, and different phrasing of the \
same fact (e.g. "Mon" vs "Monday") is correct. If REQUIRED FACTS is empty, correct means the \
answer declines to answer rather than inventing something.

Reply with JSON only: {"faithful": true|false, "correct": true|false, "why": "<12 words>"}"""


def judge(entry, context, text):
    prompt = (f"CONTEXT:\n{context}\n\nQUESTION:\n{entry['question']}\n\n"
              f"REFERENCE ANSWER:\n{entry['reference_answer'] or '(should decline)'}\n\n"
              f"REQUIRED FACTS:\n{entry['must_include'] or '(none — must decline)'}\n\n"
              f"ANSWER:\n{text}")
    try:
        raw = answer_mod.chat([{"role": "system", "content": JUDGE},
                               {"role": "user", "content": prompt}],
                              model=config.JUDGE_MODEL, max_tokens=6000).text
        body = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        return json.loads(body)
    except Exception as exc:
        # A judge that fails must not silently count as a failed answer.
        return {"faithful": None, "correct": None, "why": f"JUDGE ERROR: {exc}"}


def _report(rows, label, cost, seconds):
    """Print one Tier-2 scorecard. `must_include` is a literal substring check and is the
    strictest number here; judge correctness is the metric PLAN §8 specified."""
    ans = [r for r in rows if r[0]["expect"] == "answerable"]
    absts = [r for r in rows if r[0]["expect"] == "abstain"]
    covered = [all(m.lower() in r[1].lower() for m in r[0]["must_include"]) for r in ans]
    judged = [r for r in ans if r[3]["correct"] is not None]
    print(f"\n=== Tier 2: {label} ===")
    print(f"answerable ({len(ans)})")
    print(f"  must_include coverage  {sum(covered)/len(ans):>6.1%}   (deterministic, literal facts)")
    print(f"  judge correctness      {sum(bool(r[3]['correct']) for r in judged)/len(judged):>6.1%}"
          f"   ({len(judged)} judged, {len(ans) - len(judged)} judge errors)")
    print(f"  judge faithfulness     {sum(bool(r[3]['faithful']) for r in judged)/len(judged):>6.1%}")
    print(f"  wrongly abstained      {sum(r[2] for r in ans):>6}")
    print(f"  over {config.ANSWER_MAX_CHARS}-char cap        {sum(r[5] for r in ans):>6}"
          f"   (mean {sum(r[4] for r in ans)/len(ans):.0f} chars)")
    print(f"abstain ({len(absts)})")
    if absts:
        print(f"  abstained correctly    {sum(r[2] for r in absts)/len(absts):>6.1%}")
    print(f"cost  ${cost:.4f} total, ${cost/len(rows):.4f} per question")
    print(f"latency  {seconds/len(rows):.1f}s mean per question")

    print("\n--- failures ---")
    bad = 0
    for (e, text, abstained, j, chars, over), cov in zip(ans, covered):
        if cov and j.get("correct") and j.get("faithful") and not abstained:
            continue
        bad += 1
        flags = ("ABSTAINED " if abstained else "") + ("" if cov else "MISSING_FACTS ") + \
                ("" if j.get("correct") else "INCORRECT ") + ("" if j.get("faithful") else "UNFAITHFUL ")
        print(f"  {e['id']} [{e['category']}] {flags}| {e['question'][:48]!r}")
        print(f"      {text[:150]}")
        print(f"      judge: {j.get('why', '')}")
    for e, text, abstained, j, chars, over in absts:
        if abstained:
            continue
        bad += 1
        print(f"  {e['id']} [{e['category']}] ANSWERED_INSTEAD_OF_ABSTAINING | {e['question'][:48]!r}")
        print(f"      {text[:150]}")
    if not bad:
        print("  none")


def cmd_answers(args):
    """Tier 2: the full pipeline per question, then judged. Abstentions skip the judge."""
    idx = store.open_index(store.client())
    entries = GOLDEN[:args.limit] if args.limit else GOLDEN
    vectors = question_vectors([e["question"] for e in entries])

    def one(e):
        try:
            a = answer_mod.answer(idx, e["question"], query_vector=vectors[e["question"]])
        except Exception as exc:  # one bad question must not abort a 55-question run
            return (e, f"ERROR: {exc}", False, {"faithful": False, "correct": False,
                                                "why": "pipeline error"}, 0, False), 0, 0.0
        abstained = a.gated or a.abstained
        j = ({"faithful": True, "correct": not e["must_include"], "why": "gated/abstained"}
             if abstained else judge(e, answer_mod.build_context(a.chunks), a.text))
        return (e, a.text, abstained, j, len(a.text), a.over_cap), a.usage.get("cost", 0), a.seconds

    # 3 workers, not more: OpenRouter starts returning 429s above that, and chat() already
    # spends retries on backoff. 55 questions take about 3 minutes.
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        out = list(pool.map(one, entries))
    _report([r for r, _, _ in out], f"retrieval k={config.TOP_K}, {config.GEN_MODEL}",
            sum(c for _, c, _ in out), sum(s for _, _, s in out))


def cmd_control(args):
    """The whole KB in the prompt, no retrieval. Same prompt, judge and questions, so the
    only difference is the context."""
    entries = GOLDEN[:args.limit] if args.limit else GOLDEN
    kb = answer_mod.build_context([{"content": c.content} for c in chunk.all_chunks()])
    system = answer_mod.SYSTEM.format(fallback=answer_mod.FALLBACK,
                                      target=config.ANSWER_TARGET_CHARS,
                                      max=config.ANSWER_MAX_CHARS)

    def one(e):
        try:
            reply = answer_mod.chat([{"role": "system", "content": system},
                                     {"role": "user", "content":
                                      f"CONTEXT:\n{kb}\n\nCUSTOMER MESSAGE:\n{e['question']}"}])
        except Exception as exc:
            return (e, f"ERROR: {exc}", False, {"faithful": False, "correct": False,
                                                "why": "pipeline error"}, 0, False), 0, 0.0
        text = answer_mod._split_sources(reply.text, [])[0]
        abstained = answer_mod.is_abstention(text)
        j = ({"faithful": True, "correct": not e["must_include"], "why": "abstained"}
             if abstained else judge(e, kb, text))
        return (e, text, abstained, j, len(text), len(text) > config.ANSWER_MAX_CHARS), \
               reply.usage.get("cost", 0), reply.seconds

    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        out = list(pool.map(one, entries))
    _report([r for r, _, _ in out], "CONTROL — full KB in prompt, no retrieval",
            sum(c for _, c, _ in out), sum(s for _, _, s in out))
    print(f"prompt carries {len(kb)} chars of context per question")


def cmd_multiturn(args):
    """Answer each question with a scripted conversation behind it. Retrieval and answering
    are reported separately because they fail for different reasons."""
    idx = store.open_index(store.client())
    entries = MULTITURN[:args.limit] if args.limit else MULTITURN

    def one(e):
        try:
            a = answer_mod.answer(idx, e["question"], history=e["history"])
        except Exception as exc:
            return e, None, {"faithful": False, "correct": False, "why": f"pipeline error: {exc}"}, 0
        abstained = a.gated or a.abstained
        j = ({"faithful": True, "correct": not e["must_include"], "why": "gated/abstained"}
             if abstained else judge(e, answer_mod.build_context(a.chunks), a.text))
        return e, a, j, a.usage.get("cost", 0)

    # Serial on purpose. Each question is two LLM calls, and concurrent runs were losing
    # roughly half the judge calls to 429s — which shows up as unreliable numbers, the one
    # thing an eval cannot afford. 14 questions is small enough to just wait.
    rows = [one(e) for e in entries]

    label = "ON" if config.USE_QUERY_REWRITE else "OFF"
    print(f"\n=== multi-turn ({len(entries)} questions, query rewrite {label}) ===\n")
    print(f"{'category':16} {'n':>3} {'retrieved':>10} {'correct':>9} {'faithful':>9} {'abstained':>10}")

    by_category, failures, judge_errors = {}, [], 0
    for e, a, j, _ in rows:
        cat = by_category.setdefault(e["category"], {"n": 0, "ret": 0, "cor": 0, "fai": 0,
                                                     "abs": 0, "needs_ret": 0, "judged": 0})
        cat["n"] += 1
        abstained = a is not None and (a.gated or a.abstained)
        cat["abs"] += abstained
        # A judge that errored says nothing about the answer, so it is counted apart
        # rather than silently scored as a failure.
        graded = j.get("correct") is not None
        judge_errors += not graded
        cat["judged"] += graded
        cat["cor"] += bool(j.get("correct"))
        cat["fai"] += bool(j.get("faithful"))
        if e["expect"] == "answerable":
            cat["needs_ret"] += 1
            found = a is not None and set(e["sources"]).issubset({c["id"] for c in a.chunks})
            cat["ret"] += found
            if not found or (graded and not (j.get("correct") and j.get("faithful"))):
                failures.append((e, a, j, found))
        elif not abstained:
            failures.append((e, a, j, None))

    for name, c in by_category.items():
        retrieved = f"{c['ret']}/{c['needs_ret']}" if c["needs_ret"] else "-"
        print(f"{name:16} {c['n']:>3} {retrieved:>10} {c['cor']:>4}/{c['judged']} "
              f"{c['fai']:>4}/{c['judged']} {c['abs']:>5}/{c['n']}")

    totals = {k: sum(c[k] for c in by_category.values())
              for k in ("n", "ret", "cor", "fai", "needs_ret", "judged")}
    retrieved_total = f"{totals['ret']}/{totals['needs_ret']}"
    print(f"\n{'TOTAL':16} {totals['n']:>3} {retrieved_total:>10} "
          f"{totals['cor']:>4}/{totals['judged']} {totals['fai']:>4}/{totals['judged']}")
    if judge_errors:
        print(f"  ({judge_errors} judge call(s) failed and are excluded from correct/faithful)")
    print(f"cost ${sum(c for _, _, _, c in rows):.4f}")

    print("\n--- failures ---")
    for e, a, j, found in failures:
        flags = []
        if found is False:
            flags.append("NOT_RETRIEVED")
        if found is None:
            flags.append("ANSWERED_INSTEAD_OF_ABSTAINING")
        if not j.get("correct"):
            flags.append("INCORRECT")
        if not j.get("faithful"):
            flags.append("UNFAITHFUL")
        print(f"  {e['id']} [{e['category']}] {' '.join(flags)} | {e['question'][:52]!r}")
        if a is not None:
            print(f"      answer: {a.text[:130]}")
            print(f"      judge:  {j.get('why', '')}")
    if not failures:
        print("  none")


def main():
    p = argparse.ArgumentParser(prog="eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--k", type=int, default=config.TOP_K)
    r.set_defaults(func=cmd_retrieval)
    s = sub.add_parser("sweep")
    s.add_argument("--k", type=int, default=config.TOP_K)
    s.set_defaults(func=cmd_sweep)
    a = sub.add_parser("answers")
    a.add_argument("--limit", type=int)
    a.set_defaults(func=cmd_answers)
    c = sub.add_parser("control")
    c.add_argument("--limit", type=int)
    c.set_defaults(func=cmd_control)
    mt = sub.add_parser("multiturn")
    mt.add_argument("--limit", type=int)
    mt.set_defaults(func=cmd_multiturn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
