"""Validate eval/golden.yaml and eval/multiturn.yaml against content/.

Also prints question/evidence vocabulary overlap, the guard against an eval that
flatters itself by reusing the source documents' words.
"""
import yaml, re, pathlib, collections

norm = lambda s: " ".join(str(s).split())
C = pathlib.Path(__file__).parent.parent / "content"
FULL = {f.name: norm(f.read_text()) for f in C.glob("*.md")}
ALL = " ".join(FULL.values())

slugify = lambda h: re.sub(r"-+", "-", re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-"))
slugs = collections.defaultdict(set)
for f in C.glob("*.md"):
    txt = f.read_text()
    # sections are ## headings; a file with none (the atomic schedule) is cited by its H1
    hs = re.findall(r"^## (.+)$", txt, re.M) or re.findall(r"^# (.+)$", txt, re.M)
    slugs[f.name].update(slugify(h) for h in hs)

D = yaml.safe_load(open(pathlib.Path(__file__).parent / "golden.yaml"))
err = []
REQ = {"id","question","category","expect","sources","reference_answer","must_include","evidence"}

for e in D:
    i = e.get("id", "?")
    if set(e) != REQ: err.append(f"{i}: field mismatch {set(e) ^ REQ}")
    for s in e["sources"]:
        fn, _, sl = s.partition("#")
        if fn not in FULL: err.append(f"{i}: no such file '{fn}'")
        elif sl not in slugs[fn]: err.append(f"{i}: '{fn}' has no heading '{sl}'")
    # evidence: list of verbatim quotes, one per source
    ev = e["evidence"] if isinstance(e["evidence"], list) else [e["evidence"]]
    if e["expect"] == "answerable":
        if not any(norm(x) for x in ev): err.append(f"{i}: answerable but no evidence")
        for q in ev:
            if norm(q) and norm(q) not in ALL:
                err.append(f"{i}: evidence not verbatim -> {norm(q)[:60]!r}")
    else:
        if any(norm(x) for x in ev) or e["sources"] or e["must_include"]:
            err.append(f"{i}: abstain entries must have empty sources/must_include/evidence")
    for m in e["must_include"]:
        if norm(m) not in ALL: err.append(f"{i}: must_include not a literal corpus string: {m!r}")
    if e["category"] == "multi_hop" and len({s.partition('#')[0] for s in e["sources"]}) < 2:
        err.append(f"{i}: multi_hop but sources span <2 files")
    if e["category"] == "out_of_scope" and e["expect"] != "abstain":
        err.append(f"{i}: out_of_scope must expect abstain")

dupes = [q for q, n in collections.Counter(norm(e["question"].lower()) for e in D).items() if n > 1]
err += [f"duplicate question: {q!r}" for q in dupes]

# question/evidence vocabulary overlap — high means the question echoes the source
STOP = set("a an the is are do does did you your i my me we our it its of to in on for at and or if can "
           "could would will what when where how much many there any u ur im whats dont cant get have need".split())
ov = []
for e in D:
    if e["expect"] != "answerable": continue
    ev = " ".join(e["evidence"]) if isinstance(e["evidence"], list) else e["evidence"]
    q = {w for w in re.findall(r"[a-z0-9$]+", e["question"].lower()) if w not in STOP}
    v = {w for w in re.findall(r"[a-z0-9$]+", ev.lower()) if w not in STOP}
    if q: ov.append((len(q & v) / len(q), e["id"]))

untested = sorted({f"{fn}#{s}" for fn in slugs for s in slugs[fn]} - {s for e in D for s in e["sources"]})

print(f"entries: {len(D)}  {dict(collections.Counter(e['category'] for e in D))}")
print(f"mean question/evidence vocab overlap: {sum(o for o,_ in ov)/len(ov):.2f}  (lower is better)")
print(f"untested sections: {untested or 'none'}")
print("\nERRORS:\n" + ("\n".join("  ! " + x for x in err) if err else "  none"))


# ─────────────────────────────────────────────────────────────────────────────────────
# eval/multiturn.yaml — same rules as above, plus a scripted prior conversation. Kept in
# its own file so the 55 single-turn questions stay a stable ruler.
M = yaml.safe_load(open(pathlib.Path(__file__).parent / "multiturn.yaml"))
merr = []
MREQ = REQ | {"history"}

for e in M:
    i = e.get("id", "?")
    if set(e) != MREQ:
        merr.append(f"{i}: field mismatch {set(e) ^ MREQ}")
        continue
    for s in e["sources"]:
        fn, _, sl = s.partition("#")
        if fn not in FULL:
            merr.append(f"{i}: no such file '{fn}'")
        elif sl not in slugs[fn]:
            merr.append(f"{i}: '{fn}' has no heading '{sl}'")
    ev = e["evidence"] if isinstance(e["evidence"], list) else [e["evidence"]]
    if e["expect"] == "answerable":
        if not any(norm(x) for x in ev):
            merr.append(f"{i}: answerable but no evidence")
        for q in ev:
            if norm(q) and norm(q) not in ALL:
                merr.append(f"{i}: evidence not verbatim -> {norm(q)[:60]!r}")
    elif any(norm(x) for x in ev) or e["sources"] or e["must_include"]:
        merr.append(f"{i}: abstain entries must have empty sources/must_include/evidence")
    for m in e["must_include"]:
        if norm(m) not in ALL:
            merr.append(f"{i}: must_include not a literal corpus string: {m!r}")

    # the scripted conversation itself
    if not e["history"]:
        merr.append(f"{i}: multi-turn entry with no history")
    for turn in e["history"]:
        if set(turn) != {"direction", "body"}:
            merr.append(f"{i}: history turn must have exactly direction+body, got {set(turn)}")
        elif turn["direction"] not in ("in", "out") or not str(turn["body"]).strip():
            merr.append(f"{i}: bad history turn {turn}")
    if e["history"] and e["history"][-1]["direction"] != "out":
        merr.append(f"{i}: history should end with the bot's reply, then the new question")
    # A false_history entry is pointless unless the history really does contradict the
    # corpus, so require that its final bot turn is NOT something the corpus supports.
    if e["category"] == "out_of_scope" and e["expect"] != "abstain":
        merr.append(f"{i}: out_of_scope must expect abstain")
    if e["category"] == "false_history":
        last_bot = norm(e["history"][-1]["body"])
        if last_bot in ALL:
            merr.append(f"{i}: false_history but the prior bot turn is corpus-accurate")

mdupes = [q for q, n in collections.Counter(norm(e["question"].lower()) for e in M).items() if n > 1]
merr += [f"duplicate multi-turn question: {q!r}" for q in mdupes]
overlap = {e["id"] for e in M} & {e["id"] for e in D}
merr += [f"id collides with golden.yaml: {i}" for i in sorted(overlap)]

print(f"\nmulti-turn entries: {len(M)}  {dict(collections.Counter(e['category'] for e in M))}")
print("MULTI-TURN ERRORS:\n" + ("\n".join("  ! " + x for x in merr) if merr else "  none"))
