"""Cross-file consistency guard for content/, to run after any edit to the corpus.

The same facts repeat across 8 files, and a contradiction there is a wrong answer that no
retrieval metric would ever flag.
"""
import re, pathlib, itertools
C = pathlib.Path("content")
sched = (C/"schedule.md").read_text()
diff  = (C/"difficulty.md").read_text()
trn   = (C/"trainers.md").read_text()
fail = []

# --- parse schedule table ---
rows = []
for ln in sched.splitlines():
    m = re.match(r"\|\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*\|", ln)
    if m:
        c = [x.strip() for x in ln.strip("|").split("|")]
        rows.append(dict(day=c[0], time=c[1], cls=c[2], who=c[3], lvl=c[4], mins=c[5]))
print(f"schedule rows parsed: {len(rows)}")

# --- 1. levels in schedule vs difficulty.md ---
lvl_map = {}
for lv in ("1","2","3"):
    sec = " ".join(re.search(rf"## Level {lv}.*?(?=\n## |\Z)", diff, re.S).group(0).split())
    for cls in re.findall(r"(Sunrise Flow|Vinyasa Flow|Power Yoga|Express HIIT|Strength Circuit|Reformer Pilates|Barre Basics|Restore & Stretch)", sec):
        lvl_map.setdefault(cls, lv)
for r in rows:
    if lvl_map.get(r["cls"]) != r["lvl"]:
        fail.append(f"LEVEL MISMATCH {r['cls']}: schedule={r['lvl']} difficulty={lvl_map.get(r['cls'])}")

# --- 2. durations consistent per class ---
for cls, g in itertools.groupby(sorted(rows, key=lambda r: r["cls"]), lambda r: r["cls"]):
    d = {r["mins"] for r in g}
    if len(d) > 1: fail.append(f"DURATION VARIES {cls}: {d}")

# --- 3. trainer names + which classes they teach ---
bios = {}
for m in re.finditer(r"## ([A-Z][a-z]+ [A-Z][a-z]+)\n(.*?)(?=\n## |\Z)", trn, re.S):
    bios[m.group(1)] = " ".join(m.group(2).split())
for r in rows:
    if r["who"] not in bios:
        fail.append(f"TRAINER NOT IN BIOS: {r['who']}")
    elif r["cls"] not in bios[r["who"]]:
        fail.append(f"BIO MISSING CLASS: {r['who']} teaches {r['cls']} in schedule, not in bio")

# --- 4. bio day lists match schedule ---
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
ABBR = dict(zip(DAYS, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]))
for who, bio in bios.items():
    claimed = {ABBR[d] for d in DAYS if d in bio}
    actual  = {r["day"] for r in rows if r["who"] == who}
    if claimed != actual:
        fail.append(f"DAYS MISMATCH {who}: bio={sorted(claimed)} schedule={sorted(actual)}")

# --- 5. dangling cross-references (broken by ##-chunking, no overlap) ---
for f in C.glob("*.md"):
    for bad in re.findall(r"(?i)\b(as (?:noted|mentioned) above|see above|as described above|the previous section|listed above)\b", f.read_text()):
        fail.append(f"DANGLING REF in {f.name}: '{bad}'")

# --- 6. out-of-scope gaps must be genuinely absent ---
for term in ("sauna","childcare","child care","pool","massage","nutrition"):
    hits = [f.name for f in C.glob("*.md") if re.search(rf"(?i)\b{term}\b", f.read_text())]
    if hits: fail.append(f"GAP LEAKED: '{term}' appears in {hits} (breaks abstention eval)")

# --- 7. chunk counts ---
print("\nchunks per file (## sections):")
tot = 0
for f in sorted(C.glob("*.md")):
    n = 1 if "atomic: true" in f.read_text() else len(re.findall(r"^## ", f.read_text(), re.M))
    tot += n
    print(f"  {f.name:22} {n}")
print(f"  {'TOTAL':22} {tot}")

print("\n" + ("FAILURES:\n" + "\n".join("  ! "+x for x in fail) if fail else "consistency: ALL CHECKS PASS"))
