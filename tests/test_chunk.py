"""Checks for src/chunk.py, offline. Also asserts every chunk id cited by
eval/golden.yaml resolves, which stops the chunker and the eval drifting apart.
"""
import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import yaml

from src.chunk import all_chunks

chunks = all_chunks()
per_file = collections.Counter(c.source_file for c in chunks)
fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)


check(len(chunks) == 26, f"expected 26 chunks, got {len(chunks)}")
check(per_file["schedule.md"] == 1, f"schedule.md must be 1 atomic chunk, got {per_file['schedule.md']}")
check(per_file["policies.md"] == 5, f"policies.md should be 5 chunks, got {per_file['policies.md']}")
check([c.is_atomic for c in chunks].count(True) == 1, "exactly one chunk should be atomic")

for c in chunks:
    check(c.content.startswith(c.heading_path), f"{c.id}: content does not start with its heading path")
    check(c.id.startswith(c.source_file + "#"), f"{c.id}: id is not <file>#<slug>")
    check("|" not in c.content, f"{c.id}: pipe table survived into the chunk text")

sched = next(c for c in chunks if c.is_atomic)
rows = re.findall(r"^\w+day .*: .* with .*, Level \d, \d+ minutes\.$", sched.content, re.M)
check(len(rows) == 27, f"schedule chunk should render all 27 classes as sentences, got {len(rows)}")
for day in ("Monday", "Thursday", "Saturday", "Sunday"):
    check(day in sched.content, f"schedule chunk missing expanded day name {day}")
check("6am" in sched.content, "schedule chunk missing informal bare-hour time (§7a)")
check("Thu " not in sched.content, "schedule chunk still contains an unexpanded day abbreviation")

# the chunker's ids are the contract eval/golden.yaml cites
ids = {c.id for c in chunks}
golden = yaml.safe_load(open(pathlib.Path(__file__).parent.parent / "eval" / "golden.yaml"))
for e in golden:
    for s in e["sources"]:
        check(s in ids, f"{e['id']}: golden cites '{s}', which no chunk id matches")

print(f"chunks: {len(chunks)}  files: {dict(per_file)}")
print("FAILURES:\n" + "\n".join("  ! " + f for f in fail) if fail else "chunking: ALL CHECKS PASS")
sys.exit(1 if fail else 0)
