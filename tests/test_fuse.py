"""Checks for src/fuse.py; pass --offline to skip the live index.

The last assertion pins how far the BM25 arm can move a document at the calibrated weight.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src import fuse

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)


def hits(*ids):
    return [{"id": i, "score": 1.0 - n / 100, "content": i} for n, i in enumerate(ids)]


# --- synthetic rank lists: plain, unweighted RRF ---
PLAIN = dict(k=60, text_weight=1.0, text_depth=20)

# RRF is convex: 1/(k+1) + 1/(k+3) > 2/(k+2), so mirrored extremes beat the doc that is
# middling in both arms. Ties break on id, so the order is deterministic.
r = fuse.rrf(hits("a", "b", "c"), hits("c", "b", "a"), top_k=3, **PLAIN)
check([h["id"] for h in r] == ["a", "c", "b"], f"mirrored lists mis-ranked: {[h['id'] for h in r]}")

# The PLAN §4 mechanism, stated as a test: under plain RRF a doc at rank 1 in one arm and
# absent from the other loses to a doc that is merely 2nd in both.
r = fuse.rrf(hits("gold", "mid"), hits("other", "mid"), top_k=2, **PLAIN)
check([h["id"] for h in r] == ["mid", "gold"],
      "plain RRF should demote a one-arm rank-1 doc below a doc present in both arms")

# --- the deployed weighting bounds how far BM25 can move a document ---
# At the calibrated weight, a BM25 rank-1 hit can climb at most ~7 dense positions. That
# bound is what keeps the dense ordering intact while still letting BM25 rescue the gold
# chunks it alone finds (3 of them in the golden set).
def beats_dense_top(dense_rank):
    dense = hits(*[f"d{n}" for n in range(1, dense_rank + 1)])
    r = fuse.rrf(dense, [{"id": f"d{dense_rank}", "score": 1.0, "content": ""}], top_k=1)
    return r[0]["id"] == "d1"


check(not beats_dense_top(7), "a BM25 rank-1 hit should still climb 7 dense positions")
check(beats_dense_top(8), "BM25 must not lift a doc from dense rank 8 over a dense rank-1 hit")

r = fuse.rrf(hits("a"), hits("b"), top_k=5, **PLAIN)
check([h["id"] for h in r] == ["a", "b"], "a doc in only one arm must still be returned")
check(abs(r[0]["rrf_score"] - 1 / 61) < 1e-12, "rank-1-in-one-arm score should be 1/(k+1)")

r = fuse.rrf(hits("a", "b"), hits("a", "b"), top_k=5, **PLAIN)
check(abs(r[0]["rrf_score"] - 2 / 61) < 1e-12, "appearing at rank 1 in both arms should sum to 2/(k+1)")

check([h["id"] for h in fuse.rrf(hits(), hits("x"), **PLAIN)] == ["x"], "an empty arm must not break fusion")
check(fuse.rrf(hits(), hits(), **PLAIN) == [], "two empty arms should fuse to nothing")
check(len(fuse.rrf(hits("a", "b", "c"), hits(), top_k=2, **PLAIN)) == 2, "top_k must be honoured")
check(len(fuse.rrf(hits("a"), hits("x", "y", "z"), k=60, text_weight=1.0, text_depth=1)) == 2,
      "text_depth must truncate the BM25 arm before fusion")

# --- PLAN §4 regression, against the live index ---
if "--offline" not in sys.argv:
    from src import store
    idx = store.open_index(store.client())
    q = "is there a 6am class thursday"
    res = fuse.retrieve(idx, q)
    SCHED = "schedule.md#weekly-class-schedule"
    ranks = lambda arm: ([h["id"] for h in arm].index(SCHED) + 1) if SCHED in [h["id"] for h in arm] else None
    print(f"{q!r}\n  dense rank: {ranks(res.dense)}  bm25 rank: {ranks(res.text)}  "
          f"fused rank: {ranks(res.chunks)}  dense_top1: {res.dense_top1:.3f}")
    check(res.chunks[0]["id"] == SCHED,
          f"§4 regression: schedule chunk must fuse to rank 1, got {res.chunks[0]['id']}")
    check(ranks(res.text) == 1, f"§7a: BM25 should rank the schedule chunk 1st, got {ranks(res.text)}")

print("FAILURES:\n" + "\n".join("  ! " + f for f in fail) if fail else "fusion: ALL CHECKS PASS")
sys.exit(1 if fail else 0)
