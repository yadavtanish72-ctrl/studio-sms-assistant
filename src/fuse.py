"""Run both search arms and merge them with weighted reciprocal-rank fusion.

RRF uses rank position only, never the arms' own scores, which is why the abstention
gate reads `dense_top1` instead (PLAN §4).
"""
from dataclasses import dataclass, field

from . import config, embed, store


@dataclass
class Retrieval:
    """Everything one query produced. Both raw arms are kept, not just the fused list:
    the gate needs `dense_top1`, and eval/run.py needs per-arm ranks to explain misses."""
    chunks: list                 # fused top-k, each hit plus its rrf_score
    dense_top1: float            # raw cosine similarity, read BEFORE fusion
    dense: list = field(default_factory=list)
    text: list = field(default_factory=list)


def rrf(dense, text, k=None, top_k=None, text_weight=None, text_depth=None):
    """Weighted RRF. `text_weight` < 1 stops a doc that is merely present in both arms
    from outranking a doc the dense arm put first and BM25 never returned (PLAN §4);
    `text_depth` truncates the noisy BM25 tail. Both are calibrated in eval/run.py sweep."""
    # Defaults come from config at call time, not at import — that is what lets
    # `rag eval sweep` try 28 combinations in one process.
    k = config.RRF_K if k is None else k
    text_weight = config.RRF_TEXT_WEIGHT if text_weight is None else text_weight
    text_depth = config.RRF_TEXT_DEPTH if text_depth is None else text_depth
    scores, hits = {}, {}
    for arm, weight in ((dense, 1.0), (text[:text_depth], text_weight)):
        for rank, hit in enumerate(arm, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + weight / (k + rank)
            # First arm to mention a document supplies its content; the arms return the
            # same fields, so whichever wins the race is fine.
            hits.setdefault(hit["id"], hit)
    # Sort by score descending, then by id — the id tiebreak keeps output deterministic,
    # which matters because eval reruns are compared against each other.
    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    return [dict(hits[i], rrf_score=scores[i]) for i in ranked[:top_k or config.TOP_K]]


def retrieve(idx, question, query_vector=None, top_k=None):
    """Both arms, fused. `query_vector` lets eval reuse a cached embedding."""
    qv = query_vector if query_vector is not None else embed.embed_one(question)
    # Two round trips, because the server refuses to score by dense and text at once.
    dense = store.dense_search(idx, qv)
    text = store.text_search(idx, question)
    return Retrieval(chunks=rrf(dense, text, top_k=top_k),
                     # The one number the abstention gate is allowed to read.
                     dense_top1=dense[0]["score"] if dense else 0.0,
                     dense=dense, text=text)
