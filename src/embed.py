"""Text -> 1536-dimension vectors, via OpenRouter.

Ingest and query must use the same model or the vectors live in different spaces.
"""
import httpx

from . import config

BATCH = 64  # the whole corpus is 26 chunks, so this is one request in practice


def embed(texts, api_key=None):
    """Embed a list of strings, returning vectors in the same order. `api_key` bills a web
    visitor's key instead of ours."""
    vectors = []
    headers = {"Authorization": f"Bearer {config.openrouter_key(api_key)}"}
    with httpx.Client(timeout=120) as client:
        for start in range(0, len(texts), BATCH):
            batch = texts[start:start + BATCH]
            r = client.post(f"{config.OPENROUTER_URL}/embeddings", headers=headers,
                            json={"model": config.EMBED_MODEL, "input": batch})
            r.raise_for_status()
            # Sort by the API's own `index` field rather than trusting response order.
            # Order IS the mapping back to chunks — get it wrong and every chunk carries
            # its neighbour's vector, which no test downstream would obviously catch.
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            if len(data) != len(batch):
                raise RuntimeError(f"asked for {len(batch)} embeddings, got {len(data)}")
            vectors.extend(d["embedding"] for d in data)
    # A dimension mismatch means the index schema and the model disagree. Fail here, at
    # ingest, rather than at upsert with a less obvious Pinecone error.
    bad = {len(v) for v in vectors} - {config.EMBED_DIM}
    if bad:
        raise RuntimeError(f"{config.EMBED_MODEL} returned dimensions {bad}, "
                           f"but the index is built for {config.EMBED_DIM}")
    return vectors


def embed_one(text, api_key=None):
    """Query-time convenience: one question -> one vector."""
    return embed([text], api_key=api_key)[0]
