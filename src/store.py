"""Everything that touches Pinecone: create, wipe, upsert, and the two search arms.

A dense clause must appear alone in `score_by`, which is why the arms are two round
trips fused client-side (PLAN §7).
"""
from pinecone import Pinecone, SchemaBuilder
from pinecone.errors import NotFoundError
from pinecone.models.documents.score_by import DenseVectorQuery, TextQuery

from . import config

# Fields to bring back on every hit. `content` is what the generator reads; the rest are
# for citations and debugging.
FIELDS = ["content", "heading_path", "source_file", "is_atomic"]


def client():
    return Pinecone(api_key=config.require("PINECONE_API_KEY"))


def create_index(pc):
    """Create the index if missing, blocking until ready. The schema is the hybrid design:
    one BM25 string field and one cosine vector field."""
    if pc.indexes.exists(config.PINECONE_INDEX):
        return False
    schema = (SchemaBuilder()
              .add_string_field("content", full_text_search=True, language="en")
              .add_dense_vector_field("embedding", dimension=config.EMBED_DIM, metric="cosine")
              .build())
    pc.indexes.create(name=config.PINECONE_INDEX, schema=schema,
                      deployment={"deployment_type": "managed",
                                  "cloud": config.PINECONE_CLOUD,
                                  "region": config.PINECONE_REGION})
    return True


def open_index(pc):
    """Handle to the existing index, or a clear instruction if it was never created."""
    if not pc.indexes.exists(config.PINECONE_INDEX):
        raise SystemExit(f"index '{config.PINECONE_INDEX}' does not exist — run: rag ingest --init")
    return pc.index(name=config.PINECONE_INDEX)


def wipe(idx):
    """Empty the namespace before every rebuild. Incremental updates would orphan chunks
    whose heading was renamed."""
    try:
        idx.documents.delete(namespace=config.PINECONE_NAMESPACE, delete_all=True)
    except NotFoundError:
        pass  # first ingest: the namespace does not exist until something is written


def upsert(idx, chunks, vectors):
    """Write chunks and their vectors together. `_id` is the chunk id, so re-ingesting the
    same content overwrites in place — that is what makes `rag ingest` idempotent."""
    idx.documents.upsert(namespace=config.PINECONE_NAMESPACE, documents=[
        {"_id": c.id, "content": c.content, "heading_path": c.heading_path,
         "source_file": c.source_file, "is_atomic": c.is_atomic, "embedding": v}
        for c, v in zip(chunks, vectors)])


def count(idx):
    """Document count, used by `rag ingest` to prove the write landed."""
    return len(idx.documents.list(namespace=config.PINECONE_NAMESPACE).to_list())


def _hits(response):
    """Normalise SDK documents into plain dicts, so fuse/answer never touch SDK types."""
    return [{"id": m.id, "score": m.score, "content": m.get("content", ""),
             "heading_path": m.get("heading_path", ""), "source_file": m.get("source_file", "")}
            for m in response.matches]


def dense_search(idx, query_vector, top_k=None):
    """Semantic arm. `score` is a real cosine similarity and is what the abstention gate
    reads, so it must reach fuse.py untouched."""
    return _hits(idx.documents.search(
        namespace=config.PINECONE_NAMESPACE, top_k=top_k or config.CANDIDATES,
        score_by=[DenseVectorQuery(field="embedding", values=query_vector)],
        include_fields=FIELDS))


def text_search(idx, query, top_k=None):
    """BM25 arm over the same `content` field. Sharp on rare tokens and noisy otherwise,
    which is why fuse.py weights it at 0.1."""
    return _hits(idx.documents.search(
        namespace=config.PINECONE_NAMESPACE, top_k=top_k or config.CANDIDATES,
        score_by=[TextQuery(fields=["content"], query=query)],
        include_fields=FIELDS))
