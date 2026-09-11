# Studio Knowledge Base — RAG Pipeline Plan

A retrieval-augmented QA system over a fitness studio's knowledge base, built to answer
inbound SMS questions. This document is the output of a design grilling; every decision
below was made deliberately, with the rejected alternative and the reason recorded.

**Vector store:** Pinecone (Documents API).

## STATUS — all 10 phases complete, 2026-09-04

| phase | state |
|---|---|
| 1. Author corpus | **done** — `content/`, 8 files, 26 chunks, `check_corpus.py` passes |
| 2. Verify Pinecone API | **done** — spike run against live account, §7 verified |
| 3. `chunk.py` | **done** — 26 chunks, schedule atomic, `tests/test_chunk.py` passes |
| 4. `embed.py` + ingest | **done** — 26 vectors in `studio-kb`, idempotent across runs |
| 5. `fuse.py` | **done** — `tests/test_fuse.py` passes, incl. the live §4 regression |
| 6. `rag ask` | **done** — hand-checked across all six question categories |
| 7. Golden eval set | **done** — 55 Q, `eval/audit_golden.py` passes, vocab overlap 0.21 |
| 8. Tier-1 eval | **done** — recall@8 **97.8%**, hit@k **100%**, MRR 0.711 (§12) |
| 9. Threshold calibration | **done** — `SCORE_FLOOR=0.20`, abstention precision **1.00** |
| 10. Tier-2 + control | **done** — correctness **100%**, faithfulness **100%**, RAG beats the control on quality *and* cost (§12) |
| 11. Twilio SMS transport | **done** — `src/sms.py`, async by measurement (§13), `tests/test_sms.py` passes |
| 12. Conversation history | **done** — `src/history.py`, SQLite, `tests/test_history.py` passes (§14) |
| 13. Multi-turn eval + query rewrite | **done** — 14 scripted conversations; retrieval 10/12 → **12/12** (§14a) |

Run everything: `./rag ingest` · `./rag ask "…"` · `./rag eval retrieval|answers|control|sweep`
· `./rag serve`. Regression suite, all seven must pass: `check_corpus.py`, `eval/audit_golden.py`,
`tests/test_config.py`, `tests/test_chunk.py`, `tests/test_fuse.py`, `tests/test_sms.py`,
`tests/test_history.py`.

Environment: `.venv` on Python 3.11 with `pinecone 10.0.0`, both API keys in `.env`. Rebuild
with `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

**The open question is answered.** Plain RRF (weight 1.0, depth 20) scored *worse than dense
alone* across the 55 questions — recall@5 73.3% vs 86.7%, MRR 0.619 vs 0.813 — by exactly the
§4 mechanism: gold chunks sitting at `dense=1, bm25=None` landed at fused rank 18. But BM25
alone rescues 3 gold chunks dense misses at top-5, so the arm earns its place once
down-weighted. Calibrated: `RRF_TEXT_WEIGHT=0.1`, `RRF_TEXT_DEPTH=5`. See §12.

---

## 1. Scope

**In:** ingest → chunk → embed → upsert to Pinecone → hybrid retrieve → grounded answer,
driven by a CLI. Plus a two-tier eval harness. **And, added after the eval was green, the
Twilio SMS transport (§13)** — deliberately built last, not first.

**Out (deliberately):** conversation memory across messages, authentication, per-customer
account data ("when is *my* next class?").

**Why:** the SMS transport is a solved, boring integration. Retrieval quality is the hard
part and the thing being evaluated. Building the webhook first means debugging ngrok when
you should be debugging recall. The core is transport-agnostic, so a webhook is a thin
later addition.

That prediction held: adding SMS required **zero changes** to any retrieval module.
`src/sms.py` is a shell over `answer.answer()`. What it did NOT hold on was the assumption
that the transport would be trivial — the webhook timeout forced an async design and cost
per message turned out to hinge on character encoding. See §13.

---

## 2. Locked decisions

| # | Decision | Choice | Why this over the alternative |
|---|---|---|---|
| 1 | Build boundary | Core + eval harness, CLI-driven | Isolates retrieval quality from transport debugging |
| 2 | Corpus source | Synthetic markdown in `content/`, YAML frontmatter | Nothing exists yet; a fixed hand-authored corpus is the only way eval stays reproducible |
| 3 | Schedule representation | **One atomic, never-split chunk** | Per-class chunks truncate at `k` — 8 Saturday classes, k=5, bot lists 5 and sounds certain. Silent wrong answers are the worst failure for a customer-facing bot |
| 4 | Embeddings | `openai/text-embedding-3-small` (1536) via OpenRouter, BYO vectors — **verified live**: OpenRouter does serve `/api/v1/embeddings` | Integrated Pinecone inference would delete a module, but composing it with a `full_text_search` field is unverified (§4), and it would make the embedding model a property of the index rather than an eval variable |
| 5 | Prose chunking | Split on `##`, no overlap, prepend doc title + heading path | Overlap exists to rescue fixed-size splitters; heading sections are already complete units, so it would only manufacture near-duplicates that crowd top-k |
| 6 | Retrieval | Hybrid: dense cosine + BM25 `full_text_search`, fused with **weighted** RRF (k=60, BM25 weight 0.1, BM25 depth 5), **top-8** — both calibrated on the golden set, §12 | This domain is full of exact rare tokens — prices, trainer names, day-of-week — where embeddings are mushy and lexical search is sharp. Also rescues the diluted atomic schedule chunk |
| 7 | Index type | Pinecone **document index** (`SchemaBuilder`), one index, both field types | Vectors-API sparse+dense fuses server-side but needs an `alpha` constant across incompatible score ranges — the hand-tuned normalization RRF exists to avoid |
| 8 | Fusion location | **Client-side RRF in Python** | *Forced, not chosen.* Pinecone's Documents API has no server-side rank fusion. Costs two round trips |
| 9 | Generation model | `google/gemini-3.8-flash`, swappable via `.env` | See §3 — price is a non-constraint at this volume; selected for instruction-following on abstention |
| 10 | Abstention | Raw dense score floor **0.20** (short-circuit, no LLM call) **+** strict grounding prompt | Two independent nets, one fully deterministic. Measured: the floor is a *thin backstop* catching 2 of 10 out-of-scope questions; the prompt net caught 10 of 10 (§12) |
| 11 | Answer length | ~320 char target, 480 hard cap | 160 forces dropping facts on multi-part answers like pricing tiers, which just triggers a second inbound text |
| 12 | Client | `pinecone` SDK + `PINECONE_API_KEY` | Only path to the Documents API |
| 13 | Eval | Custom ~150-line harness, two tiers | Ragas/DeepEval bring heavy dep trees and abstractions around 50 questions; you'd write more config than code and trust your numbers less |
| 14 | Re-ingest | Full wipe + rebuild, idempotent | At 30 chunks embedding costs a fraction of a cent. Incremental hashing is premature optimization that introduces a real bug class: stale chunks orphaned when a heading is renamed |

### Moot after the Supabase → Pinecone switch

- **The deliberate no-ANN-index decision.** On pgvector, skipping HNSW gave *exact* search
  over ~30 rows. Pinecone is ANN and doesn't expose that knob. At this corpus size recall
  should be effectively perfect regardless, but it is no longer a guarantee you control —
  and it is therefore something Tier-1 eval should confirm rather than assume.
- `schema.sql`, the `match_chunks` RPC, psycopg, `DATABASE_URL`, Row Level Security.

---

## 3. Two things worth stating plainly

**The corpus fits in a context window.** The whole KB is ~3–8k words. "Stuff everything in
the prompt" is a legitimate baseline, so the eval includes it as a control (§7). RAG has to
justify itself on cost and latency, not just accuracy. If it doesn't beat the control, that
is a real finding, not a bug.

**Price is not the real constraint on model choice.** Per question this pipeline sends
~1,000 tokens in and ~50 out. At 500 questions/month:

| Model | $/M in / out | Cost/month |
|---|---|---|
| `z-ai/glm-5.3-flash` | 0.075 / 0.25 | ~$0.05 |
| `qwen/qwen3.8-flash` | 0.15 / 0.47 | ~$0.10 |
| `google/gemini-3.8-flash` | 0.75 / 3.75 | ~$0.47 |

The spread across the entire range is forty cents a month. One confidently wrong price
quote costs an actual customer. So the model is selected for reliable *refusal*, not price —
and the ID lives in `.env` so eval can settle it with data instead of assertion.

---

## 4. Traps this design specifically avoids

**Retrieval never returns empty.** Someone asks "do you have a sauna?" and with 30 chunks
and k=5 they get back five confident-looking studio chunks about amenities and what to
bring. The model sees plausible context and feels pressure to synthesize. This is where
prompt-only grounding rules erode — hence the deterministic pre-generation gate.

**You cannot threshold on the RRF score.** RRF is computed purely from rank position: the
top result always scores `1/(60+1)` whether it is a perfect match or completely irrelevant.
Gating on the fused score is a no-op that *looks* like it works. The abstention gate must
read the **raw dense similarity score, captured before fusion** — so the dense result list
must be kept intact, not discarded once fused.

**A text-match filter is a hard filter.** Pinecone's one-call alternative
(`$match_any` + dense ranking) *excludes* every document lacking the token. "Do you run
anything early before work?" shares no tokens with any document and returns zero results —
abstaining on an answerable question, the worst failure mode. Rejected for this reason.

**Table abbreviations defeat BM25 — VERIFIED IN SPIKE.** The schedule table stores `Thu`
and `6:30 AM`. A customer texts "6am class thursday". Those share **zero** tokens, so BM25
does not return the schedule chunk at all, while an irrelevant chunk matches on the common
word "class". Worse, RRF then *demoted* the correct chunk from dense rank 1 to fused rank 3,
because RRF rewards appearing in both lists — a doc mediocre in both arms beats a doc that
is rank 1 in one arm and absent from the other. **Hybrid retrieval was worse than dense
alone on this query.** Fix (validated): expand day abbreviations and normalise times into
the schedule chunk's indexed text at ingest. After expansion BM25 ranks it 1st and fused
RRF returns it 1st. See §7a.

---

## 5. Data flow

```
content/*.md
    │  chunk on ##, prepend "Doc Title > Heading" ; schedule.md → 1 atomic chunk
    ▼
embed via OpenRouter (text-embedding-3-small, 1536)
    ▼
Pinecone document index "studio-kb"
    fields: content (string, full_text_search)  ← BM25
            embedding (dense_vector, 1536, cosine)
            metadata: source_file, heading_path, is_atomic
    │
    ▼
question ──> embed ──> ┌─ dense search  (top 20) ──┐   keep raw top-1 score
                       └─ BM25 search   (top 20) ──┤
                                                   ▼
                                        RRF(60) in Python → top 5
                                                   │
                                    raw dense top-1 < THRESHOLD ?
                                       │yes                   │no
                                       ▼                      ▼
                                 fallback message      strict grounded prompt
                                 (no LLM call)         → gemini-3.8-flash
                                                              │
                                                              ▼
                                   {answer, citations[], top1_score, chunks[]}
                                    └─ answer → SMS;  rest → eval + debugging only
```

---

## 6. Repo layout

```
content/                 8 authored markdown files (the corpus)
  schedule.md            weekly class table — ingested as ONE atomic chunk
  pricing.md             membership tiers, drop-in, class packs
  intro-offer.md         intro offer terms and eligibility
  trainers.md            trainer bios, one ## per trainer
  policies.md            cancellation, no-show, freeze
  difficulty.md          class difficulty levels explained
  logistics.md           parking, arrival, facilities
  what-to-bring.md       equipment, attire, rentals

check_corpus.py          cross-file consistency guard — run after ANY content edit

rag                      shell wrapper: ./rag <cmd>  ->  .venv/bin/python -m src.cli

src/
  config.py              env loading, model IDs, k, RRF constants, threshold
  chunk.py               markdown → chunks (heading split + path prefix + atomic rule + §7a)
  embed.py               OpenRouter /embeddings, batched
  store.py               pinecone: create index, wipe, upsert, dense search, BM25 search
  fuse.py                weighted RRF + retrieve(); carries the raw dense top-1 score out
  answer.py              gate → prompt → generate → structured result
  sms.py                 Twilio webhook: signature check → async answer → REST reply
  cli.py                 ingest | ask | eval | serve

tests/
  test_chunk.py          phase-3 checks, incl. golden.yaml's cited chunk ids all resolving
  test_fuse.py           phase-5 checks: RRF maths, the weighting bound, live §4 regression
  test_sms.py            signature vector, dedup, GSM-7 encoding — all offline
  test_config.py         code defaults match .env.example and the values eval was run at
  test_history.py        storage, name detection, per-customer isolation

eval/
  golden.yaml            ~50 Q → expected chunk(s); includes out-of-scope cases
  run.py                 retrieval | answers | control | sweep
  audit_golden.py        validates golden.yaml against content/ — run after either changes
  GOLDEN_BRIEF.md        the (independent) brief golden.yaml was written from
  .cache/                query embeddings, so Tier-1 reruns cost nothing

.env                     secrets (created; you paste keys)
.env.example             committed template
```

---

## 7. Index setup — VERIFIED against pinecone 10.0.0

**Runtime: Python 3.11+ and `pinecone>=10`.** Python 3.9 silently resolves to pinecone
7.3.0, which has no Documents API and no `SchemaBuilder` at all. The wheel requires
`>=3.10`, so pip gives no error — just an old SDK missing the feature the design needs.

```python
from pinecone import Pinecone, SchemaBuilder
from pinecone.models.documents.score_by import TextQuery, DenseVectorQuery

schema = (SchemaBuilder()
          .add_string_field("content", full_text_search=True, language="en")
          .add_dense_vector_field("embedding", dimension=1536, metric="cosine")
          .build())

pc.indexes.create(name="studio-kb", schema=schema,
                  deployment={"deployment_type": "managed",
                              "cloud": "aws", "region": "us-east-1"})   # blocks until ready

idx.documents.upsert(namespace="__default__", documents=[
    {"_id": "c1", "content": ..., "heading_path": ..., "embedding": [...]}])

dense = idx.documents.search(namespace=NS, top_k=20,
            score_by=[DenseVectorQuery(field="embedding", values=qv)],
            include_fields=["content", "heading_path", "source_file"])
bm25  = idx.documents.search(namespace=NS, top_k=20,
            score_by=[TextQuery(fields=["content"], query=question)],
            include_fields=["content", "heading_path", "source_file"])
# matches: m.id, m.score, m.<field>
```

Gotchas confirmed the hard way:

- **Only searchable fields go in the schema.** Declaring `heading_path` is a 400. Metadata
  is passed on the document and auto-indexed on first appearance.
- **`upsert_records()` is a 400 on a document index** — writes go through `idx.documents`.
- **`TextQuery(field=...)` is deprecated**; use `fields=[...]`.
- **`DenseVectorQuery` takes `values=`**, not `vector=`.
- A `dense_vector` clause **must appear alone** in `score_by`; the server rejects combining
  it with `text`. This is what makes decision #8 forced rather than chosen.

At ~30 vectors this sits comfortably inside Pinecone's free tier.

## 7a. Ingest-time normalisation (added after the spike)

Before embedding or indexing the schedule chunk, rewrite its indexed text to spell out what
the table abbreviates:

- `Mon` → `Monday`, `Thu` → `Thursday`, … (all seven)
- keep the original times, and additionally emit a bare-hour form (`6:00 AM` → also `6am`)
- render each row as a sentence rather than pipes:
  `"Thursday 6:30 AM Strength Circuit with Dev Okonkwo, Level 2, 50 min."`

This is the single highest-value line of ingest code in the build: without it BM25 cannot
match any day-of-week or informal-time question against the schedule, which is the largest
category of inbound SMS. The prose rendering also embeds better than pipe-delimited rows.

---

## 8. Eval design

`eval/golden.yaml` — 55 entries, written **before any tuning**, or you tune to the test.
Composition:

- exact-fact lookups (prices, times, durations)
- paraphrases that share no vocabulary with the source text
- enumeration ("everything on Saturday") — the atomic-chunk stress test
- multi-hop (cancellation policy *for* the intro offer)
- **out-of-scope** (~8): sauna, childcare, weather — these calibrate the score threshold
- adversarial: prompt injection, requests for personal account data

**Tier 1 — `rag eval retrieval`.** No generation calls; query embeddings cached in
`eval/.cache/` after the first run, so iteration costs only Pinecone queries.
`recall@k`, `MRR`, `abstention precision/recall`, plus a per-failure list showing the gold
chunk's actual rank. *This is where the wins are* — if the right chunk isn't retrieved, no
model can rescue the answer. Now also doubles as the check that Pinecone's ANN recall is in
fact perfect at this corpus size (§2, moot decisions).

**Tier 2 — `rag eval answers`.** Adds LLM-as-judge `faithfulness` (is every claim supported
by the retrieved context?) and `correctness` (are the required facts present?). The judge
**must be a different model than the generator** — a model grading its own output has a
documented self-preference bias. Correctness is judged on facts present, not string
similarity, so a terser correct answer isn't penalized.

**Control run.** Same questions, entire KB in the prompt, no retrieval. ~10 lines. Tells you
whether retrieval is earning its complexity.

---

## 9. Build phases

| # | Step | Verify |
|---|---|---|
| 1 | Author `content/*.md`; you correct the facts | `python3 check_corpus.py` passes: 26 chunks, and levels/trainers/teaching-days join across files |
| 2 | `store.py` index creation (spike DONE — API verified §7) | Index exists with both a `full_text_search` field and a 1536-dim dense field |
| 3 | `chunk.py` | Unit test: 26 chunks total; schedule → exactly 1; policies → 5; every chunk's embed input starts with its heading path |
| 4 | `embed.py` + upsert + `rag ingest` | Vector count matches chunk count; running ingest twice leaves the count unchanged (idempotent) |
| 5 | `fuse.py` | Unit test on synthetic rank lists. **Also assert the §4 regression**: the schedule chunk must win `"is there a 6am class thursday"` after §7a normalisation |
| 6 | `rag ask "..."` end to end | 5 hand-checked questions return correct answers with sane citations |
| 7 | `eval/golden.yaml` (55 Q) | `python3 eval/audit_golden.py` passes: 0 errors, every section covered, vocab overlap < 0.3 |
| 8 | Tier-1 eval | `recall@5` reported; failures list gold-chunk ranks |
| 9 | Calibrate score threshold on out-of-scope set | Abstention precision 1.0 (never refuses an answerable question) with recall as high as that allows |
| 10 | Tier-2 eval + control run | Faithfulness/correctness reported; RAG vs full-context compared on quality *and* cost |

Tuning knobs, in the order worth trying: `k` → chunk granularity → RRF weighting →
generation model.

---

## 10. Deferred (noted, not built)

- Conversation state across messages ("and what about Sunday?")
- Pinecone's hosted reranker as a third stage — currently would reorder half the corpus
- Sparse-vector field (`pinecone-sparse-english-v0`) as a third RRF input alongside BM25
- Incremental re-embed by content hash
- Schedule as a real structured table with a query router — most correct on temporal
  questions, but it leaves the RAG path and therefore leaves what your eval measures

---

## 11. Setup you need to do

1. Create a Pinecone account; get an API key from the console → `PINECONE_API_KEY` in `.env`.
2. Get an OpenRouter key from <https://openrouter.ai/keys> → `OPENROUTER_API_KEY` in `.env`.
3. **For SMS only** (the CLI and eval need none of this): buy a Twilio number, then put
   `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` and `TWILIO_FROM_NUMBER` in `.env`. Expose
   `rag serve` on a public URL, set `TWILIO_WEBHOOK_URL` to that exact URL, and paste the
   same string into the console's "A message comes in" webhook field. See §13.

Both values go in the already-created `.env`. Nothing else is required to start —
the index itself is created by `rag ingest --init`.

---

## 12. Results — measured, not assumed

All numbers from one clean run each on the 55-question golden set, at the locked config
(`TOP_K=8`, `RRF_TEXT_WEIGHT=0.1`, `RRF_TEXT_DEPTH=5`, `SCORE_FLOOR=0.20`).

### Tier 1 — retrieval (45 answerable questions)

| arm | recall@8 (all gold sources) | hit@8 (any) | MRR |
|---|---|---|---|
| **fused** | **97.8%** | **100.0%** | 0.711 |
| dense only | 93.3% | 97.8% | 0.813 |
| bm25 only | 66.7% | 77.8% | 0.492 |

Pinecone's ANN recall is not a problem at this corpus size — the moot-decision worry from
§2 is closed. Fusion beats dense alone on recall and hit@k and loses on MRR, which is the
right trade when everything retrieved is passed to the generator anyway.

### The two tuning knobs, settled with data

**BM25 weight.** Sweeping weight × depth over the golden set (`rag eval sweep`):

| config | recall@5 | hit@5 | MRR |
|---|---|---|---|
| dense only (weight 0) | 86.7% | 95.6% | 0.813 |
| **weight 0.1, depth 5** | **88.9%** | **97.8%** | 0.711 |
| plain RRF (weight 1.0, depth 20) | 73.3% | 80.0% | 0.619 |

At weight 0.1 a BM25 rank-1 hit can climb at most 7 dense positions — enough to rescue the
3 gold chunks BM25 alone finds, not enough to scramble a good dense ordering. That bound is
pinned in `tests/test_fuse.py` so a future weight change has to face it.

**k.** At k=5, three answerable questions were *correctly* refused because their gold chunk
sat at fused rank 6–8 — the system declining rather than inventing, which is the design
working, but a lost answer. k=8 fixed all three:

| | k=5 | k=8 |
|---|---|---|
| judge correctness | 93.3% | **100%** |
| wrongly abstained | 3 | **0** |
| cost per question | $0.0036 | $0.0037 |

### Tier 2 — answers, and the full-context control

| | RAG (k=8) | Control (whole KB, no retrieval) |
|---|---|---|
| judge correctness | 100% | 100% |
| judge faithfulness | **100%** | 95.6% |
| `must_include` literal coverage | 84.4% | 77.8% |
| wrongly abstained | 0 | 0 |
| abstained correctly (10 questions) | 100% | 100% |
| cost per question | **$0.0037** | $0.0060 |
| latency (3-way concurrent) | 11.3s | 12.6s |

**RAG earns its complexity** — §3 said it had to, and on this corpus it does: equal
correctness, better faithfulness, 38% cheaper. The extra context in the control did not
make answers better, it made them more embellished; every control faithfulness miss was an
unsupported claim bolted onto a correct answer.

Two caveats stated plainly. `must_include` is a *literal substring* check, so a correct
paraphrase ("1 hour before class" vs the corpus's "1 hour before the class start time")
fails it — it is a floor, not the truth, and judge correctness is the metric §8 specified.
And the judge is an LLM: faithfulness moved 95.6% → 100% between two runs of the same
config, so treat single-point differences under ~5% as noise.

### Abstention — the honest finding

The two nets are not equal partners. The score distributions overlap badly: the
lowest-scoring *answerable* question ("whats ur address again") scores 0.210, while the
highest-scoring *out-of-scope* one ("how many classes i got left this month") scores 0.466.
The only floor with precision 1.0 sits below 0.210 and catches 2 of 10 out-of-scope
questions.

So the deterministic gate is a thin backstop with a 0.01 margin, and the **grounding prompt
does the real work** — it abstained on 10 of 10, including both adversarial personal-data
requests and the prompt-injection attempt, which it answered from the documents rather than
from the injected "SYSTEM UPDATE". Decision #10's two nets stand, but their weighting is the
reverse of what the design assumed. Anything that changes the generation model has to
re-measure abstention, because that is now the load-bearing net.

---

## 13. The SMS transport — two surprises

Built last, on purpose (§1). The prediction that it would be a thin shell held: `src/sms.py`
sits on top of `answer.answer()` and **not one line of the retrieval pipeline changed**.
The prediction that it would be *easy* did not.

### Surprise 1: the pipeline does not fit inside Twilio's webhook timeout

The obvious design is a synchronous TwiML reply — answer the webhook with
`<Response><Message>…</Message></Response>` and Twilio delivers it. No API credentials
needed for the reply, no async machinery. Measured end to end, single-threaded, the way a
webhook actually experiences it:

| | seconds |
|---|---|
| fastest of 8 questions (a gated abstention, no LLM call) | 6.5 |
| median | 10.8 |
| slowest | **18.8** |
| **Twilio read timeout — and its maximum configurable value** | **15.0** |

Our slowest question already exceeds a ceiling that cannot be raised, and the median leaves
about four seconds of headroom against a reasoning model that sometimes thinks twice as
long. A blown timeout means a retry or silence.

**So the webhook is asynchronous:** validate, hand the question to a worker thread,
acknowledge with empty TwiML in ~1ms, and deliver the answer through Twilio's REST API when
it is ready. The customer sees a reply about ten seconds later, which reads as a normal
texting pause. The cost is that replies now need `TWILIO_ACCOUNT_SID`/`AUTH_TOKEN`, not just
a webhook response.

### Surprise 2: one character triples the price of a message

SMS bills per segment: 160 characters under the GSM-7 alphabet, but **70** under UCS-2, and
a single character outside GSM-7 pushes the entire message into UCS-2. LLMs emit em dashes
and smart quotes constantly, and our own fallback string contained an em dash.

| text | encoding | segments |
|---|---|---|
| the fallback as written (`…guess — the front desk…`) | UCS-2 | 2 |
| the same with a plain hyphen | GSM-7 | **1** |
| a typical 153-char answer with one curly apostrophe | UCS-2 | 3 |
| the same with a straight apostrophe | GSM-7 | **1** |

`to_gsm7()` transliterates the common offenders at the transport boundary only — the em
dash is still fine in the CLI, in eval and in the logs. Characters with no safe equivalent
(an emoji, an accented name) are deliberately left alone and correctly fall back to UCS-2;
cheaper is not worth wrong. Measured on four live questions afterwards: 8 segments total,
all GSM-7.

### What else the webhook has to do

- **Validate `X-Twilio-Signature`.** The endpoint is public and every accepted request
  spends an LLM call and an outbound SMS, so this is the only thing between a leaked URL
  and someone else's bill. Implemented in stdlib rather than pulling in the twilio SDK,
  and pinned in `tests/test_sms.py` against **Twilio's own published test vector** — a
  self-generated vector would only prove the code agrees with itself.
- **Deduplicate on `MessageSid`.** Twilio can redeliver a webhook; without this one
  redelivery means a second LLM call and a second text to the customer.
- **Reply on failure.** A crash in the pipeline sends a distinct apology rather than the
  grounded fallback (which would claim we looked and found nothing — a lie after a crash).
  An unanswered text is the one outcome a customer definitely notices.
- **Bound the worker pool**, so a burst of texts queues instead of spawning unlimited
  threads and hammering OpenRouter into rate limits.
- **Rate limit per sender** — see below.

### Rate limiting

Every accepted message costs an LLM call plus a two-to-three segment reply, and the endpoint
is public, so one abusive number is a real bill. A sliding window of **15 messages per hour
per sender** (`SMS_RATE_LIMIT` / `SMS_RATE_WINDOW`) sits far above what a customer asking
follow-up questions sends and far below what abuse looks like.

Three details that are the whole point:

- The check runs **before** retrieval, so a rejected message costs nothing at all.
- A sender crossing the limit gets **exactly one** notice, then silence until the window
  rolls off. Replying to every over-limit message would hand an abuser a free outbound SMS
  per text — the precise cost the limit exists to prevent.
- Rejected messages do **not** count towards the window, so someone who keeps texting is
  not punished with an ever-extending block; they get their next slot as the oldest
  timestamp ages out.

Deduplication runs before the limit, so a webhook Twilio redelivered never eats the
sender's quota. Both the dedup table and the rate-limit table are bounded, because a
long-running process must not accumulate a row per number it has ever seen.

### Running it

```
rag serve --port 5000          # then expose that port publicly (ngrok, Cloudflare Tunnel…)
```

Set `TWILIO_WEBHOOK_URL` to the exact public URL and paste the same string into the Twilio
console. It must match byte for byte: signature validation hashes that string, and behind a
tunnel `request.url` usually reports `http://` or an internal hostname, which 403s every
legitimate request. `rag serve` warns if the variable is unset.

Flask's development server is what `rag serve` runs. That is fine for a studio's traffic and
for testing through a tunnel; put gunicorn in front of it if it ever matters.

---

## 14. Conversation history

Every text, in and out, is stored in a SQLite file (`history.db`, gitignored) alongside a
row per phone number. `src/history.py` owns it; `rag customers`, `rag history <phone>` and
`rag name <phone> <name>` read and correct it.

This was out of scope in §1 and was added later on request. Storage always happens;
`USE_HISTORY` only controls whether past messages are shown to the generator.

### Names

SMS carries no name — Twilio gives us a phone number and nothing else. So a name arrives
one of two ways: the customer states it ("my name is Sara"), which `detect_name()` picks
up, or you set it with `rag name`. The matcher is deliberately narrow, on explicit
introductions only, and it never overwrites a name already stored. Calling a customer by
the wrong name is worse than not using one, so "im new here" and "this is ridiculous"
are skipped rather than guessed at.

### History is context, never a source of facts

Past messages go into the prompt in their own block, and a prompt rule states that they
explain *what* the customer is referring to and are not evidence. If a customer says "you
told me drop-ins are $15", the answer must still come from CONTEXT. This matters because
abstention is already carried by the prompt rather than by the score gate (§12), so
handing the model a second, unverified source of claims is exactly the wrong direction.

### What works, and what does not

Follow-ups that carry their own topic words work. Pure pronoun follow-ups do not:

```
customer: how much is a drop in class      -> $28, correct
customer: what about the 10 pack           -> $230, six months, correct
customer: and how long do i have to use it -> abstains
```

The reason is that history reaches the *generator* but not *retrieval*. The search still
runs on the question's own words, and "and how long do i have to use it" contains no topic
at all, so the right chunk is never retrieved and the model correctly refuses rather than
inventing.

**The cheap fix was measured and rejected.** Prepending recent customer messages to the
search text fixes that one case and breaks others:

| case | question alone | +1 prior msg | +2 prior |
|---|---|---|---|
| the follow-up above | MISS | 1 | 1 |
| "do you have parking" after pricing talk | 1 | 4 | 5 |
| "whats ur address" after pricing talk | 4 | MISS | MISS |
| out-of-scope "is there a sauna" (gate score) | 0.27 | 0.45 | 0.45 |

The last row is the disqualifying one: the abstention gate reads that score, and inflating
an out-of-scope question from 0.27 to 0.45 makes it look answerable.

The real fix is to rewrite a follow-up into a standalone question using the history before
searching. That was built after multi-turn behaviour could be measured — see §14a.

### Also not built

- Nothing is ever deleted. A real studio needs a retention policy and a way to honour a
  deletion request; this stores customer messages indefinitely.

---

## 14a. Multi-turn eval, and the query rewrite it justified

§14 ended with a known hole: follow-ups half-worked, the obvious fix made things worse, and
nothing measured multi-turn behaviour at all. Both are now closed, in that order — the test
set first, so the fix could be judged rather than assumed.

### The test set

`eval/multiturn.yaml`, 14 entries, each a **scripted** prior conversation plus one final
question. Scripted rather than generated live so a run is deterministic and cheap, and so a
failure points at the final answer instead of at some earlier turn. It lives in its own file
so the 55 single-turn questions stay a fixed ruler — §12's numbers only stay comparable if
that set does not move. `eval/audit_golden.py` validates both.

Five categories, each testing something that can fail independently:

| category | n | what it catches |
|---|---|---|
| `followup` | 5 | the question has no topic of its own ("how much are they") |
| `topic_change` | 3 | a self-contained question after unrelated chat — must not be dragged off course |
| `out_of_scope` | 2 | a conversation in progress must not make an unanswerable question look answerable |
| `false_claim` | 2 | the CUSTOMER asserts something the corpus contradicts |
| `false_history` | 2 | the BOT's own earlier reply was wrong — it must not propagate |

Writing these caught a flaw in the set itself: the first `false_history` entry put the false
claim in the customer's message, not the bot's, which is a different test. The audit now
refuses a `false_history` entry whose prior bot turn is corpus-accurate.

### Before and after

Retrieval and answering are reported separately, because they fail for different reasons and
only the first is what rewriting touches.

| | retrieved | correct | faithful | failures |
|---|---|---|---|---|
| rewrite **off** | 10/12 | 12/14 | 14/14 | 3 |
| rewrite **on** | **12/12** | **14/14** | **14/14** | **0** |

Per category, the thing worth checking is that nothing regressed: `followup` retrieval went
3/5 → 5/5, while `topic_change` stayed 3/3 and `out_of_scope` still abstained 2/2. The
earlier concatenation experiment (§14) bought the same follow-up win by wrecking both.

### How it works, and what it costs

Before retrieval, a small model turns the latest message into a standalone search query
using the conversation. The instruction that carries the whole design is the first one: **a
question that already stands on its own must come back unchanged.** That is what keeps topic
changes and out-of-scope questions intact.

```
"and how long do i have to use it"  ->  "how long do i have to use the 10-class pack"
"who teaches the first one"         ->  "who teaches Power Yoga at 7:30 AM on Saturday"
"do you have parking"               ->  "do you have parking"          (unchanged)
"is there a sauna"                  ->  "is there a sauna"             (unchanged)
```

Cost: about **$0.0006** per message and roughly **3 seconds** (8.7s → 11.6s on a follow-up).
It only fires when there IS history, so a first message is untouched. Latency is affordable
only because the webhook is already asynchronous (§13); on a synchronous TwiML reply this
would have pushed straight through Twilio's 15-second ceiling.

Any failure in the rewrite returns the original question, so a broken or slow rewrite
degrades to the old behaviour rather than breaking answering.

The generator still sees the customer's real words. The rewrite is used for **searching
only** — it never becomes the question the model answers.

### The judge model changed

`qwen/qwen3.8-flash` began returning sustained upstream 429s, losing roughly a third of the
judge calls in a run even when the eval was serialised. An eval you cannot trust is worse
than a slow one, so the judge is now `z-ai/glm-5.3-flash` — still different from the
generator, which is what the rule in §8 actually requires. Single-turn Tier 2 was re-run on
the new judge and is unchanged: correctness 100%, faithfulness 100%, 0 wrong abstentions.

---

## 15. Two loose ends closed

**`rag ask --as <phone>`.** Conversations were previously unreachable from the command
line: `rag ask` has no phone number, so no history, so the follow-up rewrite could never
fire. `--as` reads and writes that number's history exactly as the webhook does, and prints
the rewritten search query when it differs from what was typed — so the mechanism in §14a
is something you can watch work rather than take on trust. Without the flag `rag ask` is
unchanged: stateless, and it stores nothing.

**Config defaults were the pre-calibration guesses.** The tuned values lived only in `.env`,
so a checkout without one silently ran `TOP_K=5`, `RRF_TEXT_WEIGHT=1.0`, `RRF_TEXT_DEPTH=20`
and `SCORE_FLOOR=0.35` — the settings the eval measured as *worse*, with nothing to report
it. The defaults in `src/config.py` are now the calibrated values.

`tests/test_config.py` guards it, and does two separate things because a single check would
miss half the problem: it compares every code default against `.env.example` so the two
cannot drift apart, and it asserts the six tuned numbers by name so they cannot drift
*together* into something nobody measured.
