# Briefing: write the golden evaluation set

You are writing the evaluation set for a question-answering system that answers **inbound
SMS messages** from customers of Northline Studio, a fitness studio. The system will
retrieve passages from the documents in `content/` and answer from them.

Your job is to write `eval/golden.yaml`: ~50 test questions with reference answers.

## Hard constraints

1. **Read only `content/*.md`.** Do not open `PLAN.md`, `check_corpus.py`, `.env`, or
   anything else in this repo. Those describe how the system was built, and knowing that
   would bias you toward questions the system happens to be good at. Your value here is
   that you don't know.
2. **Do not edit `content/`.** It is the fixed fixture. If you find a factual
   contradiction between files, report it — do not fix it, and do not write a question
   that depends on it.
3. Every non-abstain question must be answerable **strictly** from `content/`, and you
   must quote the sentence that proves it.

## The single most important instruction

**Do not write questions in the documents' vocabulary.** You are reading polished prose;
real customers text badly. If you phrase a question using the same words as the source
sentence, retrieval succeeds for the wrong reason and the whole evaluation becomes
worthless flattery.

Write how people actually text a studio:

    bad:  "What is the cancellation policy for booked classes?"
    good: "if i cancel 3 hrs before do i lose the class"

    bad:  "Is on-site parking available at the studio?"
    good: "is there parking or do i need to find street"

    bad:  "What are the membership tier prices?"
    good: "whats the cheapest monthly plan"

Lowercase, missing punctuation, abbreviations, typos, run-ons, no greeting. Some questions
should be terse to the point of ambiguity ("earliest class thurs?"). Aim for questions
that are hard for keyword matching *and* hard for semantic similarity.

## Composition — write roughly these counts

| category | n | what it tests |
|---|---|---|
| `exact_fact` | 12 | a specific number, price, time, or name |
| `paraphrase` | 10 | correct answer exists but shares almost no words with the question |
| `enumeration` | 5 | requires listing **every** matching item, not some |
| `multi_hop` | 8 | answer requires joining facts from **two or more different files** |
| `out_of_scope` | 8 | plausible studio question the documents genuinely do not answer |
| `correction` | 4 | question presupposes something false; the right answer corrects it |
| `adversarial` | 3 | prompt injection, or requests for personal account data |

Notes on the harder categories:

- **out_of_scope**: find the gaps yourself by reading what `content/` covers and asking
  what a customer would reasonably expect that simply isn't there. These must be things a
  studio customer would genuinely text. Do not pick absurd questions — "what's the
  weather" is a weak test; a question that *sounds* like it should be in a studio FAQ is a
  strong one. Verify the answer is truly absent, not merely phrased differently.
- **correction**: the question assumes a fact that is wrong. The system should not refuse,
  and should not agree — it should supply the true fact. Look for close-but-wrong
  assumptions about times, prices, or eligibility.
- **adversarial**: include at least one instruction-injection attempt inside the message
  text, and at least one request for data about a specific customer's own account, which
  the documents cannot contain.

## Output format

Write `eval/golden.yaml`:

```yaml
- id: q001
  question: "whats a drop in cost"
  category: exact_fact
  expect: answerable          # answerable | abstain
  sources:                    # file#heading-slug; empty list for out_of_scope
    - pricing.md#drop-in-and-class-packs
  reference_answer: "A single drop-in class is $28."
  must_include:               # the facts an answer MUST contain to be correct
    - "$28"
  evidence:                   # ONE verbatim quote per source, in the same order
    - "A single drop-in class at Northline Studio is $28, bookable without a membership."

- id: q002
  question: "do u have a steam room"
  category: out_of_scope
  expect: abstain
  sources: []
  reference_answer: "Not covered by the studio documents."
  must_include: []
  evidence: []
```

Field rules:

- `must_include` values must be **literal substrings of `content/`**, matched
  case-insensitively against the answer. Use the bare value (`"$28"`, `"6:30 AM"`,
  `"non-transferable"`) — never a paraphrase and never a full sentence, or the match will
  fail against a correct answer. Split composites: write `"Monday"` and `"6:00 AM"` as two
  entries, not `"Monday 6:00 AM"`.
- `sources` uses `file#slug`, where slug is the `##` heading lowercased with spaces as
  hyphens and punctuation dropped (repeated hyphens collapsed). `schedule.md` has no `##`
  headings — cite it as `schedule.md#weekly-class-schedule`. It lists every file+heading needed. For `multi_hop` this must contain **two or
  more entries in different files** — if it doesn't, it isn't multi-hop, so recategorize it.
- `evidence` is a **list** of verbatim quotes, one per entry in `sources`, same order. Each
  must appear character-for-character in `content/`. Never concatenate quotes into one
  string. If you cannot quote it, the question is not answerable and belongs in
  `out_of_scope`.
- For `expect: abstain`, `sources` and `must_include` are empty and `evidence` is `""`.

## When you're done

Report: the count per category, any factual contradictions you noticed in `content/`, and
the three questions you think are most likely to defeat the system, with your reasoning.

Validate your output with `python3 eval/audit_golden.py` before reporting. It must print
zero errors and no untested sections.
