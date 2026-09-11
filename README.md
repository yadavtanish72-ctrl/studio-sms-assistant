# Studio SMS Assistant

A retrieval-augmented question-answering system that answers customers' text messages for a
fitness studio. A customer texts *"how much is a drop in class"*, and it replies from the
studio's own documents — or says it doesn't know, rather than guessing.

It's built to be measured. Every design choice was tested against a fixed set of questions,
and several obvious-looking choices were rejected because the numbers said they made things
worse. [PLAN.md](PLAN.md) records every one of those decisions and why.

> **The studio is fictional.** "Northline Studio", its address, trainers, prices and
> schedule are all invented for this project. Everything in `content/` is synthetic.

## How it works

There are two paths.

**Ingest** runs when the documents change. It splits the 8 markdown files in `content/` into
26 pieces, turns each piece into a list of numbers that captures its meaning (an
*embedding*), and stores them in Pinecone.

**Query** runs on every question:

1. **Search twice.** One search matches *meaning*, so "havent worked out in 2 years" finds
   the beginners section even though they share no words. The other matches *exact words*,
   which is what you need for `$28`, a trainer's name, or `Thursday`.
2. **Merge** the two result lists into one ranked list of the 8 best pieces.
3. **Decide whether to answer.** If nothing relevant was found, it refuses without calling
   the AI at all.
4. **Answer** from those 8 pieces and nothing else, in about 320 characters.

If the question is a follow-up ("and how long do i have to use it"), it first rewrites it
into a standalone question using the conversation so far, so the search has something to
look for.

## Results

Measured on 55 single questions and 14 multi-turn conversations, all written in the way
people actually text.

| | |
|---|---|
| Right source found in the top 8 | 97.8% |
| Answers correct | 100% |
| Answers that stick to the documents | 100% |
| Out-of-scope questions correctly refused | 100% |
| Follow-up questions answered correctly | 14 of 14 |
| Cost per answer | about $0.0036 |

It was also compared against the simplest possible alternative: pasting the whole knowledge
base into every prompt and skipping retrieval. Retrieval matched it on correctness, beat it
on sticking to the documents (100% vs 95.6%), and cost about 40% less.

**One honest caveat:** some settings were tuned against these same questions. The questions
were written before any tuning, which is the main protection — but treat these numbers as a
tuning result, not a guarantee about questions nobody has asked yet.

## Setup

You need **Python 3.11 or newer**, a [Pinecone](https://www.pinecone.io) account, and an
[OpenRouter](https://openrouter.ai/keys) API key. Both have free tiers that cover this.

> Python 3.9 or 3.10 will install an old version of the Pinecone library that is missing
> the feature this project needs, **without any error**. Check with `python3 --version`.

```bash
git clone <this repository's URL>
cd <repository folder>

python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
```

Open `.env` and paste in your two keys:

```
PINECONE_API_KEY=...
OPENROUTER_API_KEY=...
```

Then build the index. The first run creates it, which takes a minute:

```bash
./rag ingest --init
```

You should see `upserted 26 documents; index now holds 26`.

## Using it

### Ask a question

```bash
./rag ask "how much is a drop in class"
```

```
A single drop-in class at Northline Studio is $28, bookable without a membership.

[214 chars | dense_top1 0.551]
citations: pricing.md#drop-in-and-class-packs
```

The first part is what a customer would receive. `citations` shows which documents it used,
so you can check it didn't make anything up. Add `--debug` to see every piece it
considered.

Things worth trying:

| Ask | What should happen |
|---|---|
| `"do you have a sauna"` | Refuses — there's no sauna in the documents |
| `"whats on saturday"` | Lists all five Saturday classes, not just a few |
| `"its 24 hrs notice to cancel right"` | Corrects you — it's 12 hours |
| `"how many classes do i have left"` | Refuses — it has no access to anyone's account |

### Have a conversation

`--as` makes it behave like a real texter with that phone number, remembering the
conversation:

```bash
./rag ask "how much is a drop in class"       --as +15551234567
./rag ask "what about the 10 pack"            --as +15551234567
./rag ask "and how long do i have to use it"  --as +15551234567
```

The third question only works because of the conversation. Its output shows what it
actually searched for:

```
searched for: 'how long do i have to use the 10-class pack'
```

The number is just a label — use any string to keep separate conversations apart. Look at a
conversation with `./rag history +15551234567`, list everyone with `./rag customers`, and set
a name with `./rag name +15551234567 Sara`.

### Measure it

| Command | What it does | Cost |
|---|---|---|
| `./rag eval retrieval` | Checks whether the right source is found, for all 55 questions | free |
| `./rag eval sweep` | Tries 28 search-weighting settings | free |
| `./rag eval multiturn` | Grades the 14 conversations | ~$0.05 |
| `./rag eval answers` | Grades every answer with a second AI model | ~$0.20 |
| `./rag eval control` | The paste-everything comparison | ~$0.33 |

`retrieval` is the one to run often: it's free because question embeddings are cached.

### Run the tests

Seven checks, all free. Only `test_fuse.py` needs the internet (it queries your index).

```bash
python3 check_corpus.py          # the 8 documents don't contradict each other
python3 eval/audit_golden.py     # the test questions are valid against the documents
python3 tests/test_config.py     # default settings match the measured ones
python3 tests/test_chunk.py      # documents split into the expected 26 pieces
python3 tests/test_fuse.py       # search merging, plus a live regression case
python3 tests/test_sms.py        # webhook security, deduplication, rate limiting
python3 tests/test_history.py    # conversation storage and name detection
```

## Receiving real text messages (optional)

`./rag serve` runs a [Twilio](https://www.twilio.com) SMS webhook. You'll need a Twilio
account and phone number.

1. Add `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` and `TWILIO_FROM_NUMBER` to `.env`.
2. Run `./rag serve`.
3. In another terminal, expose it publicly — for example `ngrok http 5000`.
4. Set `TWILIO_WEBHOOK_URL` in `.env` to that public address plus `/sms`, and restart.
5. Paste the **exact same** URL into your Twilio number's "A message comes in" webhook.

The URL must match character for character. The webhook checks every request's signature to
confirm it really came from Twilio, and that check depends on the exact URL. If it doesn't
match, every message gets rejected.

Replies arrive about 10 seconds after a message, because answering takes longer than
Twilio's 15-second limit allows for an immediate reply. Each phone number is limited to 15
messages an hour.

Before texting real customers: a Twilio trial account can only message numbers you've
verified, and US numbers generally need A2P 10DLC registration. Check Twilio's current
requirements for your country.

## Project layout

```
content/              the knowledge base — 8 markdown files
src/
  config.py           every setting, read from .env
  chunk.py            splits documents into 26 pieces
  embed.py            turns text into embeddings via OpenRouter
  store.py            everything that talks to Pinecone
  fuse.py             merges the two searches
  answer.py           decides whether to answer, then answers
  history.py          stores conversations in SQLite
  sms.py              the Twilio webhook
  cli.py              the ./rag commands
eval/
  golden.yaml         55 single test questions
  multiturn.yaml      14 test conversations
  run.py              the evaluation harness
  audit_golden.py     validates the test questions
tests/                the regression checks
check_corpus.py       checks the documents agree with each other
PLAN.md               every design decision, with the numbers behind it
```

## Known limitations

- **Conversations are kept forever.** `history.db` has no retention period and no way to
  delete one person's messages on request. Fine for testing; not for real customers.
- **Rate limits are per process.** Run two copies of the server and the limit doubles.
- **The test set is at 100%.** That means it has stopped catching small regressions. It
  needs harder questions to stay useful.

## Configuration

All settings live in `.env`; see `.env.example` for every option with an explanation. The
defaults are the values the evaluation was run at, so you only need to set the two API keys.
