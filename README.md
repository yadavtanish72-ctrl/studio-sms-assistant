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

Everything runs on your own accounts: your Pinecone index holds the documents, and your
OpenRouter key pays for the AI. It takes about ten minutes.

### 1. Get two keys

- **Pinecone** (free): sign up at [pinecone.io](https://www.pinecone.io), open **API Keys** in
  the console, and create a key. The free Starter plan is enough.
- **OpenRouter** (pay as you go): sign up at [openrouter.ai](https://openrouter.ai), add some
  credit, and create a key at [openrouter.ai/keys](https://openrouter.ai/keys). An answer costs
  about $0.004, so $5 covers well over a thousand questions.

### 2. Install

You need **Python 3.11 or newer**. Check with `python3 --version`.

> Python 3.9 or 3.10 will install an old version of the Pinecone library that is missing
> the feature this project needs, **without any error**.

```bash
git clone <this repository's URL>
cd <repository folder>

python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
```

### 3. Fill in `.env`

Open `.env`. These two lines are empty for you to fill in: paste each key straight after its
`=`, without quotes.

```
PINECONE_API_KEY=
OPENROUTER_API_KEY=
```

Leave everything else as it is. The other settings are the ones the evaluation measured, and
the Twilio lines stay empty unless you set up texting (below). `.env` is gitignored, so your keys
never end up in a commit.

### 4. Build your index

```bash
./rag ingest --init
```

This creates an index called `studio-kb` in your Pinecone account and loads the 26 pieces into
it. The first run takes about a minute, and you should see
`upserted 26 documents; index now holds 26`.

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

Eight checks, all free. Only `test_fuse.py` needs the internet (it queries your index). Run
`source .venv/bin/activate` first so `python3` is the project's Python.

```bash
python3 check_corpus.py            # the 8 documents don't contradict each other
python3 eval/audit_golden.py       # the test questions are valid against the documents
python3 tests/test_config.py       # default settings match the measured ones
python3 tests/test_chunk.py        # documents split into the expected 26 pieces
python3 tests/test_fuse.py         # search merging, plus a live regression case
python3 tests/test_sms.py          # webhook security, deduplication, rate limiting
python3 tests/test_history.py      # conversation storage and name detection
python3 tests/test_ui.py           # web chat: visitors' keys pay, nothing saved (skips without Gradio)
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

## Web chat (optional)

A web page for chatting with the assistant in a browser. Next to each reply it shows how the
answer was found, and you can choose which AI model writes the answers. It searches your
Pinecone index, just like `./rag ask`.

### Start it

1. Install Gradio. It's kept out of the main requirements because it adds about 30 packages.
   ```bash
   .venv/bin/pip install -r requirements-ui.txt
   ```
2. In `.env`, change `ENABLE_UI=false` to `ENABLE_UI=true`.
3. Run `./rag serve`. Among the lines it prints is the page's address:
   ```
   web chat   ->  http://127.0.0.1:7860
   ```
   Gradio also prints a tip about `share=True`. Ignore it: sharing is switched off on purpose.
4. Open http://127.0.0.1:7860 in your browser. Press Ctrl+C in the terminal to stop it.

`./rag serve` also starts the SMS webhook, but only once the three Twilio settings are filled in.

### Use it

- **Your OpenRouter API key.** Paste your key here. The page only uses a key pasted into it,
  never the one in `.env`, and never saves it, so after reloading the page you paste it again.
- **Model.** The model that writes the answers. It starts as `google/gemini-3.8-flash`, the one
  the evaluation measured. You can type any model ID from
  [openrouter.ai/models](https://openrouter.ai/models) instead, such as
  `anthropic/claude-haiku-4.5` or `openai/gpt-5-mini`. Other models work, but they haven't been
  evaluated, and you pay that model's price. Clear the box to go back to the default.
- **Your text.** Type a question and press Enter or **Send**. The examples underneath fill it
  in for you.
- **What happened.** The panel beside the chat (below it on a phone) shows how each answer was
  found: the model, what it searched for (a follow-up is rewritten first), how close the best
  match was, whether it answered or refused, the sources, and how many texts the reply would
  take as an SMS.
- **New conversation.** Clears the chat. The conversation only lives in the page, so reloading
  clears it too.

Only the answer-writing model changes. Search always uses the embedding model the index was
built with, and follow-ups are rewritten by `REWRITE_MODEL` from `.env`.

When something goes wrong, a message pops up saying what: OpenRouter didn't accept the key, the
key has no credit, OpenRouter doesn't recognise the model ID, or the model stopped before
finishing its answer.

### Running it for others

**It's meant to run on your own machine.** To let someone else try it, send them this README so
they can run their own copy with their own keys. If you put it online instead, every visitor's
search runs on your Pinecone index, though each visitor's own key still pays for the AI. If you
do:

- Use a host that gives you HTTPS, since visitors are sending a key.
- Don't give the host your OpenRouter key. The web chat never needs it, so a bug can't bill you.
- Start it with `./rag serve --host 0.0.0.0` so it accepts outside connections.

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
  ui.py               the optional web chat
eval/
  golden.yaml         55 single test questions
  multiturn.yaml      14 test conversations
  run.py              the evaluation harness
  audit_golden.py     validates the test questions
tests/                the regression checks
check_corpus.py       checks the documents agree with each other
requirements-ui.txt   the extra packages for the web chat
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
