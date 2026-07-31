# Mikha

Plain-language legal document review for Nepali documents. Upload a photo
(or paste text) of a rental agreement, employment contract, or other formal
document — Mikha explains each clause in plain Nepali, cites the actual law
behind it, flags what's left dangerously vague, and reads any explanation
aloud on request. The original clause text is always shown untouched next
to its explanation, never replaced.

See `mikha-agent-loop-prompt.md` for the full design rationale and the
phase-by-phase build spec this project followed.
s
## How it works

1. **Read** — a photo of the document goes to Gemma (multimodal), which
   transcribes it directly — no separate OCR step.
2. **Split** — the raw text is split into clauses using the document's own
   numbering (१, २, ३...), not model-guessed boundaries (`clauses.py`).
3. **Ground** — each clause is matched against the actual text of relevant
   Nepali law (Labour Act 2017, National Civil Code 2017) via embedding
   search over a pre-embedded, cached corpus (`rag.py`).
4. **Explain** — one Gemma call per clause returns a locked JSON schema:
   plain-language explanation, jargon definitions, a risk flag + reason,
   the specific law citation used, and confidence levels (`pipeline.py`).
   Clauses are processed concurrently across a pool of API keys and
   streamed to the browser as each one finishes.
5. **Summarize** — the flagged clauses are ranked into a top-5 "what you
   should know" list, pure logic, no extra model call (`summary.py`).
6. **Speak** — any clause's explanation can be converted to speech
   on demand (not pre-generated) via edge-tts, with a gTTS fallback
   (`voice.py`).
7. **Answer** — free-form questions about the document are answered using
   the already-processed clauses plus law context, not a fresh document
   read (`voice.py`).

## Project layout

```
clauses.py    Phase 2 — split raw text into clauses by Devanagari numbering
rag.py        Phase 3 — embed + cache the law corpus, retrieve by meaning
pipeline.py   Phase 4 — per-clause Gemma call, streaming + parallel
summary.py    Phase 5 — top-5 risk summary (pure logic)
voice.py      Phase 6/7 — on-demand TTS + voice Q&A
app.py        Phase 9 — Flask backend, SSE streaming, serves the frontend
static/       Phase 9 — frontend (plain HTML/CSS/JS, no framework)

doc1_raw.txt, doc2_raw.txt     sample documents used to validate the pipeline
doc1_pipeline.json, doc2_pipeline.json   their saved Phase 4 output
labour_raw.txt, civilcode_raw.txt        source text for the law corpus
rag_cache.json                           cached corpus embeddings (built once)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install google-genai numpy flask edge-tts gTTS
```

Add your Gemini API key(s) to `.env` in the project root:

```
GEMINI_API_KEY=<your key>
# optional: additional keys, comma-separated, round-robined across
# clauses to parallelize processing and raise effective throughput
GEMINI_API_KEYS=<key2>,<key3>,<key4>
```

## Running

```bash
source venv/bin/activate
python3 app.py
```

Then open `http://127.0.0.1:5050`. Upload a document photo or paste text,
and clauses will appear as each one finishes processing (see "Streaming"
below) rather than all at once at the end.

The first run also builds and caches the law corpus embeddings
(`rag_cache.json`) — subsequent runs and requests load the cache instead of
re-embedding, which is required for the app to feel fast.

### Running individual phases

Each module can also be run standalone against the bundled sample
documents, useful for checking a phase in isolation:

```bash
python3 clauses.py     # print clause splits for doc1/doc2
python3 rag.py          # print corpus stats + sample retrievals
python3 pipeline.py     # run the full clause pipeline on doc1, save doc1_pipeline.json
python3 summary.py doc1_pipeline.json
python3 voice.py        # synthesize a sample clause + answer sample questions
```

## API

- `POST /api/upload` — send `image` (file) or `text` (form field). Streams
  progress over Server-Sent Events: a `clause_count` event, then one
  `clause` (or `clause_error`) event per clause as it finishes, then
  `summary` and `done`.
- `GET /api/voice/<clause_id>` — synthesizes and returns speech (mp3) for
  that clause's explanation in the currently loaded document, on demand.
- `POST /api/ask` — `{"question": "..."}`, answered from the currently
  loaded document's already-processed clauses + law context.

## Notes

- **Streaming**: clause processing runs concurrently across the API key
  pool and results are pushed to the browser as each clause completes
  (Server-Sent Events), so the page fills in progressively instead of
  blocking on the slowest clause.
- **Speed**: the law corpus is embedded once and cached — never
  re-embedded per request. Voice is generated lazily, only for the
  specific clause a user clicks, never pre-generated in bulk.
- **Model**: `gemma-4-26b-a4b-it`, multimodal, all generated explanation
  text in Nepali.
- This is a local development server (`flask run`'s built-in server) —
  not configured for production deployment.
