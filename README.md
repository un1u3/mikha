# Mikha

**Mikha** is a local, plain-language legal document reviewer for Nepali
documents. Upload a document photo or paste its text, and Mikha:

- preserves and displays the original clause text;
- explains each numbered clause in plain Nepali;
- identifies potentially vague or risky clauses;
- grounds explanations in relevant sections of Nepal's Labour Act, 2017 and
  National Civil Code, 2017;
- streams completed clauses to the browser as they are analyzed;
- reads an explanation aloud on demand; and
- answers follow-up questions using the analyzed document.

Mikha is an interpretation and research aid, not a lawyer or a substitute for
professional legal advice. Its output can be incomplete or incorrect,
especially when the source document is hard to read or a clause falls outside
the bundled legal corpus.

## How it works

```text
document image/text
        |
        v
Gemma multimodal transcription (images only)
        |
        v
Devanagari-numbered clause splitting
        |
        v
Gemini embeddings -> relevant law sections
        |
        v
Gemma structured explanation per clause
        |
        +--> streamed clause cards in the browser
        +--> ranked top-five risk summary
        +--> optional Nepali speech and document Q&A
```

### Analysis pipeline

1. **Read:** an uploaded image is transcribed by the configured multimodal
   Gemma model. Pasted text skips this step.
2. **Split:** `clauses.py` recognizes clause markers such as `१.`, `२.`, and
   `३.` at the start of a line. It does not infer arbitrary paragraph
   boundaries.
3. **Retrieve:** `rag.py` embeds each clause and compares it with a corpus
   extracted from the bundled Acts. Embeddings are cached in
   `rag_cache.json`.
4. **Explain:** `pipeline.py` asks Gemma for a fixed JSON-shaped result
   containing an explanation, jargon definitions, a risk flag, a reason,
   citation, and confidence values. Clauses are processed concurrently.
5. **Summarize:** `summary.py` ranks flagged clauses by the confidence of
   their explanation and citation and returns at most five.
6. **Speak and answer:** `voice.py` generates speech only when requested and
   answers questions from the processed clauses and their law context.

The legal retrieval corpus currently contains:

- the full Labour Act, 2017; and
- the house-rent and contract/obligations sections of the National Civil
  Code, 2017.

The bundled PDFs are the source documents. If extracted text files named
`labour_raw.txt` or `civilcode_raw.txt` are present, `rag.py` uses them
instead.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Flask server, upload stream, voice endpoint, and Q&A endpoint |
| `static/index.html` | Self-contained browser UI |
| `clauses.py` | Devanagari-numbered clause extraction |
| `rag.py` | Law-source loading, embedding, caching, and similarity retrieval |
| `pipeline.py` | Per-clause Gemini/Gemma analysis and concurrent streaming |
| `summary.py` | Deterministic top-five risk ranking |
| `voice.py` | Edge TTS/gTTS speech and document question answering |
| `The-Labour-Act-2017.pdf` | Bundled Labour Act source |
| `National-Civil-Code-2017.pdf` | Bundled Civil Code source |
| `rag_cache.json` | Checked-in embedding cache used to avoid rebuilding the corpus |
| `requirements.txt` | Pinned Python dependencies |
| `.env` | Local API-key file; create it yourself and never commit it |

## Requirements

- Python 3.10+ recommended.
- A Gemini API key with access to the configured models:
  `gemma-4-26b-a4b-it` and `gemini-embedding-001`.
- Poppler's `pdftotext` command, unless you provide the optional extracted
  files `labour_raw.txt` and `civilcode_raw.txt`.
- Internet access for Gemini, Edge TTS, and the gTTS fallback.

`rag_cache.json` is already included, so a normal startup does not need to
embed the law corpus again. If the cache is removed or needs rebuilding,
`pdftotext` is required to extract the bundled PDFs.

## Setup

From the repository root:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, run the commands from an activated
environment or use `.venv\Scripts\python.exe` directly.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your-primary-key
# Optional: comma-separated additional keys for concurrent clause work
GEMINI_API_KEYS=your-second-key,your-third-key
```

The application reads this file directly; it does not load environment
variables from the shell as a replacement. Keep `.env` private. It is ignored
by Git.

## Run the application

```bash
python app.py
```

Open <http://127.0.0.1:5050> in a browser. Choose either an image or pasted
text, then select **Analyze document**. The page shows clauses as they finish
instead of waiting for the slowest clause.

This uses Flask's built-in development server. It is not configured for
production deployment, authentication, multi-user isolation, or persistent
document storage. The currently loaded document is kept in process memory and
is shared by all clients of that server process.

## HTTP API

### `POST /api/upload`

Accepts `multipart/form-data` with either:

- `image`: an image file; Gemini transcribes it; or
- `text`: the document's raw text.

The response is a Server-Sent Events stream. Events include:

| Event | Payload |
| --- | --- |
| `clause_count` | Total number of detected clauses |
| `clause` | `{ "index": number, "result": object }` |
| `clause_error` | `{ "index": number, "error": string }` |
| `summary` | `{ "summary": [...] }` |
| `done` | Elapsed time and completed clause count |
| `error` | A request-level error |

Each successful clause result contains `clause_id`, `original_text`,
`explanation`, `jargon`, `risk_flag`, `risk_reason`, `law_citation`, and
`confidence`.

### `GET /api/voice/<clause_id>`

Generates an MP3 for the selected clause explanation. Edge TTS is attempted
first with the Nepali voice `ne-NP-HemkalaNeural`; gTTS is used as a fallback.
Audio is generated on demand and is not retained by the server.

### `POST /api/ask`

Accepts JSON:

```json
{ "question": "किराया कति हो?" }
```

Returns the answer and the clause IDs used as sources:

```json
{
  "question": "किराया कति हो?",
  "answer": "...",
  "source_clauses": ["१", "३"]
}
```

The upload endpoint must be used first because Q&A operates on the currently
loaded in-memory document.

## Development notes

- Add more API keys with `GEMINI_API_KEYS` to increase the number of
  concurrent clause workers. This can also increase API usage and cost.
- Delete `rag_cache.json` to force a corpus rebuild. The rebuild embeds every
  legal section and may be slow and costly.
- The analyzer expects numbered Nepali clauses at the beginning of lines.
  Documents without markers such as `१.` will produce no clauses.
- The model is instructed to explain rather than declare a clause legal or
  illegal. Always verify important conclusions against the original Act and
  a qualified legal professional.
- There are no automated tests or production deployment configuration in this
  repository. The command-line module entry points are development utilities
  and some expect local artifacts that are not included in the repository.

## License

See [LICENSE](LICENSE).
