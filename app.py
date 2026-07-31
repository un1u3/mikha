"""Phase 9 — wires Phases 1-7 together, serves the frontend.

Upload -> read image (Gemma multimodal, Phase 1) -> process clauses
(Phases 2-4) -> top-5 summary (Phase 5). Voice (Phase 6) and Q&A
(Phase 7) are served on demand for the currently loaded document,
kept in memory (single-user demo, no auth/session complexity needed).
"""

import os
import io
import json
import time
import uuid

from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from google.genai import types

import pipeline
import summary
import voice

app = Flask(__name__, static_folder="static", static_url_path="")

READ_PROMPT = """तलको कागजातको फोटोमा लेखिएको सम्पूर्ण पाठ, शब्दशः, जस्ताको त्यस्तै निकाल।
कुनै व्याख्या वा अनुवाद नगर्नू, केवल कागजातमा भएको मूल पाठ मात्र देऊ। खण्ड नम्बरिङ
(१, २, ३...) जस्ताको त्यस्तै राख्नू।"""

# in-memory store for the currently loaded document
_DOCUMENT = {"clauses": None, "summary": None}


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


def _read_document(req):
    if "image" in req.files:
        image_bytes = req.files["image"].read()
        mime = req.files["image"].mimetype or "image/jpeg"
        client = next(pipeline._client_cycle)
        resp = client.models.generate_content(
            model=pipeline.MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                READ_PROMPT,
            ],
        )
        return resp.text.strip()
    if "text" in req.form:
        return req.form["text"]
    return None


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/upload")
def upload():
    """Streams progress over Server-Sent Events: clauses appear as soon as
    each finishes processing, instead of the client waiting for the whole
    batch. Falls back to a single JSON response if the client can't stream."""
    raw_text = _read_document(request)
    if raw_text is None:
        return jsonify({"error": "upload an 'image' file or 'text' field"}), 400

    def generate():
        t0 = time.time()
        results = []
        total = None
        # A clause is ready to speak as soon as it is streamed to the page.
        # Start a fresh document state so a voice request cannot read stale
        # clauses from the previous upload.
        _DOCUMENT["clauses"] = None
        _DOCUMENT["summary"] = None
        voice._clause_vector_cache.clear()
        try:
            for kind, i, result, error in pipeline.process_text_stream(raw_text):
                if kind == "clause_count":
                    total = i
                    results = [None] * total
                    if total == 0:
                        yield _sse("error", {"error": "no numbered clauses (१, २, ३...) found in document"})
                        return
                    yield _sse("clause_count", {"count": total})
                elif error:
                    yield _sse("clause_error", {"index": i, "error": error})
                else:
                    results[i] = result
                    # Publish only completed clauses.  This makes their
                    # on-demand voice endpoint available immediately.
                    _DOCUMENT["clauses"] = [r for r in results if r is not None]
                    yield _sse("clause", {"index": i, "result": result})
        except Exception as e:
            yield _sse("error", {"error": str(e)})
            return

        results = [r for r in results if r is not None]
        ranked = summary.top5(results)
        _DOCUMENT["clauses"] = results
        _DOCUMENT["summary"] = ranked
        yield _sse("summary", {"summary": ranked})
        yield _sse("done", {"elapsed_seconds": round(time.time() - t0, 2), "count": len(results)})

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.get("/api/voice/<clause_id>")
def get_voice(clause_id):
    if not _DOCUMENT["clauses"]:
        return jsonify({"error": "no document loaded"}), 400
    clause = next((c for c in _DOCUMENT["clauses"] if c["clause_id"] == clause_id), None)
    if not clause:
        return jsonify({"error": f"no clause {clause_id} in current document"}), 404

    # Use a temporary file only while a TTS provider is writing.  Returning
    # bytes keeps stale audio out of the cache and guarantees this response
    # corresponds to the clause that was requested.
    out_path = os.path.join("/tmp", f"mikha-{uuid.uuid4().hex}.mp3")
    try:
        voice.synthesize(clause["explanation"], out_path)
        with open(out_path, "rb") as audio_file:
            audio = audio_file.read()
    except Exception as exc:
        app.logger.exception("Could not synthesize clause %s", clause_id)
        return jsonify({"error": f"audio generation failed: {exc}"}), 502
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

    if len(audio) < 1000:
        return jsonify({"error": "audio generation returned an empty file"}), 502

    response = send_file(
        io.BytesIO(audio),
        mimetype="audio/mpeg",
        download_name=f"clause-{clause_id}.mp3",
        max_age=0,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/api/ask")
def ask():
    if not _DOCUMENT["clauses"]:
        return jsonify({"error": "no document loaded"}), 400
    question = (request.get_json() or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "missing 'question'"}), 400
    result = voice.answer_question(question, _DOCUMENT["clauses"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5050, debug=False, threaded=True)
