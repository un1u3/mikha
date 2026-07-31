import os
import re
import json
import time
import asyncio

import numpy as np
from gtts import gTTS
from google.genai import types

try:
    import edge_tts
except ImportError:
    edge_tts = None

import rag
from pipeline import _CLIENTS, _client_cycle

EDGE_VOICE = "ne-NP-HemkalaNeural"

ANSWER_PROMPT = """तिमी एक कानुनी कागजात सहायक हौ। तलको प्रयोगकर्ताको प्रश्नको जवाफ तलका
पहिले नै विश्लेषण गरिएका खण्डहरू र कानुनी सन्दर्भ प्रयोग गरी छोटो, प्रत्यक्ष नेपालीमा देऊ।
आफैं नयाँ कानुन नबनाउनू, दिइएको जानकारीमा मात्र आधारित रहनू।

सान्दर्भिक खण्डहरू:
{clauses_context}

प्रश्न: {question}

जवाफ (छोटो, प्रत्यक्ष नेपालीमा):"""


def synthesize(text, out_path):
    """Generate speech for `text` into `out_path` (mp3). Tries edge-tts
    first; falls back to gTTS only if edge-tts fails. Fails loudly if both
    fail — never silently produces no audio."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("cannot synthesize empty text")

    try:
        if edge_tts is None:
            raise RuntimeError("edge-tts is not installed")
        asyncio.run(_edge_synthesize(text, out_path))
        return {"engine": "edge-tts", "voice": EDGE_VOICE, "path": out_path}
    except Exception as e:
        print(f"[voice] edge-tts unavailable ({e}), falling back to gTTS")
        # A failed Edge request can leave a partial file behind.  Remove it
        # before asking gTTS to write the replacement audio.
        if os.path.exists(out_path):
            os.remove(out_path)
        gTTS(text=text, lang="ne").save(out_path)
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            raise RuntimeError("gTTS produced empty/near-empty output")
        return {"engine": "gTTS", "path": out_path}


async def _edge_synthesize(text, out_path):
    communicate = edge_tts.Communicate(text, EDGE_VOICE)
    await communicate.save(out_path)
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError("edge-tts produced empty/near-empty output")


_clause_vector_cache = {}


def _find_relevant_clauses(question, processed_clauses, top_k=2):
    import re
    mentioned_ids = []
    word_map = {
        "one": "१", "first": "१", "पहिलो": "१", "१": "१", "1": "१",
        "two": "२", "second": "२", "दोस्रो": "२", "२": "२", "2": "२",
        "three": "३", "third": "३", "तेस्रो": "३", "३": "३", "3": "३",
        "four": "४", "fourth": "४", "चौथो": "४", "४": "४", "4": "४",
        "five": "५", "fifth": "५", "पाँचौ": "५", "५": "५", "5": "५",
    }
    for word, dev_digit in word_map.items():
        if word in question.lower():
            if dev_digit not in mentioned_ids:
                mentioned_ids.append(dev_digit)
    digits = re.findall(r'[0-9०-९]+', question)
    for d in digits:
        dev_d = "".join("०१२३४५६७८९"["0123456789".index(c)] if c in "0123456789" else c for c in d)
        if dev_d not in mentioned_ids:
            mentioned_ids.append(dev_d)

    matched_clauses = [c for c in processed_clauses if c["clause_id"] in mentioned_ids]

    cache_key = id(processed_clauses)
    if cache_key not in _clause_vector_cache:
        texts = [c["original_text"] + " " + c["explanation"] for c in processed_clauses]
        _clause_vector_cache[cache_key] = np.array(
            [rag._embed_one(t) for t in texts], dtype=np.float32
        )
    vectors = _clause_vector_cache[cache_key]
    q = np.array(rag._embed_one(question), dtype=np.float32)
    q = q / np.linalg.norm(q)
    norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = norms @ q
    top_idx = np.argsort(-sims)

    result = list(matched_clauses)
    for idx in top_idx:
        if len(result) >= top_k:
            break
        c = processed_clauses[idx]
        if c not in result:
            result.append(c)
    return result[:top_k]


def answer_question(question, processed_clauses):
    relevant = _find_relevant_clauses(question, processed_clauses)
    clauses_context = "\n\n".join(
        f"खण्ड {c['clause_id']}: {c['original_text']}\n"
        f"व्याख्या: {c['explanation']}\n"
        f"कानुन: {c['law_citation']['act']} दफा {c['law_citation']['section']}: "
        f"{c['law_citation']['text'][:400]}"
        for c in relevant
    )
    prompt = ANSWER_PROMPT.format(clauses_context=clauses_context, question=question)
    client = next(_client_cycle)
    resp = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return {
        "question": question,
        "answer": resp.text.strip(),
        "source_clauses": [c["clause_id"] for c in relevant],
    }


if __name__ == "__main__":
    with open("doc1_pipeline.json", encoding="utf-8") as f:
        processed = json.load(f)

    t0 = time.time()
    out = synthesize(processed[3]["explanation"], "voice_test_clause4.mp3")
    print(f"[voice] Phase 6: synthesized clause ४ explanation via {out['engine']} "
          f"in {time.time() - t0:.2f}s -> {out['path']} "
          f"({os.path.getsize(out['path'])} bytes)")

    questions = [
        "किराया कति हो?",
        "जमानत रकम फिर्ता हुन्छ कि हुँदैन?",
        "घर छोड्नु परे कति दिन अगाडि सूचना दिनुपर्छ?",
    ]
    for q in questions:
        t0 = time.time()
        result = answer_question(q, processed)
        print(f"\n[voice] Phase 7 Q: {q}")
        print(f"  A ({time.time() - t0:.2f}s, from खण्ड {result['source_clauses']}): "
              f"{result['answer']}")
