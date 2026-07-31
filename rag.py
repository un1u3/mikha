"""
Corpus scope (chosen by reading the source Acts' own chapter structure,
not guessed line ranges):
  - Labour Act, 2017: entire act (183 numbered sections) — short enough,
    and all plausibly relevant to an employment contract (Phase 8).
  - National Civil Code, 2017: only Part-4 Chapter-9 "Provisions Relating
    to House Rent" (sections 383-405) and Part-5 chapters 1-16
    "Provisions Relating to Contract and Other Obligations" (sections
    493-671, stopping before Chapter-17 Torts / Chapter-18 Defective
    Products, which are not contract law). Marriage/adoption/divorce/etc.
    chapters are excluded — irrelevant to legal-document comprehension
    and would only dilute retrieval.
"""

import os
import re
import json
import time
import subprocess
import threading
from pathlib import Path

import numpy as np
from google import genai

BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "rag_cache.json"
EMBED_MODEL = "gemini-embedding-001"


def _read_law_source(text_filename, pdf_filename):
    """Read an extracted law text file, or derive it from the bundled PDF.

    The raw text files are convenient development artifacts but are not
    required at runtime: the repository ships the authoritative PDFs.
    """
    text_path = BASE_DIR / text_filename
    if text_path.exists():
        return text_path.read_text(encoding="utf-8")

    pdf_path = BASE_DIR / pdf_filename
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Neither {text_path.name} nor bundled source PDF {pdf_path.name} exists"
        )
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{text_path.name} is absent and pdftotext is not installed. "
            "Install Poppler (pdftotext) or add the extracted text file."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not extract {pdf_path.name}: {exc.stderr.strip()}") from exc
    return completed.stdout


def _load_api_key():
    with open(".env", encoding="utf-8") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("GEMINI_API_KEY not found in .env")


_client = genai.Client(api_key=_load_api_key())


def _chunk_by_section(text, start_line, end_line, pattern, act_name):
    # pdftotext -layout inserts form-feed (\x0c) page breaks, which Python's
    # str.splitlines() treats as line boundaries but grep/wc -l do not — that
    # mismatch silently drops sections after any page break. Split on '\n'
    # only so line numbers match what grep -n reports.
    lines = text.split("\n")[start_line - 1:end_line - 1]
    body = "\n".join(lines)
    matches = list(pattern.finditer(body))
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw = body[start:end].strip()
        raw = re.sub(r"\n\s*\d+\s*\n", "\n", raw)  # drop page-number lines
        raw = re.sub(r"[ \t]+", " ", raw)
        title = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
        chunks.append({
            "act": act_name,
            "section": m.group(1),
            "title": title,
            "text": raw,
        })
    return chunks


def build_corpus():
    labour = _read_law_source("labour_raw.txt", "The-Labour-Act-2017.pdf")
    civil = _read_law_source("civilcode_raw.txt", "National-Civil-Code-2017.pdf")

    labour_pattern = re.compile(r"^(\d{1,3})\.\s+([^\n:]+):", re.MULTILINE)
    civil_pattern = re.compile(r"^(\d{2,3})\.\s+([^\n:]+):", re.MULTILINE)

    chunks = []
    chunks += _chunk_by_section(labour, 1, len(labour.splitlines()) + 1,
                                 labour_pattern, "Labour Act, 2017")
    chunks += _chunk_by_section(civil, 7106, 7535, civil_pattern,
                                 "National Civil Code, 2017 (House Rent)")
    chunks += _chunk_by_section(civil, 9109, 12973, civil_pattern,
                                 "National Civil Code, 2017 (Contracts/Obligations)")
    return chunks


def _embed_one(text, retries=5):
    delay = 2
    for attempt in range(retries):
        try:
            resp = _client.models.embed_content(model=EMBED_MODEL, contents=text)
            return resp.embeddings[0].values
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"[rag] embed retry {attempt + 1}/{retries} after error: {e}")
            time.sleep(delay)
            delay *= 2


def _embed_texts(texts, checkpoint_path=None):
    vectors = []
    for i, t in enumerate(texts):
        vectors.append(_embed_one(t))
        if checkpoint_path and (i + 1) % 20 == 0:
            print(f"[rag] embedded {i + 1}/{len(texts)}")
    return vectors


def load_or_build_corpus():
    if os.path.exists(CACHE_PATH):
        t0 = time.time()
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        chunks = data["chunks"]
        vectors = np.array(data["vectors"], dtype=np.float32)
        elapsed = time.time() - t0
        print(f"[rag] loaded {len(chunks)} cached chunks in {elapsed:.3f}s")
        return chunks, vectors

    t0 = time.time()
    chunks = build_corpus()

    partial_path = CACHE_PATH.with_name(CACHE_PATH.name + ".partial")
    done = 0
    vectors = []
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            partial = json.load(f)
        if partial["chunk_count"] == len(chunks):
            vectors = partial["vectors"]
            done = len(vectors)
            print(f"[rag] resuming from checkpoint: {done}/{len(chunks)} already embedded")

    for i in range(done, len(chunks)):
        vectors.append(_embed_one(chunks[i]["text"]))
        if (i + 1) % 20 == 0 or i == len(chunks) - 1:
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump({"chunk_count": len(chunks), "vectors": vectors}, f)
            print(f"[rag] embedded {i + 1}/{len(chunks)}")

    elapsed = time.time() - t0
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "vectors": vectors}, f, ensure_ascii=False)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"[rag] embedded {len(chunks)} chunks from scratch in {elapsed:.2f}s, cached to {CACHE_PATH}")
    return chunks, np.array(vectors, dtype=np.float32)


_CHUNKS, _VECTORS = None, None
_LOAD_LOCK = threading.Lock()


def _ensure_loaded():
    global _CHUNKS, _VECTORS
    if _CHUNKS is None:
        # Every uploaded clause is processed concurrently. Only one worker
        # must build/load the shared corpus; the others wait for that result.
        with _LOAD_LOCK:
            if _CHUNKS is None:
                _CHUNKS, _VECTORS = load_or_build_corpus()


def retrieve(clause_text, top_k=2):
    _ensure_loaded()
    q = np.array(_embed_texts([clause_text])[0], dtype=np.float32)
    q = q / np.linalg.norm(q)
    norms = _VECTORS / np.linalg.norm(_VECTORS, axis=1, keepdims=True)
    sims = norms @ q
    top_idx = np.argsort(-sims)[:top_k]
    return [{**_CHUNKS[i], "score": float(sims[i])} for i in top_idx]


if __name__ == "__main__":
    _ensure_loaded()
    per_act = {}
    for c in _CHUNKS:
        per_act[c["act"]] = per_act.get(c["act"], 0) + 1
    print("Corpus size:", per_act, "total:", len(_CHUNKS))

    test_clauses = {
        "२ (ढिलो भाडा भुक्तानी / पूर्व सूचना बिना घर खाली)":
            "भाडावाले प्रत्येक महिनाको ५ गते भित्र भाडा रकम बुझाउनु पर्नेछ। ढिलो भएमा मालिकले "
            "निमुखा भाडावालालाई कुनै पूर्व सूचना बिना नै घर खाली गराउन सक्नेछन्।",
        "३ (जमानत रकम, फिर्ता/नगर्ने अधिकार मालिकमा)":
            "सम्झौता हुँदा भाडावाले जमानतस्वरूप रु. ४५,०००/- अग्रिम बुझाउनु पर्नेछ, जुन सम्झौता "
            "अन्त्य हुँदा फिर्ता गरिने वा नगरिने सम्पूर्ण अधिकार मालिकमा नै रहनेछ।",
        "६ (बीचमै छोड्दा ३ महिना सूचना नदिए जमानत जफत)":
            "यदि भाडावाले घर बीचमै छोड्ने भएमा, तीन महिना अगावै लिखित सूचना नदिई छोडेमा जमानत "
            "रकम पूर्णतः जफत हुनेछ र यस बापत कुनै मुद्दा दायर गर्न पाइने छैन।",
    }
    for label, clause in test_clauses.items():
        t0 = time.time()
        results = retrieve(clause, top_k=2)
        elapsed = time.time() - t0
        print(f"\n=== clause {label} (retrieved in {elapsed:.2f}s) ===")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['act']} Sec {r['section']} {r['title']}")
            print(f"    {r['text'][:300]}")
