import os
import re
import json
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

from clauses import split_clauses
import rag

MODEL = "gemma-4-26b-a4b-it"

SCHEMA_KEYS = {
    "clause_id", "original_text", "explanation", "jargon",
    "risk_flag", "risk_reason", "law_citation", "confidence",
}

PROMPT_TEMPLATE = """तिमी एक कानुनी कागजात व्याख्याता हौ। तलको खण्ड (clause) लाई सामान्य नेपाली भाषामा
व्याख्या गर। तिमीलाई सान्दर्भिक नेपाल कानुनको वास्तविक पाठ पनि दिइएको छ — यसैलाई आधार बनाएर
व्याख्या गर, आफैं कानुन नबनाउनू। कुनै कुरा "गैरकानुनी छ" भनी घोषणा नगर्नू — त्यो मान्छेले
निर्णय गर्ने कुरा हो, तिमीले केवल व्याख्या गर्ने र कानुन देखाउने मात्र हो।

खण्ड नं: {clause_id}
खण्डको मूल पाठ: {clause_text}

तलका सान्दर्भिक कानुनी दफाहरू मध्ये यो खण्डसँग सबैभन्दा प्रत्यक्ष रूपमा मिल्ने दफा एउटा मात्र छान:
{law_candidates}

तलको EXACT JSON structure मा मात्र जवाफ देऊ, अरू कुनै पाठ नथप्नू:
{{
  "explanation": "<khanda ko saral nepali व्याख्या>",
  "jargon": [{{"term": "<kathin sabda>", "definition": "<saral artha>"}}],
  "risk_flag": <true वा false — यदि खण्डमा केही अस्पष्ट/खुला छोडिएको वा भाडावाल/कामदारको लागि जोखिमपूर्ण छ भने true>,
  "risk_reason": "<risk_flag true भए किन जोखिमपूर्ण छ, नत्र खाली स्ट्रिङ>",
  "citation_section": "<mathi ka candidates माझबाट छानिएको दफा नम्बर, EXACT रूपमा>",
  "confidence": {{"explanation": "high|medium|low", "citation": "high|medium|low"}}
}}
"""


def _load_api_keys():
    keys = []
    with open(".env", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                keys.append(line.split("=", 1)[1])
            elif line.startswith("GEMINI_API_KEYS="):
                keys.extend(k for k in line.split("=", 1)[1].split(",") if k)
    if not keys:
        raise RuntimeError("no GEMINI_API_KEY(S) found in .env")
    return keys


_CLIENTS = [genai.Client(api_key=k) for k in _load_api_keys()]
_client_cycle = itertools.cycle(_CLIENTS)


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def process_clause(clause, retries=3, client=None):
    client = client or next(_client_cycle)
    candidates = rag.retrieve(clause["text"], top_k=3)
    law_candidates = "\n\n".join(
        f"[दफा {c['section']}] ({c['act']}): {c['text'][:500]}"
        for c in candidates
    )
    prompt = PROMPT_TEMPLATE.format(
        clause_id=clause["clause_id"],
        clause_text=clause["text"],
        law_candidates=law_candidates,
    )
    last_err = None
    for attempt in range(retries):
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        try:
            parsed = _extract_json(resp.text)
            chosen_section = str(parsed.get("citation_section", "")).strip()
            law = next(
                (c for c in candidates if c["section"] == chosen_section),
                candidates[0],
            )
            result = {
                "clause_id": clause["clause_id"],
                "original_text": clause["text"],
                "explanation": parsed["explanation"],
                "jargon": parsed.get("jargon", []),
                "risk_flag": bool(parsed["risk_flag"]),
                "risk_reason": parsed.get("risk_reason", ""),
                "law_citation": {
                    "act": law["act"],
                    "section": law["section"],
                    "text": law["text"],
                },
                "confidence": parsed["confidence"],
            }
            missing = SCHEMA_KEYS - result.keys()
            if missing:
                raise ValueError(f"missing keys: {missing}")
            return result
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_err = e
            print(f"[pipeline] clause {clause['clause_id']} parse retry "
                  f"{attempt + 1}/{retries}: {e}")
    raise RuntimeError(
        f"clause {clause['clause_id']} failed to produce valid JSON "
        f"after {retries} attempts: {last_err}"
    )


def process_text_stream(raw):
    """Same work as process_text, but yields (index, clause_id, result_or_None,
    error_or_None) as each clause finishes, instead of waiting for the whole
    batch — lets the frontend show clauses as soon as they're ready."""
    clause_list = split_clauses(raw)
    yield ("clause_count", len(clause_list), None, None)
    with ThreadPoolExecutor(max_workers=len(_CLIENTS)) as pool:
        futures = {
            pool.submit(process_clause, c, 3, next(_client_cycle)): (i, c["clause_id"])
            for i, c in enumerate(clause_list)
        }
        for fut in as_completed(futures):
            i, clause_id = futures[fut]
            try:
                yield ("clause", i, fut.result(), None)
            except Exception as e:
                yield ("clause", i, None, f"{clause_id}: {e}")


def process_text(raw):
    results = [None] * 0
    for kind, i, result, error in process_text_stream(raw):
        if kind == "clause_count":
            results = [None] * i
        elif error:
            raise RuntimeError(error)
        else:
            results[i] = result
    return results


def process_document(raw_text_path):
    with open(raw_text_path, encoding="utf-8") as f:
        raw = f.read()
    return process_text(raw)


if __name__ == "__main__":
    t0 = time.time()
    results = process_document("doc1_raw.txt")
    elapsed = time.time() - t0
    for r in results:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"\n[pipeline] processed {len(results)} clauses in {elapsed:.2f}s "
          f"({elapsed / len(results):.2f}s/clause)")
    with open("doc1_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
