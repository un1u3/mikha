import re

DEVANAGARI_DIGITS = "०१२३४५६७८९"


def split_clauses(text):
    """Split a document's raw text into clauses using its own Devanagari
    numeral markers (e.g. १., २., ३. ...) at the start of a line."""
    pattern = re.compile(rf"^([{DEVANAGARI_DIGITS}]+)\.\s+", re.MULTILINE)
    matches = list(pattern.finditer(text))
    clauses = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause_text = text[start:end].strip()
        if i == len(matches) - 1:
            boilerplate = re.search(r"\n\s*(इति\s+सम्+वत्|_{5,})", clause_text)
            if boilerplate:
                clause_text = clause_text[:boilerplate.start()].strip()
        clauses.append({
            "clause_id": m.group(1),
            "text": clause_text,
        })
    return clauses


if __name__ == "__main__":
    for fname in ["doc1_raw.txt", "doc2_raw.txt"]:
        print(f"\n=== {fname} ===")
        with open(fname, encoding="utf-8") as f:
            raw = f.read()
        clauses = split_clauses(raw)
        for c in clauses:
            print(f"[{c['clause_id']}] {c['text']}\n")
        print(f"--> {len(clauses)} clauses found in {fname}")
