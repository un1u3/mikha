import json
import sys

_CONF_RANK = {"high": 2, "medium": 1, "low": 0}


def _clause_num(clause_id):
    # sort key: convert Devanagari numeral clause_id to an int
    devanagari = "०१२३४५६७८९"
    return int("".join(str(devanagari.index(ch)) for ch in clause_id))


def top5(results):
    flagged = [c for c in results if c["risk_flag"]]
    scored = sorted(
        flagged,
        key=lambda c: (
            -(_CONF_RANK[c["confidence"]["explanation"]]
              + _CONF_RANK[c["confidence"]["citation"]]),
            _clause_num(c["clause_id"]),
        ),
    )
    return [
        {
            "clause_id": c["clause_id"],
            "risk_reason": c["risk_reason"],
            "law_citation": c["law_citation"]["section"],
        }
        for c in scored[:5]
    ]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "doc1_pipeline.json"
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    ranked = top5(results)
    print(f"[summary] {len(ranked)} of {sum(c['risk_flag'] for c in results)} "
          f"flagged clauses (from {len(results)} total) — top {len(ranked)}:\n")
    for i, r in enumerate(ranked, 1):
        print(f"{i}. खण्ड {r['clause_id']} (दफा {r['law_citation']}): {r['risk_reason']}\n")
