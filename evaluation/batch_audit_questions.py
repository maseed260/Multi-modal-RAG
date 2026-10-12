import sys
import json
import re
from pathlib import Path

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent.parent
CHUNKS_PATH = BASE / "chunking" / "output" / "unified_chunks.json"
GT_PATH = BASE / "evaluation" / "ground_truth_dataset.json"

with open(CHUNKS_PATH, encoding="utf-8") as f:
    ALL_CHUNKS = json.load(f)

CHUNK_MAP = {c["chunk_id"]: c for c in ALL_CHUNKS}

with open(GT_PATH, encoding="utf-8") as f:
    GT_DATA = json.load(f)

Q_MAP = {q["id"]: q for q in GT_DATA["questions"]}


def audit_question(qid):
    if qid not in Q_MAP:
        return None
    q = Q_MAP[qid]
    criteria = q.get("evaluation_criteria", [])
    current_ids = [c["chunk_id"] for c in q.get("ground_truth_chunk_ids", [])]

    # Find primary metric numbers and keywords
    numbers = re.findall(r"[\$]?\d+(?:,\d+)*(?:\.\d+)?%?", q["ground_truth_answer"])
    primary_num = None
    for n in sorted(numbers, key=len, reverse=True):
        clean_n = n.replace("$", "").replace("%", "")
        if len(clean_n) >= 3 or "." in clean_n:
            primary_num = clean_n
            break

    # 1. Audit current chunks
    current_valid = []
    current_spurious = []
    for cid in current_ids:
        chunk = CHUNK_MAP.get(cid)
        if not chunk:
            current_spurious.append({"chunk_id": cid, "reason": "not_found"})
            continue
        text = chunk.get("text", "")
        p = chunk["provenance"]["page_start"]

        is_relevant = False
        sample_line = ""
        for line in text.split("\n"):
            line_str = line.strip()
            if line_str.startswith("- (") or line_str.startswith("*("):
                continue
            if primary_num and primary_num in line_str:
                is_relevant = True
                sample_line = line_str
                break
            matched_crit = [crit for crit in criteria if crit.lower() in line_str.lower()]
            if len(matched_crit) >= 2:
                is_relevant = True
                sample_line = line_str
                break

        if is_relevant:
            current_valid.append({"chunk_id": cid, "page": p, "sample_line": sample_line})
        else:
            current_spurious.append({"chunk_id": cid, "page": p})

    # 2. Search all 1,452 chunks across all 364 pages
    # Strict matching: MUST match the primary metric AND at least one thematic keyword
    authoritative_chunks = []
    for c in ALL_CHUNKS:
        text = c.get("text", "")
        p = c["provenance"]["page_start"]
        cid = c["chunk_id"]

        matched_crit = [crit for crit in criteria if crit.lower() in text.lower()]
        has_num = primary_num and (primary_num in text)

        # Must have metric number AND at least one criteria word in the chunk text
        if (has_num and len(matched_crit) >= 1) or len(matched_crit) >= 3:
            # Find the best representative line
            best_line = ""
            for line in text.split("\n"):
                line_str = line.strip()
                if line_str.startswith("- (") or line_str.startswith("*("):
                    continue
                if primary_num and primary_num in line_str:
                    best_line = line_str
                    break
                elif any(crit.lower() in line_str.lower() for crit in criteria):
                    best_line = line_str

            if best_line:
                authoritative_chunks.append({
                    "chunk_id": cid,
                    "modality": c["modality"],
                    "page_start": p,
                    "page_end": c["provenance"]["page_end"],
                    "toc_section": c.get("toc", {}).get("toc_section", ""),
                    "citation": c["provenance"].get("citation", ""),
                    "sample_line": best_line,
                    "match_score": 25 if has_num else 20
                })

    # Deduplicate
    seen = set()
    deduped = []
    for ac in authoritative_chunks:
        if ac["chunk_id"] not in seen:
            seen.add(ac["chunk_id"])
            deduped.append(ac)

    deduped.sort(key=lambda x: (x["page_start"], x["chunk_id"]))
    pages = sorted(list({c["page_start"] for c in deduped}))

    return {
        "id": qid,
        "question": q["question"],
        "answer": q["ground_truth_answer"],
        "primary_num": primary_num,
        "current_total": len(current_ids),
        "current_valid": current_valid,
        "current_spurious": current_spurious,
        "authoritative_chunks": deduped,
        "recommended_pages": pages
    }


def main():
    qids = sys.argv[1:] if len(sys.argv) > 1 else [f"GT-E-{i:03d}" for i in range(7, 13)]
    results = []
    for qid in qids:
        res = audit_question(qid)
        if not res:
            continue
        results.append(res)
        print("\n" + "=" * 90)
        print(f"[{res['id']}] Q: {res['question']}")
        print(f"Answer: {res['answer']}")
        print(f"Primary Metric: '{res['primary_num']}'")
        print(f"Current Chunks: {res['current_total']} total -> {len(res['current_valid'])} VALID, {len(res['current_spurious'])} SPURIOUS")
        if res['current_spurious']:
            print(f"  Spurious Chunks to Remove: {[s['chunk_id'] for s in res['current_spurious']]}")
        print(f"Authoritative Chunks Across Whole Report ({len(res['authoritative_chunks'])} found on pages {res['recommended_pages']}):")
        for ac in res['authoritative_chunks']:
            sample = ac['sample_line'][:60].encode("ascii", "replace").decode("ascii")
            sec = ac['toc_section'][:35].encode("ascii", "replace").decode("ascii")
            print(f"  + {ac['chunk_id']:30s} [p.{ac['page_start']:3d}] [{ac['modality']:6s}] {sec:35s} | {sample}")

    out_file = BASE / "evaluation" / "batch_audit_preview.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
