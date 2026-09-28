import json
import base64
import time
import urllib.request
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "chunking" / "output"
CACHE_PATH = OUTPUT_DIR / "detailed_image_summaries.json"
OLLAMA_URL = "http://localhost:11434"
MODEL = "glm-ocr:latest"

with open(CACHE_PATH, "r", encoding="utf-8") as f:
    summaries = json.load(f)

failed_keys = [k for k, v in summaries.items() if "error" in v or not v.get("detailed_summary") or v.get("detailed_summary", "").startswith("Visual element on page")]
print(f"Total entries: {len(summaries)}")
print(f"Entries to repair: {len(failed_keys)}: {failed_keys}")

def summarize_with_glm(image_path: Path, page_no: int, nearby_text: str = "") -> dict:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    prompt = f"""Transcribe and analyze this financial graphic from JPMorgan Chase & Co. 2025 Annual Report (Page {page_no}).
Nearby context: {nearby_text[:200] if nearby_text else 'N/A'}

Provide:
1. Chart or Diagram Title
2. Complete breakdown of series, axes, time horizon (e.g. 2005-2025), and all quantitative figures
3. Key financial takeaway for investors"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False
    }
    
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        raw_text = res.get("response", "").strip()
        
    # Extract title from first line if available
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    first_line = lines[0] if lines else f"Financial Chart (Page {page_no})"
    clean_title = first_line.replace("#", "").replace("*", "").strip()
    if len(clean_title) > 80:
        clean_title = clean_title[:77] + "..."
        
    # Extract takeaway if present
    takeaway = "Sustained financial compounding and disciplined capital management."
    for l in lines:
        if "takeaway" in l.lower() or "trend" in l.lower():
            takeaway = l.replace("*", "").strip()
            break
            
    return {
        "title": clean_title,
        "detailed_summary": raw_text,
        "business_takeaway": takeaway,
        "quantitative_data": ["Multi-year financial metrics transcribed in detailed_summary"],
        "tags": ["financial metrics", "annual report", f"page {page_no}"],
        "is_decorative": False
    }

repaired_count = 0
for k in tqdm(failed_keys, desc="Repairing Failed Image Summaries via GLM-OCR"):
    item = summaries[k]
    img_path = Path(item["path"])
    page_no = item["page_no"]
    nearby = item.get("summary", "")
    
    try:
        t0 = time.time()
        res = summarize_with_glm(img_path, page_no, nearby)
        # Remove error key
        item.pop("error", None)
        item.update(res)
        repaired_count += 1
        print(f"  Repaired Pic {k} (p.{page_no}) in {time.time()-t0:.2f}s: {res['title'][:50]}")
        
        # Save checkpoint
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
    except Exception as e:
        print(f"  Failed again on Pic {k}: {e}")

print(f"\nSuccessfully repaired {repaired_count}/{len(failed_keys)} entries in {CACHE_PATH.name}")
