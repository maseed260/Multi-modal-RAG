import json
import base64
import time
import re
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

# Identify entries that have placeholder / unextracted metrics or generic titles
target_keys = []
for k, v in summaries.items():
    if v.get("is_decorative") or v.get("rag_action") == "discard":
        continue
    qd = v.get("quantitative_data", [])
    ds = v.get("detailed_summary", "")
    title = v.get("title", "")
    if (len(qd) <= 1 or 
        any("Detailed figures" in str(x) or "transcribed in" in str(x) for x in qd) or 
        title.startswith("The title of the") or 
        title.startswith("1. Title:") or
        title.startswith("The title is")):
        target_keys.append(k)

print(f"Total entries in cache: {len(summaries)}")
print(f"Entries requiring high-fidelity metric re-extraction: {len(target_keys)}: {target_keys}")

def extract_metrics_with_gemma(image_path: Path, page_no: int, nearby_text: str = "") -> dict:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    prompt = f"""You are an institutional financial analyst. Analyze this financial visual element from the JPMorgan Chase & Co. 2025 Annual Report (Page {page_no}).
Nearby context: {nearby_text[:200] if nearby_text else 'N/A'}

Extract:
1. Exact Title and Subtitle (including time horizon and units e.g. $ in billions).
2. Data series names, categories, and legends.
3. Summary of key metrics shown. Explicitly transcribe all milestone numbers, dollar values ($B, $M, per share), and percentages (%).
4. Specific values from callouts, annotations, or footnotes.
5. Strategic business takeaway for institutional investors.

Keep your response factual, structured, and ensure all visible numbers, dollar figures, and percentages are explicitly transcribed."""

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
    
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        raw_text = res.get("response", "").strip()
        
    # Extract title
    title = f"Visual Figure (Page {page_no})"
    m_title = re.search(r'\*\*Title(?:\s*and\s*Subtitle)?:\*\*\s*(.+)', raw_text)
    if not m_title:
        m_title = re.search(r'#+\s*(.+)', raw_text)
    if not m_title:
        for line in raw_text.splitlines():
            line_str = line.strip().replace('*', '').replace('#', '')
            if line_str.lower().startswith('title:'):
                title = line_str.split(':', 1)[1].strip()
                break
    else:
        title = m_title.group(1).replace('*', '').replace('#', '').strip()
            
    # Extract takeaway
    takeaway = "Sustained financial compounding and disciplined capital management."
    m_take = re.search(r'\*\*5\.\s*Business Takeaway.*?\*\*\s*\n*(.+)', raw_text)
    if not m_take:
        m_take = re.search(r'(?:Business Takeaway|Strategic Takeaway)[:\s]+\s*(.+)', raw_text, re.IGNORECASE)
    if m_take:
        takeaway = m_take.group(1).replace('*', '').strip()
        
    # Extract numeric metrics
    found_metrics = re.findall(r'(\$\d+(?:\.\d+)?[BMKbmk]?|\b\d+(?:\.\d+)?%)', raw_text)
    unique_metrics = list(dict.fromkeys(found_metrics))[:15]
    if not unique_metrics:
        unique_metrics = [f"Metrics for page {page_no} charted in summary"]
        
    return {
        "title": title[:90],
        "detailed_summary": raw_text,
        "business_takeaway": takeaway,
        "quantitative_data": unique_metrics,
        "tags": ["financial metrics", "annual report", f"page {page_no}", "jpmorgan chase"],
        "is_decorative": False
    }

if __name__ == "__main__":
    repaired = 0
    for k in target_keys:
        item = summaries[k]
        img_path = Path(item["path"])
        page_no = item["page_no"]
        nearby = item.get("summary", "")
        
        print(f"\nProcessing Pic {k} (Page {page_no})...")
        t0 = time.time()
        try:
            res = extract_metrics_with_gemma(img_path, page_no, nearby)
            item.update(res)
            repaired += 1
            print(f"  [SUCCESS] Finished in {time.time()-t0:.1f}s | Title: {res['title'][:50]}")
            print(f"  Metrics extracted ({len(res['quantitative_data'])}): {res['quantitative_data'][:6]}")
            
            # Save checkpoint after each image
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2)
        except Exception as e:
            print(f"  [ERROR] Failed on Pic {k}: {e}")
            
    print(f"\nRe-extracted high-fidelity metrics for {repaired}/{len(target_keys)} figures.")
