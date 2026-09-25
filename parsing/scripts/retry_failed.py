import json
import base64
import time
import urllib.request
from pathlib import Path
from io import BytesIO
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "parsing" / "output"
CLASSIFIED_JSON_PATH = OUTPUT_DIR / "classified_images.json"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma4:26b"

def clean_json_response(raw_text):
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1]
    elif "```" in text:
        text = text.split("```", 1)[1]
    if "```" in text:
        text = text.rsplit("```", 1)[0]
    return text.strip()

def classify_single(pic_meta):
    img = Image.open(pic_meta['path'])
    orig_w, orig_h = img.size
    max_dim = 768
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        new_size = (int(orig_w * scale), int(orig_h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    json_metadata = {
        'picture_index': pic_meta['index'],
        'page_number': pic_meta['page_no'],
        'dimensions_pt': {'width': pic_meta['width'], 'height': pic_meta['height']}
    }

    prompt = f"""You are an expert document preprocessor and financial analyst for a Multi-Modal RAG pipeline.
Analyze the attached cropped image and the accompanying layout metadata from the document's serialized JSON.

Layout Metadata from Docling JSON:
{json.dumps(json_metadata, indent=2)}

Task:
1. Classify the image into exactly ONE category:
   - 'Financial Charts / Graphs'
   - 'Informative Diagram / Visual Table'
   - 'Executive Headshots / Portraits'
   - 'Decorative / Icon / Logo'
2. If it is a chart or diagram, provide a 2-sentence summary of metrics shown. If it is a headshot or decorative, provide a short 1-sentence description (e.g. 'Executive portrait').
3. Determine rag_action: 'keep_and_index' or 'discard'.

Return your answer strictly as a JSON object with keys:
'category', 'classification_reason', 'summary', 'rag_action'."""

    req_payload = {
        'model': MODEL_NAME,
        'prompt': prompt,
        'images': [img_b64],
        'stream': False
    }

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(req_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    
    clean_text = clean_json_response(res.get("response", ""))
    return json.loads(clean_text)

def main():
    with open(CLASSIFIED_JSON_PATH, "r", encoding="utf-8") as f:
        classified = json.load(f)

    failed_keys = [k for k, v in classified.items() if v.get("category") == "Error"]
    print(f"Retrying {len(failed_keys)} failed items: {failed_keys}")

    success_count = 0
    for k in failed_keys:
        meta = classified[k]
        print(f"\nProcessing Pic {k} (Page {meta['page_no']}, {meta['width']}x{meta['height']})...")
        t0 = time.perf_counter()
        try:
            res = classify_single(meta)
            classified[k] = {
                **meta,
                **res
            }
            el = time.perf_counter() - t0
            print(f" -> SUCCESS in {el:.2f}s: {res.get('category')} ({res.get('rag_action')})")
            success_count += 1
            # Save checkpoint
            with open(CLASSIFIED_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(classified, f, indent=2)
        except Exception as e:
            print(f" -> FAILED: {e}")

    print(f"\nCompleted retries: {success_count}/{len(failed_keys)} fixed!")

if __name__ == "__main__":
    main()
