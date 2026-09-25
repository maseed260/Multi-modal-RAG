import json
import base64
import time
import urllib.request
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "parsing" / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DOC_JSON_PATH = OUTPUT_DIR / "full_docling_parsed.json"
CLASSIFIED_JSON_PATH = OUTPUT_DIR / "classified_images.json"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma4:26b"

def crop_and_save_picture(pic_idx, pic, pages_map, figures_dir):
    prov = pic["prov"][0]
    page_no = str(prov["page_no"])
    bbox = prov["bbox"]
    
    if page_no not in pages_map or "image" not in pages_map[page_no]:
        return None, None
    
    page_info = pages_map[page_no]
    page_h = page_info["size"]["height"]
    
    # Coordinate conversion: BOTTOMLEFT -> TOPLEFT
    crop_box = (
        max(0, bbox["l"]),
        max(0, page_h - bbox["t"]),
        bbox["r"],
        page_h - bbox["b"]
    )
    
    uri = page_info["image"]["uri"]
    img_b64 = uri.split(",", 1)[1]
    raw_page_img = Image.open(BytesIO(base64.b64decode(img_b64)))
    cropped_img = raw_page_img.crop(crop_box)
    
    out_path = figures_dir / f"pic_{pic_idx:02d}_page_{page_no}.png"
    cropped_img.save(out_path, format="PNG")
    return out_path, cropped_img

def get_nearby_text_context(page_no, texts, max_items=4):
    page_texts = [
        t["text"].strip() for t in texts
        if t.get("prov") and t["prov"][0].get("page_no") == page_no and len(t.get("text", "").strip()) > 3
    ]
    return " | ".join(page_texts[:max_items])

def classify_figure(pic_meta, texts, model_name=MODEL_NAME):
    # Heuristic fast-path: tiny icons / bullets
    if pic_meta["area"] < 2500:
        return {
            "category": "Decorative / Icon / Logo",
            "classification_reason": f"Geometric filter: Tiny dimensions ({pic_meta['width']}x{pic_meta['height']} pt, area {pic_meta['area']}).",
            "summary": "Decorative bullet, icon, or formatting marker.",
            "rag_action": "discard"
        }
    
    with open(pic_meta["path"], "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    nearby_context = get_nearby_text_context(pic_meta["page_no"], texts)
    
    json_metadata = {
        "picture_index": pic_meta["index"],
        "page_number": pic_meta["page_no"],
        "dimensions_pt": {"width": pic_meta["width"], "height": pic_meta["height"]},
        "nearby_text_from_docling": nearby_context[:400]
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
2. Provide a 2-sentence financial summary of what this image shows.
3. Determine rag_action: 'keep_and_index' or 'discard'.

Return your answer strictly as a JSON object with keys:
'category', 'classification_reason', 'summary', 'rag_action'."""

    req_payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "format": "json"
    }
    
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(req_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            raw_text = res.get("response", "").strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1]
                if raw_text.endswith("```"):
                    raw_text = raw_text.rsplit("```", 1)[0]
            return json.loads(raw_text.strip())
    except Exception as e:
        return {
            "category": "Error",
            "classification_reason": str(e),
            "summary": "",
            "rag_action": "manual_review"
        }

def main():
    print(f"Loading {DOC_JSON_PATH.name}...")
    with open(DOC_JSON_PATH, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    pictures = doc_data.get("pictures", [])
    pages_map = doc_data.get("pages", {})
    texts = doc_data.get("texts", [])

    print(f"Total pictures: {len(pictures)}")
    print("Extracting and saving all cropped figures to disk...")
    saved_pictures = []
    for idx, pic in enumerate(pictures):
        out_path, img = crop_and_save_picture(idx, pic, pages_map, FIGURES_DIR)
        prov = pic["prov"][0]
        bbox = prov["bbox"]
        w = round(abs(bbox["r"] - bbox["l"]), 1)
        h = round(abs(bbox["t"] - bbox["b"]), 1)
        saved_pictures.append({
            "index": idx,
            "page_no": prov["page_no"],
            "path": str(out_path),
            "width": w,
            "height": h,
            "area": round(w * h, 1),
            "captions": pic.get("captions", [])
        })

    print(f"All {len(saved_pictures)} pictures saved to {FIGURES_DIR}")

    # Checkpoint loading
    classified_results = {}
    if CLASSIFIED_JSON_PATH.exists():
        try:
            with open(CLASSIFIED_JSON_PATH, "r", encoding="utf-8") as f:
                classified_results = json.load(f)
            print(f"Loaded {len(classified_results)} existing classifications from {CLASSIFIED_JSON_PATH.name}")
        except Exception:
            pass

    # Batch loop
    for pic_meta in tqdm(saved_pictures, desc="Classifying with gemma4:26b"):
        key = str(pic_meta["index"])
        if key in classified_results and classified_results[key].get("category") not in [None, "Error"]:
            continue
            
        print(f"\nProcessing Picture {pic_meta['index']} on Page {pic_meta['page_no']} (Area: {pic_meta['area']})...")
        res = classify_figure(pic_meta, texts)
        classified_results[key] = {
            **pic_meta,
            **res
        }
        print(f" -> Result: {res.get('category')} ({res.get('rag_action')})")
        
        # Save checkpoint immediately
        with open(CLASSIFIED_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(classified_results, f, indent=2)

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
