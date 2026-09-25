import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            },
            "kernelspec": {
                "display_name": "Python (.venv)",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    add_md("""# Multi-Modal PDF Preprocessing: Visual Classification with Local VLM (`gemma4:26b`)

This notebook preprocesses visual elements (figures, charts, diagrams, headshots) extracted from `jpmc_annualreport-2025.pdf` via **Docling**.

### Key Workflow:
1. **Load Serialized Docling JSON**: Read `parsing/output/full_docling_parsed.json`.
2. **Crop & Export Visuals**: Extract all picture bounding boxes directly from the embedded 72 DPI page canvas images using coordinate transformations.
3. **Geometric Pre-Filtering**: Automatically tag tiny elements (area $< 2500\\text{ pt}^2$, e.g., bullet icons) as `Decorative / Icon / Logo` with zero GPU compute.
4. **Multimodal Context Construction**: For each candidate figure, merge the **cropped image** with layout metadata from Docling (page number, bounding box dimensions).
5. **VLM Classification & Dense Captioning**: Use local **`gemma4:26b`** via Ollama to classify each image and generate dense financial summaries for vector indexing.
6. **Checkpointing & Persistence**: Results are incrementally written to `parsing/output/classified_images.json` so the job can be safely paused and resumed.""")

    add_code("""import sys
import json
import base64
import time
import urllib.request
from pathlib import Path
from io import BytesIO
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Configure directories
PROJECT_ROOT = Path("..").resolve()
OUTPUT_DIR = PROJECT_ROOT / "parsing" / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DOC_JSON_PATH = OUTPUT_DIR / "full_docling_parsed.json"
CLASSIFIED_JSON_PATH = OUTPUT_DIR / "classified_images.json"

print(f"Project root: {PROJECT_ROOT}")
print(f"Docling JSON exists: {DOC_JSON_PATH.exists()}")
print(f"Figures directory: {FIGURES_DIR}")""")

    add_md("""### Step 1: Verify Local Ollama Service and Model Availability""")

    add_code("""# Check Ollama connection and ensure gemma4:26b is ready
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma4:26b"

try:
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
        models_data = json.loads(resp.read().decode())
        available_models = [m["name"] for m in models_data.get("models", [])]
        print(f"Available Ollama models: {available_models}")
        assert any(MODEL_NAME in m for m in available_models), f"{MODEL_NAME} not found in Ollama!"
        print(f"SUCCESS: {MODEL_NAME} is available for multimodal inference!")
except Exception as e:
    print(f"Error connecting to Ollama: {e}")""")

    add_md("""### Step 2: Load Parsed Docling Document & Inspect Picture Distribution""")

    add_code("""# Load serialized Docling document
print(f"Loading {DOC_JSON_PATH.name}...")
with open(DOC_JSON_PATH, "r", encoding="utf-8") as f:
    doc_data = json.load(f)

pictures = doc_data.get("pictures", [])
pages_map = doc_data.get("pages", {})
texts = doc_data.get("texts", [])

print(f"Total pages: {len(pages_map)}")
print(f"Total text items: {len(texts)}")
print(f"Total visual figures (pictures): {len(pictures)}")""")

    add_md("""### Step 3: Crop All Visual Elements from Page Canvas Images
Docling uses PDF standard coordinates where origin is `BOTTOMLEFT`. 
We transform these to PIL canvas space (`TOPLEFT`):
$$\\text{crop\\_box} = (l,\\ \\text{page\\_height} - t,\\ r,\\ \\text{page\\_height} - b)$$""")

    add_code("""def crop_and_save_picture(pic_idx, pic, pages_map, figures_dir):
    \"\"\"Extracts and crops an image from page canvas, saving to disk.\"\"\"
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
    
    # Decode base64 page canvas
    uri = page_info["image"]["uri"]
    img_b64 = uri.split(",", 1)[1]
    raw_page_img = Image.open(BytesIO(base64.b64decode(img_b64)))
    
    cropped_img = raw_page_img.crop(crop_box)
    
    # Save cropped image
    out_path = figures_dir / f"pic_{pic_idx:02d}_page_{page_no}.png"
    cropped_img.save(out_path, format="PNG")
    
    return out_path, cropped_img

# Crop and save all 73 figures
print("Cropping and saving all pictures...")
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

print(f"Successfully saved {len(saved_pictures)} cropped figures to {FIGURES_DIR}!")""")

    add_md("""### Step 4: Multi-Modal Classification with `gemma4:26b` & Checkpointing

We construct the multimodal payload:
1. **Cropped Image** (Base64, scaled to max dimension 768px to prevent VRAM saturation)
2. **Serialized Docling JSON Context** (Page number, Bounding Box dimensions)
3. **Structured Classification Request** (Taxonomy: Financial Charts, Informative Diagram, Headshots, Decorative)

*Resume feature: If an image is already in `classified_images.json`, it is skipped.*""")

    add_code("""def clean_json_response(raw_text):
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1]
    elif "```" in text:
        text = text.split("```", 1)[1]
    if "```" in text:
        text = text.rsplit("```", 1)[0]
    return text.strip()

def classify_figure(pic_meta, model_name=MODEL_NAME):
    \"\"\"Sends image + Docling serialized JSON context to local Ollama VLM.\"\"\"
    # Heuristic fast-path: tiny icons / bullets
    if pic_meta["area"] < 2500:
        return {
            "category": "Decorative / Icon / Logo",
            "classification_reason": f"Geometric filter: Tiny dimensions ({pic_meta['width']}x{pic_meta['height']} pt, area {pic_meta['area']}).",
            "summary": "Decorative bullet, icon, or formatting marker.",
            "rag_action": "discard"
        }
    
    # Read cropped image and resize if needed to prevent VRAM saturation
    img = Image.open(pic_meta['path'])
    orig_w, orig_h = img.size
    max_dim = 768
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.LANCZOS)
        
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    json_metadata = {
        "picture_index": pic_meta["index"],
        "page_number": pic_meta["page_no"],
        "dimensions_pt": {"width": pic_meta["width"], "height": pic_meta["height"]}
    }
    
    prompt = f\"\"\"You are an expert document preprocessor and financial analyst for a Multi-Modal RAG pipeline.
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
'category', 'classification_reason', 'summary', 'rag_action'.\"\"\"

    req_payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False
    }
    
    req = urllib.request.Request(
        f\"{OLLAMA_URL}/api/generate\",
        data=json.dumps(req_payload).encode(\"utf-8\"),
        headers={\"Content-Type\": \"application/json\"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode(\"utf-8\"))
            clean_text = clean_json_response(res.get(\"response\", \"\"))
            return json.loads(clean_text)
    except Exception as e:
        return {
            \"category\": \"Error\",
            \"classification_reason\": str(e),
            \"summary\": \"\",
            \"rag_action\": \"manual_review\"
        }""")

    add_md("""### Step 5: Run Batch Classification with Checkpointing
Runs the classification loop with instant persistence. If interrupted, simply re-run this cell to resume!""")

    add_code("""# Load existing checkpoint if available
classified_results = {}
if CLASSIFIED_JSON_PATH.exists():
    with open(CLASSIFIED_JSON_PATH, "r", encoding="utf-8") as f:
        classified_results = json.load(f)
    print(f"Loaded {len(classified_results)} existing classifications from checkpoint!")

# Process items
for pic_meta in tqdm(saved_pictures, desc="Classifying figures"):
    key = str(pic_meta["index"])
    if key in classified_results and classified_results[key].get("category") != "Error":
        continue  # Skip already classified items
        
    res = classify_figure(pic_meta)
    classified_results[key] = {
        **pic_meta,
        **res
    }
    
    # Save checkpoint after each item
    with open(CLASSIFIED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(classified_results, f, indent=2)

print(f"All {len(classified_results)} figures classified and saved to {CLASSIFIED_JSON_PATH}!")""")

    add_md("""### Step 6: Inspect Results & Category Distribution""")

    add_code("""# Summary dataframe
df = pd.DataFrame.from_dict(classified_results, orient="index")

# Analytical RAG Policy: Index only quantitative charts and informative tables
analytical_cats = ["Financial Charts / Graphs", "Informative Diagram / Visual Table"]
df["rag_action"] = df["category"].apply(lambda c: "keep_and_index" if c in analytical_cats else "discard")

print("=== Category Distribution ===")
print(df["category"].value_counts())

print("\\n=== RAG Action Distribution (Analytical Policy) ===")
print(df["rag_action"].value_counts())

# Display top financial charts
charts_df = df[df["category"] == "Financial Charts / Graphs"][["index", "page_no", "width", "height", "summary"]]
print(f"\\nDetected {len(charts_df)} Financial Charts:")
charts_df.head(10)""")

    nb_path = Path(__file__).resolve().parents[1] / "02_image_classification_preprocessing.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated notebook at: {nb_path.resolve()}")

if __name__ == "__main__":
    create_notebook()
