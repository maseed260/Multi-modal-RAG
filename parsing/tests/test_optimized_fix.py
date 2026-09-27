import json
import base64
import time
import urllib.request
from io import BytesIO
from PIL import Image

def test_fix(pic_idx):
    with open('parsing/output/full_docling_parsed.json', 'r', encoding='utf-8') as f:
        doc_data = json.load(f)

    with open('parsing/output/classified_images.json', 'r', encoding='utf-8') as f:
        classified = json.load(f)

    pic_meta = classified[str(pic_idx)]
    page_no = pic_meta["page_no"]
    print(f"\nTesting optimized inference on Pic {pic_idx} (Page {page_no}, original size: {pic_meta['width']}x{pic_meta['height']})...")

    # Load image and resize if larger than 768px to prevent VRAM thrashing
    img = Image.open(pic_meta['path'])
    orig_w, orig_h = img.size
    max_dim = 768
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        new_size = (int(orig_w * scale), int(orig_h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f" -> Resized image for VLM: {orig_w}x{orig_h} -> {img.size[0]}x{img.size[1]}")

    buf = BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    json_metadata = {
        'picture_index': pic_idx,
        'page_number': page_no,
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
2. If it is a chart or diagram, provide a 2-sentence summary of metrics shown. If it is a headshot or decorative, provide a short description (e.g. 'Executive portrait').
3. Determine rag_action: 'keep_and_index' or 'discard'.

Return your answer strictly as a JSON object with keys:
'category', 'classification_reason', 'summary', 'rag_action'."""

    req_payload = {
        'model': 'gemma4:26b',
        'prompt': prompt,
        'images': [img_b64],
        'stream': False
    }

    req = urllib.request.Request(
        'http://localhost:11434/api/generate',
        data=json.dumps(req_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    el = time.perf_counter() - t0
    print(f" -> Completed in {el:.2f} seconds!")
    raw_resp = res.get("response", "").strip()
    print("RAW RESP LENGTH:", len(raw_resp))
    try:
        result = json.loads(raw_resp.strip())
        print(" -> Category:", result.get("category"))
        print(" -> Reason:", result.get("classification_reason"))
        print(" -> Summary:", result.get("summary"))
        print(" -> Action:", result.get("rag_action"))
    except Exception as e:
        print("JSON parse error:", e)
        print("START:", raw_resp[:150].encode("ascii", "replace").decode())
        print("END:", raw_resp[-150:].encode("ascii", "replace").decode())

if __name__ == '__main__':
    test_fix(11)
