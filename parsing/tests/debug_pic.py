import json
import base64
import time
import urllib.request
from pathlib import Path

with open('parsing/output/classified_images.json', 'r', encoding='utf-8') as f:
    classified = json.load(f)

with open('parsing/output/full_docling_parsed.json', 'r', encoding='utf-8') as f:
    doc_data = json.load(f)

texts = doc_data.get('texts', [])

def test_pic(pic_idx):
    pic_meta = classified[str(pic_idx)]
    page_no = pic_meta["page_no"]
    w = pic_meta["width"]
    h = pic_meta["height"]
    print(f"\n=======================================================")
    print(f"=== Testing Pic {pic_idx} on Page {page_no} ({w}x{h}) ===")
    print(f"=======================================================")
    
    with open(pic_meta['path'], 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    page_texts = [
        t['text'].strip() for t in texts
        if t.get('prov') and t['prov'][0].get('page_no') == page_no and len(t.get('text', '').strip()) > 3
    ]
    nearby_context = ' | '.join(page_texts[:4])
    json_metadata = {
        'picture_index': pic_idx,
        'page_number': page_no,
        'dimensions_pt': {'width': w, 'height': h},
        'nearby_text_from_docling': nearby_context[:400]
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
        'model': 'gemma4:26b',
        'prompt': prompt,
        'images': [img_b64],
        'stream': False,
        'format': 'json',
        'options': {
            'num_predict': 1024
        }
    }

    req = urllib.request.Request(
        'http://localhost:11434/api/generate',
        data=json.dumps(req_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            res = json.loads(resp.read().decode('utf-8'))
        el = time.perf_counter() - t0
        print(f"Completed in {el:.2f} seconds")
        print("Done reason:", res.get("done_reason"))
        print("Prompt eval duration (s):", res.get("prompt_eval_duration", 0)/1e9)
        print("Eval count:", res.get("eval_count"), "Eval duration (s):", res.get("eval_duration", 0)/1e9)
        raw_resp = res.get("response", "")
        print("Raw response length:", len(raw_resp))
        print("Raw response:\n", repr(raw_resp))
        if res.get("thinking"):
            print("Thinking length:", len(res.get("thinking")))
            print("Thinking snippet:\n", res.get("thinking")[:200])
        
        # Test JSON parse
        parsed = json.loads(raw_resp)
        print("Successfully parsed JSON:", parsed.get("category"))
    except Exception as e:
        print(f"Error after {time.perf_counter()-t0:.2f}s: {e}")

if __name__ == '__main__':
    # Test pic 10 (which had Unterminated string) and pic 11 (which had timed out)
    test_pic(10)
