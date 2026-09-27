import urllib.request
import json
import base64
import time
from io import BytesIO
from PIL import Image

def test_inference():
    with open('parsing/output/full_docling_parsed.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Test with Picture 11 (Page 8 - 20-year net income chart)
    pic_idx = 11
    pic = data['pictures'][pic_idx]
    prov = pic['prov'][0]
    page_no = str(prov['page_no'])
    bbox = prov['bbox']
    page_h = data['pages'][page_no]['size']['height']

    crop_box = (bbox['l'], page_h - bbox['t'], bbox['r'], page_h - bbox['b'])
    uri = data['pages'][page_no]['image']['uri']
    raw_img = Image.open(BytesIO(base64.b64decode(uri.split(',', 1)[1])))
    cropped = raw_img.crop(crop_box)

    buf = BytesIO()
    cropped.save(buf, format='PNG')
    crop_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    page_texts = [
        t['text'] for t in data['texts']
        if t.get('prov') and t['prov'][0].get('page_no') == int(page_no)
    ]
    context_snippet = ' | '.join(page_texts[:4])

    json_metadata = {
        'picture_index': pic_idx,
        'page_number': int(page_no),
        'dimensions_pt': {'width': round(bbox['r'] - bbox['l'], 1), 'height': round(bbox['t'] - bbox['b'], 1)},
        'nearby_text_from_docling': context_snippet[:400]
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

    print('Calling gemma4:26b with image + JSON metadata...')
    t0 = time.perf_counter()
    req_payload = {
        'model': 'gemma4:26b',
        'prompt': prompt,
        'images': [crop_b64],
        'stream': False
    }

    req = urllib.request.Request(
        'http://localhost:11434/api/generate',
        data=json.dumps(req_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.loads(resp.read().decode('utf-8'))

    elapsed = time.perf_counter() - t0
    print(f'Completed in {elapsed:.2f} seconds')
    print('Response:\n', res.get('response'))
    if res.get('thinking'):
        print('Thinking snippet:', res.get('thinking')[:300])

if __name__ == '__main__':
    test_inference()
