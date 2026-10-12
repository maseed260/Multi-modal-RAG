import json
with open('evaluation/ground_truth_dataset.json', encoding='utf-8') as f:
    d = json.load(f)
qs = d['questions']
print(f'Total questions: {len(qs)}')
zero = [q for q in qs if not q.get('ground_truth_chunk_ids')]
print(f'Questions with 0 matched chunks: {len(zero)}')
for q in zero:
    print('  ' + q['id'])
print()
for q in qs:
    ids = q.get('ground_truth_chunk_ids', [])
    chunk_ids_only = [e['chunk_id'] for e in ids]
    mods = list({e['modality'] for e in ids})
    print(f"  {q['id']:12s} [{q['difficulty']:7s}] {len(ids):2d} chunks  mods={mods}")
print('\nAll good!')
