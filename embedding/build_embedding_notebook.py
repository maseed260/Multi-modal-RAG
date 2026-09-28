import json
import os
from pathlib import Path

def create_notebook():
    notebook_dir = Path("embedding")
    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / "01_qdrant_hybrid_embedding_indexing.ipynb"

    cells = []

    def md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    md("""
# Pillar 3: Hybrid Multi-Vector Embedding & Indexing in Qdrant

Welcome to **Phase 3** of our Enterprise Multi-Modal RAG pipeline.

In Phase 1 and 2, we completed the ingestion benchmarks and generated **1,452 enriched multi-modal chunks** stored in `chunking/output/unified_chunks.jsonl`.

Now, we will embed and store these chunks into **Qdrant Vector Database** using a production-grade **Hybrid Multi-Vector Architecture**:
1. **Dense Semantic Embeddings:** Generated using the local **`qwen3-embedding:4b`** model via Ollama with **1024-dimension Matryoshka Representation Learning (MRL)** for deep conceptual and semantic matching.
2. **Sparse Lexical Embeddings:** Generated using **FastEmbed BM25 (`Qdrant/bm25`)** for exact numerical, ticker, and non-GAAP financial keyword matching (e.g. `$57,048`, `ROTCE`, `TBVPS`, `CET1`, `(a)`).
3. **Qdrant Vector Storage & HNSW Indexing:** Creating a multi-vector collection with custom HNSW graph parameters and Sparse Inverted Indexes.
4. **Payload Indexing for Rapid Pre-Filtering:** Indexing metadata fields (`modality`, `page_start`, `toc_section`, `visual_category`) to enable instant pre-filtered hybrid retrieval.
5. **Hybrid Search with Reciprocal Rank Fusion (RRF):** Combining dense semantic recall and sparse keyword precision into a unified ranked result set.
    """)

    # Qdrant Table vs Point explanation
    md("""
---
### 💡 Architectural Deep Dive: Does Qdrant Store Data as a "Table"?

**Short Answer:** No, Qdrant is not a relational or tabular SQL database, but its **Collection & Point** model provides even greater flexibility, and the **Qdrant Web UI renders it as an interactive table**.

#### How Qdrant Stores Data Under the Hood:
1. **Collections:** Equivalent to a SQL table or MongoDB collection. A collection holds points that share a common vector and indexing configuration.
2. **Points:** The fundamental unit of data in Qdrant (similar to a row in SQL or a document in MongoDB). Each point contains:
   - **`id`:** A unique identifier (UUID string or 64-bit integer).
   - **`vector`:** A vector or named vector dictionary. In our hybrid setup, each point holds two vectors simultaneously:
     - `dense`: 1024-dimensional floating point array `[0.012, -0.045, ...]`
     - `sparse`: Sparse vector with token indices and BM25 weights `{"indices": [104, 892], "values": [1.45, 0.82]}`
   - **`payload`:** A JSON object storing arbitrary structured metadata (`modality`, `doc_id`, `page_range`, `toc_path`, `footnotes`, raw markdown `text`, etc.).
3. **Payload Indexes:** Unlike standard document stores that must scan every JSON document to filter, Qdrant builds specialized in-memory indexes (Keyword, Integer, Geo, Full-text) on specified payload fields. This enables **Filtered HNSW Search**, where filtering happens directly during graph traversal with zero performance penalty.
4. **Web UI Dashboard:** When you open `http://localhost:6333/dashboard`, Qdrant presents your collection as an **interactive data table** displaying Point IDs, Vectors, and Payload columns!
    """)

    # Cell 1: Environment & Setup
    md("## 1. Environment & Dependency Verification")
    code("""# 1. Imports and Verification
import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any

import requests
from tqdm.auto import tqdm
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

print(f"Python Version: {sys.version.split()[0]}")
print("Qdrant Client & FastEmbed loaded successfully.")
    """)

    # Cell 2: Health check
    md("## 2. Connect to Qdrant & Ollama Services")
    code("""# 2. Verify Connections to Qdrant and Ollama
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
DENSE_MODEL_NAME = "qwen3-embedding:4b"
EMBEDDING_DIM = 1024

# Verify Qdrant connection
client = QdrantClient(url=QDRANT_URL)
try:
    collections_response = client.get_collections()
    print(f"✅ Connected to Qdrant at {QDRANT_URL} (Existing collections: {len(collections_response.collections)})")
except Exception as e:
    print(f"❌ Failed to connect to Qdrant at {QDRANT_URL}. Ensure Docker container is running: {e}")

# Verify Ollama connection and model availability
try:
    tags_resp = requests.get(f"{OLLAMA_URL}/api/tags").json()
    available_models = [m['name'] for m in tags_resp.get('models', [])]
    if any(DENSE_MODEL_NAME in m for m in available_models):
        print(f"✅ Connected to Ollama at {OLLAMA_URL}. Found '{DENSE_MODEL_NAME}'.")
    else:
        print(f"⚠️ Connected to Ollama, but '{DENSE_MODEL_NAME}' was not found in: {available_models}")
except Exception as e:
    print(f"❌ Failed to connect to Ollama at {OLLAMA_URL}: {e}")
    """)

    # Cell 3: Load Chunks
    md("## 3. Load & Inspect Unified Multi-Modal Chunks")
    code("""# 3. Load chunks from unified_chunks.jsonl
PROJECT_ROOT = Path("..")
CHUNKS_PATH = PROJECT_ROOT / "chunking" / "output" / "unified_chunks.jsonl"

if not CHUNKS_PATH.exists():
    # Fallback to local path if executed from root
    CHUNKS_PATH = Path("chunking/output/unified_chunks.jsonl")

print(f"Loading chunks from: {CHUNKS_PATH.resolve()}")

chunks = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

print(f"✅ Total Chunks Loaded: {len(chunks):,}")

# Breakdown by Modality
modality_counts = {}
for c in chunks:
    m = c.get("modality", "unknown")
    modality_counts[m] = modality_counts.get(m, 0) + 1

for mod, count in modality_counts.items():
    print(f"   • {mod.capitalize()} Chunks: {count:,} ({count/len(chunks)*100:.1f}%)")
    """)

    # Cell 4: Inspect Sample Chunks
    md("## 4. Inspect Modality Examples (Text, Table, Figure)")
    code("""# 4. Display sample chunks across modalities
for target_mod in ["text", "table", "figure"]:
    sample = next((c for c in chunks if c.get("modality") == target_mod), None)
    if sample:
        print(f"==================================================")
        print(f"MODALITY: {target_mod.upper()} | ID: {sample['chunk_id']}")
        print(f"TOC Path: {' > '.join(sample.get('toc', {}).get('toc_path', []))}")
        print(f"Pages: {sample.get('provenance', {}).get('page_range')}")
        if target_mod == "figure":
            print(f"Image Path: {sample.get('multimodal', {}).get('image_path')}")
            print(f"Metrics: {sample.get('multimodal', {}).get('quantitative_metrics', [])[:3]}")
        print(f"Snippet:\n{sample['text'][:280]}...")
        print(f"==================================================\n")
    """)

    # Cell 5: Create Qdrant Collection
    md("""
## 5. Define Qdrant Collection Schema with Named Vectors & HNSW

Here we create the collection `jpmc_annual_report_2025` with:
- **`dense` Named Vector:**
  - Size: **1024**
  - Distance: **Cosine**
  - **HNSW Parameters:** `m=16` (number of bi-directional links per node), `ef_construct=100` (search depth during index construction for high recall).
- **`sparse` Named Vector:**
  - Configured with `SparseIndexParams(on_disk=False)` for in-memory inverted index lexical search.
    """)
    code("""# 5. Create Qdrant Collection with Dense HNSW and Sparse BM25
COLLECTION_NAME = "jpmc_annual_report_2025"

# If collection already exists, delete to ensure clean schema recreation
if client.collection_exists(COLLECTION_NAME):
    print(f"Removing existing collection '{COLLECTION_NAME}'...")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=EMBEDDING_DIM,
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
                on_disk=False
            )
        )
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(
                on_disk=False
            )
        )
    }
)

print(f"✅ Collection '{COLLECTION_NAME}' created successfully with Dense (1024d HNSW) + Sparse (BM25) vectors.")
    """)

    # Cell 6: Create Payload Indexes
    md("""
## 6. Create Payload Indexes for High-Performance Filtering

In multi-modal financial RAG, users frequently ask queries like:
- *"Show me all financial highlights tables from pages 1-10"*
- *"Retrieve charts showing ROTCE progression"*
- *"Search executive letters only"*

Without payload indexes, a vector database must perform a brute-force filter over the entire dataset or post-filter after approximate nearest neighbor search.
With Qdrant **Payload Indexes**, filters are indexed ahead of time into B-trees/hashsets, enabling **single-stage pre-filtered HNSW traversal**.
    """)
    code("""# 6. Create Payload Indexes on Filtering Attributes
INDEX_FIELDS = [
    ("modality", models.PayloadSchemaType.KEYWORD),
    ("doc_id", models.PayloadSchemaType.KEYWORD),
    ("page_start", models.PayloadSchemaType.INTEGER),
    ("page_end", models.PayloadSchemaType.INTEGER),
    ("toc_section", models.PayloadSchemaType.KEYWORD),
    ("toc_level", models.PayloadSchemaType.INTEGER),
    ("visual_category", models.PayloadSchemaType.KEYWORD),
    ("chunk_id", models.PayloadSchemaType.KEYWORD),
]

for field_name, schema_type in INDEX_FIELDS:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field_name,
        field_schema=schema_type
    )
    print(f"  • Created payload index on field: '{field_name}' ({schema_type.value})")

print("✅ All payload indexes successfully established.")
    """)

    # Cell 7: Embedding Generators
    md("## 7. Initialize Dense (Qwen) & Sparse (BM25) Embedding Generators")
    code("""# 7. Dense & Sparse Embedding Helper Functions

# Initialize FastEmbed BM25 Sparse model
print("Initializing FastEmbed BM25 sparse model (Qdrant/bm25)...")
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def get_dense_embeddings_batch(texts: List[str], model: str = DENSE_MODEL_NAME, dim: int = EMBEDDING_DIM) -> List[List[float]]:
    \"\"\"Fetch dense embeddings from Ollama /api/embed with 1024d MRL truncation.\"\"\"
    payload = {
        "model": model,
        "input": texts,
        "dimensions": dim
    }
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("embeddings", [])

def get_sparse_embeddings_batch(texts: List[str]) -> List[models.SparseVector]:
    \"\"\"Compute BM25 sparse vector representations (token indices and weights).\"\"\"
    sparse_generator = bm25_model.embed(texts)
    sparse_vectors = []
    for s in sparse_generator:
        sparse_vectors.append(models.SparseVector(
            indices=s.indices.tolist(),
            values=s.values.tolist()
        ))
    return sparse_vectors

# Test on a small query
test_text = "JPMorgan Chase reported net income of $57.0 billion for 2025."
t0 = time.time()
test_dense = get_dense_embeddings_batch([test_text])[0]
test_sparse = get_sparse_embeddings_batch([test_text])[0]
dt = time.time() - t0

print(f"✅ Test Embedding Verified in {dt:.3f}s:")
print(f"   • Dense Vector Shape: {len(test_dense)} floats")
print(f"   • Sparse Non-Zero Tokens: {len(test_sparse.indices)} terms")
    """)

    # Cell 8: Ingestion Pipeline
    md("""
## 8. Run Batch Ingestion & Upsert Pipeline into Qdrant

Now we process all 1,452 chunks in batches:
1. Extract the text representation for embedding.
2. Batch-encode dense vectors via Ollama `qwen3-embedding:4b` at 1024 dimensions.
3. Batch-encode sparse vectors via `FastEmbed BM25`.
4. Construct rich payload containing:
   - Identifiers: `chunk_id`, `modality`, `token_count`
   - Provenance: `doc_id`, `doc_title`, `page_start`, `page_end`, `citation`
   - TOC Section Tree: `toc_section`, `toc_parent`, `toc_path`, `toc_level`
   - Graph Navigation: `prev_chunk_id`, `next_chunk_id`, `section_sibling_ids`
   - Multimodal Details: `image_path`, `visual_category`, `quantitative_metrics`
   - Inlined Footnotes: `inlined_footnotes`
   - Full Content: `text`
5. Upsert points into Qdrant using deterministic UUIDs generated from `chunk_id`.
    """)
    code("""# 8. Batch Upsert Pipeline
BATCH_SIZE = 32
total_chunks = len(chunks)
print(f"Starting ingestion of {total_chunks:,} chunks into '{COLLECTION_NAME}' (Batch size: {BATCH_SIZE})...")

start_time = time.time()
total_points_upserted = 0

for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Ingesting Batches"):
    batch_chunks = chunks[i:i + BATCH_SIZE]
    batch_texts = [c["text"] for c in batch_chunks]
    
    # 1. Compute Dense Embeddings (Ollama Qwen3 1024d)
    dense_vectors = get_dense_embeddings_batch(batch_texts)
    
    # 2. Compute Sparse Embeddings (FastEmbed BM25)
    sparse_vectors = get_sparse_embeddings_batch(batch_texts)
    
    # 3. Construct Qdrant PointStruct list
    points = []
    for chunk, dense_vec, sparse_vec in zip(batch_chunks, dense_vectors, sparse_vectors):
        chunk_id = chunk["chunk_id"]
        # Generate deterministic UUID from chunk_id
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        
        # Build comprehensive payload for filtering & retrieval
        prov = chunk.get("provenance", {})
        toc = chunk.get("toc", {})
        nav = chunk.get("navigation", {})
        mm = chunk.get("multimodal") or {}
        
        payload = {
            "chunk_id": chunk_id,
            "modality": chunk.get("modality"),
            "token_count": chunk.get("token_count", 0),
            "char_count": chunk.get("char_count", 0),
            
            # Provenance
            "doc_id": prov.get("doc_id", "jpmc_annualreport-2025"),
            "doc_title": prov.get("doc_title", ""),
            "page_start": prov.get("page_start", 0),
            "page_end": prov.get("page_end", 0),
            "page_range": prov.get("page_range", []),
            "citation": prov.get("citation", ""),
            
            # TOC Hierarchy
            "toc_section": toc.get("toc_section", ""),
            "toc_parent": toc.get("toc_parent", ""),
            "toc_path": toc.get("toc_path", []),
            "toc_level": toc.get("toc_level", 0),
            
            # Navigation Pointers
            "prev_chunk_id": nav.get("prev_chunk_id"),
            "next_chunk_id": nav.get("next_chunk_id"),
            "section_sibling_ids": nav.get("section_sibling_ids", []),
            
            # Multimodal Metadata
            "image_path": mm.get("image_path"),
            "visual_category": mm.get("visual_category"),
            "quantitative_metrics": mm.get("quantitative_metrics", []),
            
            # Footnotes & Text
            "inlined_footnotes": chunk.get("inlined_footnotes", []),
            "text": chunk.get("text", "")
        }
        
        points.append(models.PointStruct(
            id=point_id,
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec
            },
            payload=payload
        ))
    
    # 4. Upsert batch into Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )
    total_points_upserted += len(points)

elapsed_total = time.time() - start_time
print(f"\\n🎉 Successfully indexed {total_points_upserted:,} points in {elapsed_total:.2f} seconds ({total_points_upserted/elapsed_total:.1f} chunks/sec)!")
    """)

    # Cell 9: Collection Info & Dashboard Verification
    md("## 9. Verify Collection Statistics & Dashboard Access")
    code("""# 9. Verify Collection Status
col_info = client.get_collection(COLLECTION_NAME)

print(f"Collection Name:     {COLLECTION_NAME}")
print(f"Status:              {col_info.status.value.upper()}")
print(f"Points Count:        {col_info.points_count:,}")
print(f"Indexed Vectors:     {col_info.indexed_vectors_count:,}")
print(f"Segments Count:      {len(col_info.segments)}")

print("\\n🔗 Qdrant Web UI Dashboard:")
print(f"Open in your browser: http://localhost:6333/dashboard")
print(f"Navigate to '{COLLECTION_NAME}' to view your indexed points and payloads rendered in an interactive table view!")
    """)

    # Cell 10: Hybrid Search Helper & Validation Queries
    md("""
## 10. Hybrid Search with Reciprocal Rank Fusion (RRF)

Now we test the collection using Qdrant's native **Hybrid Search Query API** with **Reciprocal Rank Fusion (RRF)**:
- Prefetches top-K candidates from the **Dense vector index** (semantic similarity).
- Prefetches top-K candidates from the **Sparse vector index** (exact lexical matches for figures and terms).
- Automatically fuses candidate ranks using RRF so neither scale dominates.
    """)
    code("""# 10. Hybrid Search Function
def search_hybrid(
    query_text: str,
    filter_conditions: models.Filter = None,
    limit: int = 5,
    dense_prefetch_limit: int = 25,
    sparse_prefetch_limit: int = 25
):
    \"\"\"Execute a hybrid search fusing dense semantic vectors and sparse BM25 vectors via RRF.\"\"\"
    # 1. Embed query
    query_dense = get_dense_embeddings_batch([query_text])[0]
    query_sparse = get_sparse_embeddings_batch([query_text])[0]
    
    # 2. Build prefetch clauses
    prefetch = [
        models.Prefetch(
            query=query_dense,
            using="dense",
            filter=filter_conditions,
            limit=dense_prefetch_limit
        ),
        models.Prefetch(
            query=query_sparse,
            using="sparse",
            filter=filter_conditions,
            limit=sparse_prefetch_limit
        )
    ]
    
    # 3. Execute fused query using RRF
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit
    )
    return res.points

def display_results(query_title: str, results):
    print(f"==================================================")
    print(f"🔎 QUERY: {query_title}")
    print(f"Returned {len(results)} ranked matches:")
    print(f"==================================================")
    for idx, hit in enumerate(results, 1):
        p = hit.payload
        mod = p.get("modality", "unknown").upper()
        chunk_id = p.get("chunk_id")
        page = p.get("page_start")
        toc = " > ".join(p.get("toc_path", []))
        score = hit.score
        
        print(f"\\n[{idx}] Score: {score:.4f} | Modality: {mod} | Page {page} | ID: {chunk_id}")
        print(f"    TOC: {toc}")
        if mod == "FIGURE":
            print(f"    Visual Category: {p.get('visual_category')}")
            print(f"    Metrics: {p.get('quantitative_metrics', [])[:2]}")
        snippet = p.get('text', '').replace('\\n', ' ')[:160]
        print(f"    Text: {snippet}...")
    print(f"==================================================\\n")
    """)

    # Cell 11: Validation Test Queries
    md("## 11. Run Representative Test Queries")
    code("""# Test 1: Exact Numerical Query (Tests Sparse BM25 advantage)
results_num = search_hybrid(
    query_text="Apple Card transaction provision for lending-related commitments of $2.2 billion",
    limit=3
)
display_results("Exact Financial & Footnote Lookup (Apple Card)", results_num)

# Test 2: Visual Chart Trend Query (Tests Figure Modality)
results_chart = search_hybrid(
    query_text="20-year net income and ROTCE progression from 2005 to 2025",
    limit=3
)
display_results("20-Year Trend Progression (Vector Chart)", results_chart)

# Test 3: Metadata Pre-Filtered Query (Filtered strictly to Table chunks)
table_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="modality",
            match=models.MatchValue(value="table")
        ),
        models.FieldCondition(
            key="page_start",
            range=models.Range(gte=1, lte=10)
        )
    ]
)

results_filtered = search_hybrid(
    query_text="Total net revenue and pre-provision profit financial highlights",
    filter_conditions=table_filter,
    limit=3
)
display_results("Pre-Filtered Table Query (Pages 1-10)", results_filtered)
    """)

    # Cell 12: Two-Stage RAG with BAAI/bge-reranker-v2-m3
    md("""
---
## 12. Two-Stage RAG: Cross-Encoder Reranking with BAAI/bge-reranker-v2-m3

### Why Add a Cross-Encoder Reranker?
1. **Bi-Encoder / Hybrid Limitation:** Dense embeddings map queries and documents into separate vector spaces independently (bi-encoder). While fast (sub-10ms via HNSW), the model cannot observe token-level interactions between the specific query tokens and document tokens until vector dot-product.
2. **Cross-Encoder Advantage:** `BAAI/bge-reranker-v2-m3` feeds the concatenated `[CLS] Query [SEP] Document [EOS]` into the transformer layers simultaneously. All query tokens attend to all document tokens across all self-attention heads, unlocking state-of-the-art precision for subtle financial qualifiers, footnotes, and accounting terms.
3. **The Two-Stage Architecture:**
   - **Stage 1 (High Recall):** Qdrant retrieves top 20 candidate chunks in ~15ms using Hybrid Dense + Sparse BM25 + RRF.
   - **Stage 2 (High Precision):** `bge-reranker-v2-m3` scores all 20 candidate chunks on GPU/CPU in ~30ms, re-ordering them to place the exact audit-grade evidence at ranks 1–5.
    """)

    code("""import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
reranker_model_name = "BAAI/bge-reranker-v2-m3"

print(f"[*] Initializing Cross-Encoder '{reranker_model_name}' on device: {device}...")
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name).to(device)
reranker_model.eval()

print(f"[+] '{reranker_model_name}' ready for two-stage inference!")
    """)

    code("""def search_and_rerank(query_text: str, candidate_pool_size: int = 20, top_k: int = 5):
    \"\"\"Execute two-stage retrieval: Qdrant Hybrid RRF candidate fetch -> BGE Cross-Encoder reranking.\"\"\"
    # Stage 1: Over-retrieve candidates from Qdrant
    candidates = search_hybrid(query_text=query_text, limit=candidate_pool_size)
    if not candidates:
        print("[!] No candidates found in Qdrant.")
        return []

    # Stage 2: Prepare pairs for BGE Cross-Encoder
    pairs = []
    for hit in candidates:
        p = hit.payload
        toc_str = " > ".join(p.get("toc_path", []))
        text = p.get("text", "")
        context = f"{toc_str}\\n{text}" if toc_str else text
        pairs.append([query_text, context[:1500]])

    with torch.no_grad():
        inputs = reranker_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(device)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

    # Combine candidates with cross-encoder scores and sort
    reranked = list(zip(candidates, scores))
    reranked.sort(key=lambda x: x[1], reverse=True)

    print(f"==================================================")
    print(f"🎯 TWO-STAGE RETRIEVAL & RERANKING")
    print(f"Query: '{query_text}'")
    print(f"Stage 1 (Qdrant RRF Pool): {len(candidates)} candidates")
    print(f"Stage 2 (BGE Reranked Top {top_k}):")
    print(f"==================================================")
    for idx, (hit, rerank_score) in enumerate(reranked[:top_k], 1):
        p = hit.payload
        mod = p.get("modality", "unknown").upper()
        cid = p.get("chunk_id")
        page = p.get("page_start")
        toc = " > ".join(p.get("toc_path", []))
        
        print(f"\\n[{idx}] BGE Score: {rerank_score:+.2f} (Qdrant RRF: {hit.score:.4f}) | {mod} | p.{page} | {cid}")
        print(f"    TOC: {toc}")
        if mod == "FIGURE" and p.get("quantitative_metrics"):
            print(f"    Metrics: {p.get('quantitative_metrics')[:2]}")
        snippet = p.get("text", "").replace("\\n", " ")[:160]
        print(f"    Text: {snippet}...")
    print(f"==================================================\\n")
    return [hit for hit, _ in reranked[:top_k]]
    """)

    code("""# Demonstration: Complex Financial Metric Reranking
query = "what was JPMorgan's net income and return on tangible common equity in 2025?"
top_reranked = search_and_rerank(query, candidate_pool_size=20, top_k=5)
    """)

    # Build notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"[OK] Generated notebook at {notebook_path.resolve()} with {len(cells)} cells.")

if __name__ == "__main__":
    create_notebook()
