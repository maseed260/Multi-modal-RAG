"""
Enterprise Multi-Modal RAG: Hybrid Dense + Sparse Embedding & Ingestion Pipeline
Stores 1,452 multimodal chunks from JPMorgan Chase & Co. 2025 Annual Report into Qdrant.

Architecture:
- Dense: Qwen3-Embedding (4B parameters) via Ollama with 1024d MRL truncation (Cosine distance)
- Sparse: FastEmbed BM25 (Qdrant/bm25) for lexical and numerical term matching
- Vector Database: Qdrant with custom HNSW parameters (m=16, ef_construct=100)
- Payload Indexing: Fast pre-filtered queries on modality, page ranges, and TOC sections
- Hybrid Search: Multi-vector prefetch with Reciprocal Rank Fusion (RRF)
"""

import sys
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any

import requests
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding


def main():
    qdrant_url = "http://localhost:6333"
    ollama_url = "http://localhost:11434"
    collection_name = "jpmc_annual_report_2025"
    dense_model_name = "qwen3-embedding:4b"
    embedding_dim = 1024
    batch_size = 32

    # 1. Connect to Qdrant
    print(f"[*] Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url)
    collections = client.get_collections()
    print(f"[+] Qdrant connection established. Existing collections: {[c.name for c in collections.collections]}")

    # 2. Check Ollama
    print(f"[*] Checking Ollama at {ollama_url} for '{dense_model_name}'...")
    try:
        tags_resp = requests.get(f"{ollama_url}/api/tags").json()
        models_list = [m["name"] for m in tags_resp.get("models", [])]
        if not any(dense_model_name in m for m in models_list):
            print(f"[!] Warning: '{dense_model_name}' not found in Ollama models: {models_list}")
        else:
            print(f"[+] Ollama model '{dense_model_name}' verified.")
    except Exception as e:
        print(f"[!] Error contacting Ollama: {e}")
        sys.exit(1)

    # 3. Load chunks
    chunks_path = Path("chunking/output/unified_chunks.jsonl")
    if not chunks_path.exists():
        chunks_path = Path("../chunking/output/unified_chunks.jsonl")
    
    print(f"[*] Loading unified chunks from: {chunks_path.resolve()}")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line.strip()))
    
    print(f"[+] Loaded {len(chunks):,} chunks.")

    # 4. Initialize FastEmbed BM25
    print("[*] Initializing FastEmbed BM25 sparse model (Qdrant/bm25)...")
    bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    # 5. Create or Recreate Collection
    if client.collection_exists(collection_name):
        print(f"[*] Collection '{collection_name}' already exists. Recreating fresh...")
        client.delete_collection(collection_name)

    print(f"[*] Creating collection '{collection_name}' with 1024d HNSW Dense + BM25 Sparse...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                    on_disk=False,
                ),
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )

    # 6. Create Payload Indexes for fast pre-filtering
    index_fields = [
        ("modality", models.PayloadSchemaType.KEYWORD),
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("page_start", models.PayloadSchemaType.INTEGER),
        ("page_end", models.PayloadSchemaType.INTEGER),
        ("toc_section", models.PayloadSchemaType.KEYWORD),
        ("toc_level", models.PayloadSchemaType.INTEGER),
        ("visual_category", models.PayloadSchemaType.KEYWORD),
        ("chunk_id", models.PayloadSchemaType.KEYWORD),
    ]
    for field_name, schema_type in index_fields:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )
    print(f"[+] Created {len(index_fields)} payload indexes.")

    # Helper functions
    def get_dense_batch(texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            f"{ollama_url}/api/embed",
            json={"model": dense_model_name, "input": texts, "dimensions": embedding_dim},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("embeddings", [])

    def get_sparse_batch(texts: List[str]) -> List[models.SparseVector]:
        sparse_gen = bm25_model.embed(texts)
        return [
            models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
            for s in sparse_gen
        ]

    # 7. Ingestion loop
    print(f"[*] Ingesting {len(chunks):,} chunks in batches of {batch_size}...")
    t0 = time.time()
    total_upserted = 0

    for i in tqdm(range(0, len(chunks), batch_size), desc="Ingesting"):
        batch = chunks[i : i + batch_size]
        batch_texts = [c["text"] for c in batch]

        dense_vecs = get_dense_batch(batch_texts)
        sparse_vecs = get_sparse_batch(batch_texts)

        points = []
        for chunk, d_vec, s_vec in zip(batch, dense_vecs, sparse_vecs):
            chunk_id = chunk["chunk_id"]
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

            prov = chunk.get("provenance", {})
            toc = chunk.get("toc", {})
            nav = chunk.get("navigation", {})
            mm = chunk.get("multimodal") or {}

            payload = {
                "chunk_id": chunk_id,
                "modality": chunk.get("modality"),
                "token_count": chunk.get("token_count", 0),
                "char_count": chunk.get("char_count", 0),
                "doc_id": prov.get("doc_id", "jpmc_annualreport-2025"),
                "doc_title": prov.get("doc_title", ""),
                "page_start": prov.get("page_start", 0),
                "page_end": prov.get("page_end", 0),
                "page_range": prov.get("page_range", []),
                "citation": prov.get("citation", ""),
                "toc_section": toc.get("toc_section", ""),
                "toc_parent": toc.get("toc_parent", ""),
                "toc_path": toc.get("toc_path", []),
                "toc_level": toc.get("toc_level", 0),
                "prev_chunk_id": nav.get("prev_chunk_id"),
                "next_chunk_id": nav.get("next_chunk_id"),
                "section_sibling_ids": nav.get("section_sibling_ids", []),
                "image_path": mm.get("image_path"),
                "visual_category": mm.get("visual_category"),
                "quantitative_metrics": mm.get("quantitative_metrics", []),
                "inlined_footnotes": chunk.get("inlined_footnotes", []),
                "text": chunk.get("text", ""),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"dense": d_vec, "sparse": s_vec},
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)
        total_upserted += len(points)

    dt = time.time() - t0
    print(f"\n[+] Ingestion complete! {total_upserted:,} points in {dt:.2f}s ({total_upserted/dt:.1f} pts/sec).")

    # 8. Collection stats
    info = client.get_collection(collection_name)
    print(f"[+] Collection Status: {info.status.value}")
    print(f"[+] Total Points: {info.points_count:,}")
    print(f"[+] Indexed Vectors: {info.indexed_vectors_count:,}")
    print(f"[+] Dashboard URL: {qdrant_url}/dashboard")


if __name__ == "__main__":
    main()
