import os
import requests
from typing import Optional, List, Union, Any, Dict
from langchain_core.tools import tool
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from agent.models import get_synthesis_llm

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION_NAME = "jpmc_annual_report_2025"
DENSE_MODEL_NAME = "qwen3-embedding:4b"
EMBEDDING_DIM = 1024

# Lazy clients
_qdrant_client = None
_bm25_model = None
_synthesis_llm = None


def normalize_text_input(val: Any) -> str:
    """Safely extracts plain string from str, list of content blocks (e.g. Studio UI), or dict."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", item))))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    if isinstance(val, dict):
        return str(val.get("text", val.get("content", str(val))))
    return str(val)


def get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def get_bm25() -> SparseTextEmbedding:
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25_model


def get_synthesis() -> any:
    global _synthesis_llm
    if _synthesis_llm is None:
        _synthesis_llm = get_synthesis_llm()
    return _synthesis_llm


@tool
def retrieve_chunks(
    query: Union[str, List[Any]],
    filter_modality: Optional[str] = None,
    limit: int = 10
) -> str:
    """Retrieve up to 10 relevant chunks from the JPMC 2025 Annual Report using hybrid search (Dense Qwen 1024d + Sparse BM25 + RRF).
    
    Use this tool to find information about financial metrics, executive strategy, accounting tables, risk management, and visual charts.
    You can call this tool multiple times with different query rewrites to assemble multi-angle evidence.
    
    Args:
        query: Specific search phrase or question rewrite (e.g. 'Apple Card transaction provision', '20-year net income progression', 'CET1 capital ratios').
        filter_modality: Optional filter to restrict results to a specific modality: 'text', 'table', or 'figure'.
        limit: Number of chunks to retrieve (default is 10).
    """
    query = normalize_text_input(query)
    client = get_qdrant()
    bm25 = get_bm25()

    # 1. Dense query vector via Ollama
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": DENSE_MODEL_NAME, "input": query, "dimensions": EMBEDDING_DIM},
            timeout=60,
        )
        resp.raise_for_status()
        dense_vec = resp.json().get("embeddings", [[]])[0]
    except Exception as e:
        return f"Error computing dense embedding for query: {e}"

    # 2. Sparse query vector via FastEmbed BM25
    try:
        sparse_gen = list(bm25.embed([query]))[0]
        sparse_vec = models.SparseVector(
            indices=sparse_gen.indices.tolist(),
            values=sparse_gen.values.tolist(),
        )
    except Exception as e:
        return f"Error computing BM25 sparse embedding for query: {e}"

    # 3. Payload filter if modality specified
    q_filter = None
    if filter_modality:
        mod_val = filter_modality.strip().lower()
        q_filter = models.Filter(
            must=[models.FieldCondition(key="modality", match=models.MatchValue(value=mod_val))]
        )

    # 4. Hybrid prefetch + RRF fusion
    try:
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                filter=q_filter,
                limit=limit * 3,
            ),
            models.Prefetch(
                query=sparse_vec,
                using="sparse",
                filter=q_filter,
                limit=limit * 3,
            ),
        ]

        search_res = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
    except Exception as e:
        return f"Error executing Qdrant hybrid query: {e}"

    if not search_res.points:
        return f"No matching chunks found in Qdrant for query: '{query}'"

    # Format output for agent consumption
    results = [f"=== Retrieved {len(search_res.points)} Chunks for Query: '{query}' ==="]
    for idx, point in enumerate(search_res.points, 1):
        p = point.payload or {}
        cid = p.get("chunk_id", "unknown")
        mod = (p.get("modality") or "unknown").upper()
        page = p.get("page_start", 0)
        citation = p.get("citation", f"Page {page}")
        toc_path = " > ".join(p.get("toc_path") or [])
        score = point.score

        chunk_header = f"[{idx}] {cid} | {mod} | p.{page}"
        if toc_path:
            # Keep deepest 2 levels of TOC hierarchy
            short_toc = " > ".join((p.get("toc_path") or [])[-2:])
            chunk_header += f" | TOC: {short_toc}"
        
        # Include visual metrics if figure
        if mod == "FIGURE" and p.get("quantitative_metrics"):
            metrics_str = "; ".join(p.get("quantitative_metrics")[:2])
            chunk_header += f" | Metrics: {metrics_str}"
        
        # Include inlined footnotes if present
        footnotes = p.get("inlined_footnotes") or []
        if footnotes:
            fn_str = " | ".join(f[:100] for f in footnotes[:2])
            chunk_header += f" | FN: {fn_str}"

        body = p.get("text", "").strip()
        # Clean and compact excerpt to keep tokens efficient across multiple rewrites
        clean_body = " ".join(body.split())
        if len(clean_body) > 160:
            body_preview = clean_body[:160] + "..."
        else:
            body_preview = clean_body

        results.append(f"{chunk_header}\n    Text: {body_preview}")

    return "\n".join(results)


@tool
def generate_grounded_answer(
    question: Union[str, List[Any]],
    retrieved_evidence: Union[str, List[Any]],
    synthesis_notes: Optional[str] = None
) -> str:
    """Generate a rigorous, audit-ready answer grounded strictly in the retrieved multimodal chunks.
    
    Call this tool after you have gathered all relevant chunks (from one or multiple retrieve_chunks calls).
    The tool uses a high-capacity reasoning model (Llama 3.3 70B / equivalent) to reconcile numbers, narrative,
    footnotes, and charts into an executive-ready synthesized answer with exact citations.
    
    Args:
        question: The user's original inquiry.
        retrieved_evidence: Combined text/markdown content of the retrieved chunks (or summary of key findings).
        synthesis_notes: Optional guidance or specific aspects to emphasize (e.g. 'reconcile table with letter', 'cite footnotes').
    """
    question = normalize_text_input(question)
    retrieved_evidence = normalize_text_input(retrieved_evidence)
    synthesis_llm = get_synthesis()

    system_prompt = (
        "You are an Elite Enterprise Financial AI Analyst synthesizing an authoritative response from the "
        "JPMorgan Chase & Co. 2025 Annual Report.\n\n"
        "Guidelines:\n"
        "1. GROUNDING: Base every assertion strictly on the provided retrieved evidence. Never hallucinate or assume.\n"
        "2. MULTI-MODAL RECONCILIATION: Reconcile financial figures from accounting tables with the narrative from executive letters and visual charts.\n"
        "3. AUDITABLE CITATIONS: Explicitly cite specific page numbers, TOC sections, or table names for every claim (e.g. '[JPMC 2025 Annual Report, p. 2]').\n"
        "4. FOOTNOTE INTEGRITY: When numbers carry footnote references (e.g. non-GAAP adjustments, Apple Card provisions, Visa share gains), explicitly explain them.\n"
        "5. STRUCTURE: Use executive bullet points, clear bold headings, and markdown tables where comparing figures across years."
    )

    user_prompt = f"USER QUESTION:\n{question}\n\nRETRIEVED EVIDENCE:\n{retrieved_evidence}"
    if synthesis_notes:
        user_prompt += f"\n\nSPECIAL ANALYST NOTES:\n{synthesis_notes}"

    try:
        response = synthesis_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        return response.content
    except Exception as e:
        return f"Error during grounded answer generation: {e}"
