"""
Batch Evaluation Runner with MLflow Experiment Tracking
========================================================
Runs statistical IR evaluation across all (or a filtered tier) of ground truth
questions, logs every metric to MLflow, and writes a markdown scorecard.

Usage
-----
# Run on easy questions (default)
uv run evaluation/run_evaluation.py

# Specify tier and a named experiment
uv run evaluation/run_evaluation.py --tier easy --experiment "Experiment-1-Hybrid-BGE"

# Run on all tiers
uv run evaluation/run_evaluation.py --tier all --experiment "Baseline-All-Tiers"

# Specify K values and retrieval limit
uv run evaluation/run_evaluation.py --tier easy --k-values 1 3 5 10 --limit 10

# Resume from a previous checkpoint
uv run evaluation/run_evaluation.py --tier easy --resume

MLflow UI
---------
mlflow ui --backend-store-uri ./mlruns
Then visit http://localhost:5000
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import evaluate_query, aggregate_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth_dataset.json"
CHECKPOINTS_DIR = EVAL_DIR / "checkpoints"
SCORECARDS_DIR = EVAL_DIR / "scorecards"
MLRUNS_DIR = REPO_ROOT / "mlruns"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION_NAME = "jpmc_annual_report_2025"
DENSE_MODEL_NAME = "qwen3-embedding:4b"
EMBEDDING_DIM = 1024

DEFAULT_K_VALUES = [1, 3, 5, 10]
DEFAULT_LIMIT = 10


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _get_dense_embedding(query: str) -> Optional[List[float]]:
    """Call Ollama embed endpoint to get a dense embedding."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": DENSE_MODEL_NAME, "input": query, "dimensions": EMBEDDING_DIM},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("embeddings", [[]])[0]
    except Exception as e:
        print(f"    [!] Dense embedding error: {e}")
        return None


def _get_sparse_embedding(query: str):
    """Get BM25 sparse vector via FastEmbed."""
    try:
        from fastembed import SparseTextEmbedding
        model = SparseTextEmbedding(model_name="Qdrant/bm25")
        vec = list(model.embed([query]))[0]
        return vec
    except Exception as e:
        print(f"    [!] Sparse embedding error: {e}")
        return None


def retrieve_chunk_ids(query: str, limit: int = 10) -> List[str]:
    """
    Hybrid Qdrant retrieval: Dense (Qwen) + Sparse (BM25) -> RRF fusion.
    No reranking - fast mode.
    """
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=QDRANT_URL)
    candidate_limit = max(limit * 3, 25)

    dense_vec = _get_dense_embedding(query)
    if dense_vec is None:
        return []

    sparse_vec_raw = _get_sparse_embedding(query)
    if sparse_vec_raw is None:
        return []

    sparse_vec = models.SparseVector(
        indices=sparse_vec_raw.indices.tolist(),
        values=sparse_vec_raw.values.tolist(),
    )

    try:
        prefetch = [
            models.Prefetch(query=dense_vec, using="dense", limit=candidate_limit),
            models.Prefetch(query=sparse_vec, using="sparse", limit=candidate_limit),
        ]
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [pt.payload.get("chunk_id", pt.id) for pt in result.points]
    except Exception as e:
        print(f"    [!] Qdrant query error: {e}")
        return []


def retrieve_chunk_ids_with_rerank(query: str, limit: int = 10) -> List[str]:
    """
    Full two-stage retrieval matching agent/tools.py exactly:
    Stage 1: Hybrid RRF -> Stage 2: BGE cross-encoder reranking.
    """
    from qdrant_client import QdrantClient, models
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    client = QdrantClient(url=QDRANT_URL)
    candidate_limit = max(limit * 3, 25)

    dense_vec = _get_dense_embedding(query)
    if dense_vec is None:
        return []

    sparse_vec_raw = _get_sparse_embedding(query)
    if sparse_vec_raw is None:
        return []

    sparse_vec = models.SparseVector(
        indices=sparse_vec_raw.indices.tolist(),
        values=sparse_vec_raw.values.tolist(),
    )

    try:
        prefetch = [
            models.Prefetch(query=dense_vec, using="dense", limit=candidate_limit),
            models.Prefetch(query=sparse_vec, using="sparse", limit=candidate_limit),
        ]
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidate_limit,
            with_payload=True,
        )
        candidates = result.points
    except Exception as e:
        print(f"    [!] Qdrant query error: {e}")
        return []

    if not candidates:
        return []

    # BGE cross-encoder reranking
    try:
        model_name = "BAAI/bge-reranker-v2-m3"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()

        pairs = []
        for pt in candidates:
            p = pt.payload or {}
            toc_str = " > ".join(p.get("toc_path") or [])
            text = p.get("text", "")
            doc_ctx = f"{toc_str}\n{text}" if toc_str else text
            pairs.append([query, doc_ctx[:1500]])

        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            scores = model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [pt.payload.get("chunk_id", pt.id) for pt, _ in scored[:limit]]
    except Exception as e:
        print(f"    [!] BGE reranker error ({e}), falling back to RRF order.")
        return [pt.payload.get("chunk_id", pt.id) for pt in candidates[:limit]]


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def load_questions(tier: str) -> List[Dict[str, Any]]:
    """Load questions from ground truth, filtering by difficulty tier."""
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if tier == "all":
        print(f"[i] Loaded all {len(questions)} questions.")
        return questions

    filtered = [q for q in questions if q.get("difficulty", "").lower() == tier.lower()]
    print(f"[i] Loaded {len(filtered)} '{tier}' questions from {len(questions)} total.")
    return filtered


def get_ground_truth_chunk_ids(question: Dict[str, Any]) -> List[str]:
    """Extract list of ground truth chunk_id strings from a question entry."""
    gt_chunks = question.get("ground_truth_chunk_ids", [])
    if not gt_chunks:
        return []
    if isinstance(gt_chunks[0], dict):
        return [c["chunk_id"] for c in gt_chunks]
    return [str(c) for c in gt_chunks]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(experiment_name: str, tier: str) -> Dict[str, Any]:
    """Load a previously saved checkpoint, returns {} if not found."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = experiment_name.replace("/", "_").replace(" ", "_")
    checkpoint_file = CHECKPOINTS_DIR / f"{safe_name}_{tier}_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            cp = json.load(f)
        print(f"[i] Resuming from checkpoint: {checkpoint_file.name} ({len(cp.get('completed', []))} done)")
        return cp
    return {"completed": [], "results": []}


def save_checkpoint(experiment_name: str, tier: str, checkpoint: Dict[str, Any]):
    """Persist checkpoint to disk."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = experiment_name.replace("/", "_").replace(" ", "_")
    checkpoint_file = CHECKPOINTS_DIR / f"{safe_name}_{tier}_checkpoint.json"
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def clear_checkpoint(experiment_name: str, tier: str):
    """Remove checkpoint file after successful completion."""
    safe_name = experiment_name.replace("/", "_").replace(" ", "_")
    checkpoint_file = CHECKPOINTS_DIR / f"{safe_name}_{tier}_checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"[i] Checkpoint cleared: {checkpoint_file.name}")


# ---------------------------------------------------------------------------
# Scorecard generation
# ---------------------------------------------------------------------------

def generate_scorecard(
    experiment_name: str,
    tier: str,
    run_config: Dict[str, Any],
    per_question_results: List[Dict[str, Any]],
    aggregated: Dict[str, float],
    mlflow_run_id: Optional[str],
    elapsed_sec: float,
) -> Path:
    """Write a detailed markdown scorecard to evaluation/scorecards/."""
    SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = experiment_name.replace("/", "_").replace(" ", "_")
    scorecard_path = SCORECARDS_DIR / f"{safe_name}_{tier}_{ts}.md"

    k_vals = run_config.get("k_values", DEFAULT_K_VALUES)
    primary_k = max(k_vals)

    lines = [
        "# Evaluation Scorecard",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Experiment** | `{experiment_name}` |",
        f"| **Tier** | `{tier}` |",
        f"| **Questions** | {len(per_question_results)} |",
        f"| **Retrieval Limit** | {run_config.get('limit', DEFAULT_LIMIT)} |",
        f"| **K values** | {k_vals} |",
        f"| **Rerank** | {'BGE-reranker-v2-m3' if run_config.get('rerank') else 'RRF only'} |",
        f"| **Timestamp** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| **Elapsed** | {elapsed_sec:.1f}s ({elapsed_sec/max(len(per_question_results),1):.1f}s/q) |",
    ]
    if mlflow_run_id:
        lines.append(f"| **MLflow Run ID** | `{mlflow_run_id}` |")
    lines.append("")

    # Aggregate metrics table
    lines += [
        "## Aggregate Metrics (Macro-Average)",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for metric_key in sorted(aggregated.keys()):
        if metric_key in ("total_queries", "total_elapsed_sec", "avg_latency_per_question_sec"):
            continue
        val = aggregated[metric_key]
        pct = f"{val*100:.1f}%"
        lines.append(f"| `{metric_key}` | {pct} |")
    lines.append("")

    # Key headline metrics
    hit = aggregated.get(f"mean_hit_rate@{primary_k}", 0)
    recall = aggregated.get(f"mean_recall@{primary_k}", 0)
    precision = aggregated.get(f"mean_precision@{primary_k}", 0)
    ndcg = aggregated.get(f"mean_ndcg@{primary_k}", 0)
    mrr = aggregated.get(f"mean_mrr@{primary_k}", 0)
    map_score = aggregated.get(f"mean_map@{primary_k}", 0)

    lines += [
        f"## Key Headline Metrics (K={primary_k})",
        "",
        f"| Hit Rate@{primary_k} | Recall@{primary_k} | Precision@{primary_k} | NDCG@{primary_k} | MRR@{primary_k} | MAP@{primary_k} |",
        f"|---:|---:|---:|---:|---:|---:|",
        f"| {hit*100:.1f}% | {recall*100:.1f}% | {precision*100:.1f}% | {ndcg*100:.1f}% | {mrr*100:.1f}% | {map_score*100:.1f}% |",
        "",
    ]

    # Per-question breakdown
    lines += [
        "## Per-Question Results",
        "",
        f"| ID | Question | GT Chunks | Retrieved | Hit@{primary_k} | Recall@{primary_k} | MAP@{primary_k} | NDCG@{primary_k} | Status |",
        f"|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in per_question_results:
        q_id = r.get("id", "?")
        question_short = (r.get("question", "")[:55] + "...") if len(r.get("question", "")) > 55 else r.get("question", "")
        gt_n = r.get("gt_count", "?")
        ret_n = len(r.get("retrieved_ids", []))
        hit_q = f"{r.get(f'hit_rate@{primary_k}', 0)*100:.0f}%"
        rec_q = f"{r.get(f'recall@{primary_k}', 0)*100:.0f}%"
        map_q = f"{r.get(f'map@{primary_k}', 0)*100:.0f}%"
        ndcg_q = f"{r.get(f'ndcg@{primary_k}', 0)*100:.0f}%"
        status = "ERROR" if r.get("error") else "OK"
        lines.append(f"| {q_id} | {question_short} | {gt_n} | {ret_n} | {hit_q} | {rec_q} | {map_q} | {ndcg_q} | {status} |")

    lines += [
        "",
        "---",
        f"*Generated by `evaluation/run_evaluation.py` on {datetime.now().isoformat()}*",
    ]

    scorecard_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Scorecard written -> {scorecard_path}")
    return scorecard_path


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def setup_mlflow(experiment_name: str):
    """Initialize MLflow with a local tracking URI and set/create the experiment."""
    try:
        import mlflow

        tracking_uri = MLRUNS_DIR.as_uri()
        mlflow.set_tracking_uri(tracking_uri)

        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            mlflow.create_experiment(
                experiment_name,
                tags={
                    "project": "Multi-modal-RAG",
                    "document": "JPMC 2025 Annual Report",
                    "collection": COLLECTION_NAME,
                },
            )
        mlflow.set_experiment(experiment_name)
        print(f"[i] MLflow tracking URI : {tracking_uri}")
        print(f"[i] MLflow experiment   : {experiment_name}")
        return mlflow
    except ImportError:
        print("[!] mlflow not installed. Run: pip install mlflow")
        return None


def log_question_to_mlflow(mlflow, question_result: Dict[str, Any]):
    """Log per-question metrics as a nested MLflow child run."""
    try:
        with mlflow.start_run(run_name=question_result["id"], nested=True):
            mlflow.set_tags({
                "question_id": question_result["id"],
                "difficulty": question_result.get("difficulty", ""),
                "category": question_result.get("category", ""),
                "source_modality": question_result.get("source_modality", ""),
                "has_error": str(bool(question_result.get("error"))),
            })
            metrics_to_log = {
                k: v for k, v in question_result.items()
                if isinstance(v, (int, float)) and k not in ("gt_count",)
            }
            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log)

            # Log retrieved vs GT chunk IDs as a JSON artifact
            artifact_data = {
                "question": question_result.get("question"),
                "ground_truth_chunk_ids": question_result.get("ground_truth_chunk_ids", []),
                "retrieved_chunk_ids": question_result.get("retrieved_ids", []),
                "gt_count": question_result.get("gt_count"),
                "error": question_result.get("error"),
            }
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as tmp:
                json.dump(artifact_data, tmp, indent=2)
                tmp_path = tmp.name
            mlflow.log_artifact(tmp_path, artifact_path="question_details")
            os.unlink(tmp_path)
    except Exception as e:
        print(f"    [!] MLflow child run logging error: {e}")


# ---------------------------------------------------------------------------
# Null context manager (when mlflow is unavailable)
# ---------------------------------------------------------------------------

class _null_context:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args):
    start_time = time.time()

    experiment_name = (
        args.experiment
        or f"RAG-Eval-{args.tier.capitalize()}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )

    questions = load_questions(args.tier)
    if not questions:
        print(f"[!] No questions found for tier='{args.tier}'. Exiting.")
        sys.exit(1)

    checkpoint = (
        load_checkpoint(experiment_name, args.tier)
        if args.resume
        else {"completed": [], "results": []}
    )
    completed_ids = set(checkpoint.get("completed", []))
    per_question_results = checkpoint.get("results", [])

    mlflow = setup_mlflow(experiment_name)
    mlflow_run_id = None

    run_config = {
        "tier": args.tier,
        "k_values": args.k_values,
        "limit": args.limit,
        "rerank": args.rerank,
        "collection": COLLECTION_NAME,
        "dense_model": DENSE_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "total_questions": len(questions),
    }

    retrieve_fn = retrieve_chunk_ids_with_rerank if args.rerank else retrieve_chunk_ids

    ctx = mlflow.start_run(run_name=f"{experiment_name}") if mlflow else _null_context()
    with ctx as parent_run:
        if mlflow and parent_run:
            mlflow_run_id = parent_run.info.run_id
            mlflow.log_params({
                "tier": args.tier,
                "k_values": str(args.k_values),
                "retrieval_limit": args.limit,
                "rerank": args.rerank,
                "collection": COLLECTION_NAME,
                "dense_model": DENSE_MODEL_NAME,
                "embedding_dim": EMBEDDING_DIM,
            })
            mlflow.set_tags({
                "project": "Multi-modal-RAG",
                "document": "JPMC 2025 Annual Report",
                "retrieval_stack": "Qwen+BM25+RRF" + ("+BGE-reranker" if args.rerank else ""),
            })

        remaining = [q for q in questions if q["id"] not in completed_ids]
        total = len(questions)
        done_count = len(completed_ids)

        print(f"\n{'='*65}")
        print(f"  Experiment : {experiment_name}")
        print(f"  Tier       : {args.tier}  |  Questions: {total}  |  Remaining: {len(remaining)}")
        print(f"  K values   : {args.k_values}  |  Limit: {args.limit}  |  Rerank: {args.rerank}")
        print(f"{'='*65}\n")

        for q in remaining:
            done_count += 1
            q_id = q["id"]
            question_text = q.get("question", "")
            gt_ids = get_ground_truth_chunk_ids(q)

            print(f"  [{done_count:02d}/{total}] {q_id}: {question_text[:70]}...")
            t0 = time.time()

            result = {
                "id": q_id,
                "question": question_text,
                "difficulty": q.get("difficulty", ""),
                "category": q.get("category", ""),
                "source_modality": q.get("source_modality", ""),
                "ground_truth_chunk_ids": gt_ids,
                "gt_count": len(gt_ids),
                "retrieved_ids": [],
                "error": None,
                "latency_sec": 0.0,
            }

            try:
                retrieved_ids = retrieve_fn(question_text, limit=args.limit)
                result["retrieved_ids"] = retrieved_ids

                ir_metrics = evaluate_query(retrieved_ids, gt_ids, k_values=args.k_values)
                result.update(ir_metrics)

                elapsed_q = time.time() - t0
                result["latency_sec"] = round(elapsed_q, 2)

                hit_k = max(args.k_values)
                hit_val = ir_metrics.get(f"hit_rate@{hit_k}", 0)
                rec_val = ir_metrics.get(f"recall@{hit_k}", 0)
                print(
                    f"         Hit@{hit_k}={hit_val:.0%}  Recall@{hit_k}={rec_val:.0%}  "
                    f"Retrieved={len(retrieved_ids)}  GT={len(gt_ids)}  ({elapsed_q:.1f}s)"
                )

            except Exception as e:
                result["error"] = str(e)
                traceback.print_exc()
                print(f"         [!] ERROR: {e}")

            per_question_results.append(result)

            if mlflow:
                log_question_to_mlflow(mlflow, result)

            checkpoint["completed"].append(q_id)
            checkpoint["results"] = per_question_results
            save_checkpoint(experiment_name, args.tier, checkpoint)

        # --- Aggregate ---
        valid_results = [r for r in per_question_results if not r.get("error")]
        metric_dicts = [
            {k: v for k, v in r.items() if isinstance(v, float) and k != "latency_sec"}
            for r in valid_results
        ]
        aggregated = aggregate_metrics(metric_dicts)

        elapsed_total = time.time() - start_time
        aggregated["total_elapsed_sec"] = round(elapsed_total, 2)
        aggregated["avg_latency_per_question_sec"] = round(
            sum(r.get("latency_sec", 0) for r in per_question_results) / max(len(per_question_results), 1), 2
        )

        print(f"\n{'='*65}")
        print(f"  RESULTS SUMMARY  ({len(valid_results)}/{total} successful)")
        print(f"{'='*65}")
        k_report = max(args.k_values)
        for key in [
            f"mean_hit_rate@{k_report}", f"mean_recall@{k_report}",
            f"mean_precision@{k_report}", f"mean_ndcg@{k_report}",
            f"mean_mrr@{k_report}", f"mean_map@{k_report}",
        ]:
            val = aggregated.get(key, 0)
            print(f"    {key:<30} {val*100:.1f}%")
        print(f"{'='*65}\n")

        if mlflow:
            try:
                mlflow.log_metrics({k: v for k, v in aggregated.items() if isinstance(v, float)})
                mlflow.log_metric("total_questions", total)
                mlflow.log_metric("successful_questions", len(valid_results))
                mlflow.log_metric("failed_questions", total - len(valid_results))
            except Exception as e:
                print(f"[!] MLflow aggregate logging error: {e}")

        scorecard_path = generate_scorecard(
            experiment_name=experiment_name,
            tier=args.tier,
            run_config=run_config,
            per_question_results=per_question_results,
            aggregated=aggregated,
            mlflow_run_id=mlflow_run_id,
            elapsed_sec=elapsed_total,
        )

        if mlflow:
            try:
                mlflow.log_artifact(str(scorecard_path), artifact_path="scorecards")
                print(f"[OK] Scorecard uploaded to MLflow run {mlflow_run_id}")
            except Exception as e:
                print(f"[!] MLflow scorecard artifact error: {e}")

        clear_checkpoint(experiment_name, args.tier)

        print(f"\n[OK] Evaluation complete in {elapsed_total:.1f}s")
        if mlflow_run_id:
            print(f"[i] View results: mlflow ui --backend-store-uri {MLRUNS_DIR}")
            print(f"[i] Run ID: {mlflow_run_id}")

    return aggregated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch RAG evaluation runner with MLflow tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier",
        choices=["easy", "medium", "complex", "all"],
        default="easy",
        help="Question difficulty tier to evaluate (default: easy).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        metavar="K",
        help=f"Cutoff K values for metric calculation (default: {DEFAULT_K_VALUES}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of top chunks to retrieve per query (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Enable BGE cross-encoder reranking (slower, matches live agent exactly).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from the last checkpoint for this experiment+tier combination.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
