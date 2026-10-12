"""
Statistical & Information Retrieval (IR) Evaluation Metrics for RAG Pipelines.

Provides deterministic, zero-LLM-cost metrics for evaluating vector & hybrid retrieval
against ground truth reference chunks:
- Hit Rate@K (Did the top-K contain at least one valid reference?)
- Recall@K (What proportion of all valid references were retrieved in top-K?)
- Precision@K (What proportion of top-K chunks were valid references?)
- Reciprocal Rank / MRR (Reciprocal rank of the first relevant chunk)
- Average Precision / MAP (Precision weighted by rank of every relevant chunk)
- R-Precision (Precision at K = len(ground_truth))
- NDCG@K (Normalized Discounted Cumulative Gain)
"""

import math
from typing import List, Set, Dict, Any, Union


def compute_hit_rate(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k: int = 5) -> float:
    """Returns 1.0 if at least one ground truth chunk is in top-K, else 0.0."""
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(cid in gt_set for cid in top_k) else 0.0


def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k: int = 5) -> float:
    """Returns fraction of ground truth chunks present in top-K: |retrieved[:k] ∩ GT| / |GT|."""
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k.intersection(gt_set))
    return hits / len(gt_set)


def compute_precision_at_k(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k: int = 5) -> float:
    """Returns fraction of top-K chunks that are relevant: |retrieved[:k] ∩ GT| / K."""
    if k <= 0:
        return 0.0
    gt_set = set(ground_truth_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in gt_set)
    return hits / k


def compute_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k: int = 10) -> float:
    """Returns 1 / rank of the first relevant chunk in top-K, or 0.0 if none found."""
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gt_set:
            return 1.0 / rank
    return 0.0


def compute_average_precision(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k: int = 10) -> float:
    """
    Computes Average Precision (AP) at cutoff K:
    AP@K = (1 / |GT|) * sum_{i=1}^K [Precision@i * is_relevant(i)]
    """
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gt_set:
            hits += 1
            sum_precisions += hits / rank
            
    return sum_precisions / len(gt_set)


def compute_r_precision(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]]) -> float:
    """Computes Precision at K where K equals the number of ground truth chunks (|GT|)."""
    gt_set = set(ground_truth_ids)
    r = len(gt_set)
    if r == 0:
        return 0.0
    return compute_precision_at_k(retrieved_ids, gt_set, k=r)


def compute_ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG) at K assuming binary relevance (1 or 0).
    DCG@K = sum_{i=1}^K [rel_i / log2(i + 1)]
    IDCG@K = sum_{i=1}^{min(|GT|, K)} [1 / log2(i + 1)]
    """
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    
    # DCG
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gt_set:
            dcg += 1.0 / math.log2(rank + 1)
            
    # IDCG (ideal ranking puts all relevant chunks first)
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_query(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """Calculates all retrieval metrics for a single query across multiple cutoffs."""
    gt_set = set(ground_truth_ids)
    metrics = {
        "gt_count": len(gt_set),
        "r_precision": compute_r_precision(retrieved_ids, gt_set)
    }
    
    for k in k_values:
        metrics[f"hit_rate@{k}"] = compute_hit_rate(retrieved_ids, gt_set, k=k)
        metrics[f"recall@{k}"] = compute_recall_at_k(retrieved_ids, gt_set, k=k)
        metrics[f"precision@{k}"] = compute_precision_at_k(retrieved_ids, gt_set, k=k)
        metrics[f"mrr@{k}"] = compute_reciprocal_rank(retrieved_ids, gt_set, k=k)
        metrics[f"map@{k}"] = compute_average_precision(retrieved_ids, gt_set, k=k)
        metrics[f"ndcg@{k}"] = compute_ndcg_at_k(retrieved_ids, gt_set, k=k)
        
    return metrics


def aggregate_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Computes Macro-Average across a list of per-query metric dictionaries.

    Gracefully skips any non-numeric fields (e.g. 'id', 'difficulty', 'category')
    that may be present alongside IR metric scores.
    """
    if not per_query_metrics:
        return {}

    # Exclude known non-metric string/list fields
    _skip = {"gt_count", "id", "question", "difficulty", "category",
             "source_modality", "error", "retrieved_ids", "ground_truth_chunk_ids"}
    keys = [k for k in per_query_metrics[0].keys() if k not in _skip]
    agg = {"total_queries": len(per_query_metrics)}

    for k in keys:
        # Only average entries where the value is actually numeric
        vals = [m[k] for m in per_query_metrics if k in m and isinstance(m[k], (int, float))]
        agg[f"mean_{k}"] = sum(vals) / len(vals) if vals else 0.0

    return agg
