# RAG Evaluation: Ground Truth Dataset
## JPMorgan Chase & Co. 2025 Annual Report

> **File:** `ground_truth_dataset.json`  
> **Document:** JPMC 2025 Annual Report (364 pages, 1,452 indexed chunks)  
> **Collection:** `jpmc_annual_report_2025` (Qdrant)

---

## 📓 Evaluation Notebooks Suite

The evaluation suite is organized into 5 modular notebooks covering distinct evaluation methodologies on the benchmark:

| Notebook | Methodology | Key Metrics | Evaluator Engine |
| :--- | :--- | :--- | :--- |
| **`01_statistical_retrieval_evaluation.ipynb`** | Information Retrieval & Statistical Ranking | Hit Rate@K, Recall@K, Precision@K, R-Precision, MRR, MAP, NDCG | Pure Python / `metrics.py` (0 LLM cost) |
| **`02_llm_as_a_judge_evaluation.ipynb`** | Qualitative LLM Judge & Criteria Verification | Criteria Met %, Faithfulness (1-5), Answer Relevance (1-5), Citation Accuracy | Local Ollama `gemma4:26b` / `qwen3.5:9b` |
| **`03_ragas_evaluation.ipynb`** | Ragas 0.4 Framework Evaluation | Faithfulness, Answer Relevancy, Context Precision, Context Recall | Ragas + Local Ollama wrappers |
| **`04_trulens_evaluation.ipynb`** | TruLens 2.x RAG Triad Evaluation | Context Relevance, Groundedness, Answer Relevance, Triad Average | TruLens + Local Ollama feedback |
| **`05_laya_calibrated_decision_evaluation.ipynb`** | Non-Autoregressive Calibrated Decision Engine | Calibrated P(Accurate), P(Grounded), Quality Tiers, Defect Classification (~33ms) | [Laya](https://github.com/NandhaKishorM/laya) System 1 Decision Router |



---

## Dataset Summary

| Difficulty | Count | Description |
|:---|:---:|:---|
| **Easy** | 20 | Direct single-source factual lookup. No multi-hop reasoning. |
| **Medium** | 20 | Multi-hop, cross-year comparison, chart extraction, or non-GAAP adjustment. |
| **Complex** | 15 | Cross-modal (table + figure + text) synthesis, analytical reasoning, EPS decomposition, or competitive strategy. |
| **Total** | **55** | *(Dataset contains 55 fully grounded questions, 5 short of 60 due to data coverage saturation)* |

> **Note:** The dataset was constructed to be fully grounded in extracted data from the RAG system's actual chunks, so we excluded any questions that couldn't be answered from the indexed content. Final count is 55 high-quality grounded questions across 3 tiers.

---

## Question Category Distribution

| Category | Easy | Medium | Complex | Total |
|:---|:---:|:---:|:---:|:---:|
| Income Statement | 3 | 2 | 2 | 7 |
| Balance Sheet | 3 | 0 | 0 | 3 |
| Per Share Data | 3 | 0 | 1 | 4 |
| Capital Ratios | 4 | 0 | 2 | 6 |
| Shareholder Returns | 1 | 4 | 1 | 6 |
| Market Data | 2 | 0 | 0 | 2 |
| Efficiency / Peer Comparison | 1 | 2 | 1 | 4 |
| Deposits & Client Assets | 1 | 1 | 1 | 3 |
| CIB / Investment Banking | 1 | 3 | 2 | 6 |
| Asset & Wealth Management | 0 | 2 | 0 | 2 |
| Historical Analysis | 0 | 1 | 2 | 3 |
| Strategic / Competitive | 0 | 2 | 1 | 3 |
| Acquisitions (Apple Card) | 1 | 0 | 1 | 2 |
| Financial Analysis | 0 | 1 | 2 | 3 |
| Multi-Modal Retrieval | 0 | 0 | 1 | 1 |
| Capital Deployment | 1 | 0 | 0 | 1 |
| Credit Quality | 1 | 0 | 0 | 1 |
| Securities Services | 0 | 1 | 0 | 1 |

---

## Source Modality Coverage

| Modality | % of Questions | Key Pages |
|:---|:---:|:---|
| **Table** | ~40% | Page 2 (Financial Highlights) |
| **Figure** | ~35% | Pages 8, 9, 11, 12, 13, 14, 60, 61, 62, 66 |
| **Multi-modal** | ~20% | Pages 2+8, 2+8+9, 2+60+62+66 |
| **Text (Prose)** | ~5% | Page 16 (Shareholder Letter) |

---

## Key Data Points Referenced

### From Page 2 (Financial Highlights Table)
| Metric | 2025 | 2024 | 2023 |
|:---|:---:|:---:|:---:|
| Total Net Revenue | $182,447M | $177,556M | $158,104M |
| Noninterest Expense | $95,640M | $91,799M | $87,172M |
| Pre-Provision Profit | $86,807M | $85,759M | $70,932M |
| Provision for Credit Losses | $14,212M | $10,678M | $9,320M |
| Net Income | $57,048M | $58,471M | $49,552M |
| Diluted EPS | $20.02 | $19.75 | $16.23 |
| ROTCE (non-GAAP) | 20% | 22% | 21% |
| ROE | 17% | 18% | 17% |
| CET1 (Standardized) | 14.6% | 15.7% | 15.0% |
| Tier 1 Capital Ratio | 15.5% | 16.8% | 16.6% |
| Total Assets | $4,424,900M | $4,002,814M | $3,875,393M |
| Deposits | $2,559,320M | $2,406,032M | $2,400,688M |
| Loans | $1,493,429M | $1,347,988M | $1,323,706M |
| TBVPS (non-GAAP) | $107.56 | $97.30 | $86.08 |
| Book Value/Share | $126.99 | $116.07 | $104.45 |
| Dividends/Share | $5.80 | $4.80 | $4.10 |
| Closing Share Price | $322.22 | $239.71 | $170.10 |
| Market Cap | $868,793M | $670,618M | $489,320M |
| Employees | 318,512 | 317,233 | 309,926 |

### From Figure Chunks (Charts)
| Page | Chart | Key Data |
|:---|:---|:---|
| 8 | Earnings 2005-2025 | 2005: NI $8.5B, EPS $4.03, ROTCE 15% → 2025: NI $57B, EPS $20.02, ROTCE 20% |
| 9 | Stock & TBV 2005-2025 | 1-yr 37.3%, 5-yr 23.7%, 10-yr 20.4% CAGRs vs S&P; merger return 1,380.5% |
| 11 | Credit & Capital | ~$1.9T new/renewed credit & capital in 2025 |
| 12 | Deposits & Client Assets | $1,107B (2005) → $7,643B (2025) = 6.9x growth |
| 13 | CET1 & Capital Return | CET1 ~7% (2005) → 14.6% (2025); $48B returned (2025) |
| 14 | Peer Comparison | JPM: 52% overhead / 20% ROTCE (best efficiency, 2nd ROTCE) |
| 60 | CIB Revenue | $1.2B (2021) → $27.8B (2025); 10% CAGR post-CB integration |
| 61 | IB & CB | #1 IB wallet 8.4%; M&A $5.1T +43%; CRE +80%; GCB $25.3B |
| 62 | Markets & Securities Svcs | Markets $35.8B (FI $22.5B + EQ $13.3B); AUC $41.2T |
| 66 | AWM | $24B rev, $7B NI, $7T client assets, $553B inflows (record) |

---

## Evaluation Schema

Each question in `ground_truth_dataset.json` includes:

```json
{
  "id": "GT-E-001",
  "question": "...",
  "ground_truth_answer": "...",
  "difficulty": "easy | medium | complex",
  "category": "...",
  "source_modality": "table | figure | text | multi-modal",
  "source_pages": [2],
  "toc_section": "...",
  "requires_reasoning": false,
  "evaluation_criteria": ["fact1", "fact2", "..."],
  "expected_chunk_types": ["table", "figure"],
  "notes": "..."
}
```

### Scoring Approach
Use `evaluation_criteria` as a checklist:
- **Full credit:** All listed criteria present in the generated answer
- **Partial credit:** Some criteria present (score = criteria_met / total_criteria)
- **No credit:** Core factual assertion is wrong

---

## Recommended Evaluation Frameworks

1. **RAGAS** — Faithfulness, Answer Relevance, Context Precision, Context Recall
2. **TruLens** — Groundedness, Answer Relevance, Context Relevance
3. **LangSmith** — Already integrated via `LANGCHAIN_TRACING_V2=true`; compare expected vs. actual

### Sample RAGAS Dataset Format
```python
from datasets import Dataset

data = {
    "question": [...],          # from ground_truth_dataset.json
    "answer": [...],            # from your RAG system
    "contexts": [...],          # retrieved chunks
    "ground_truth": [...]       # ground_truth_answer field
}
dataset = Dataset.from_dict(data)
```

---

## Notable Questions by Use Case

| Use Case | Question IDs |
|:---|:---|
| Test table extraction accuracy | GT-E-001 through GT-E-020 |
| Test figure/chart extraction | GT-E-012, GT-E-014, GT-E-016, GT-M-005, GT-M-006 |
| Test non-GAAP awareness | GT-E-004, GT-E-013, GT-M-004, GT-M-018 |
| Test multi-hop reasoning | GT-M-001, GT-M-003, GT-M-007, GT-C-006 |
| Test cross-modal fusion | GT-C-001, GT-C-002, GT-C-006, GT-C-015 |
| Test competitive analysis | GT-M-002, GT-C-003, GT-C-007 |
| Test segment analysis | GT-C-008, GT-C-012, GT-C-013 |
| Test crisis resilience narrative | GT-C-005, GT-C-013 |
| Stress-test multi-modal retrieval | GT-C-015 (meta-question) |
