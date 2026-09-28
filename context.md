# Project Context: Multi-Modal RAG from Scratch (Enterprise Architecture)

## 1. Vision & Core Objective
This repository is an intensive, end-to-end implementation of an **Enterprise-Grade Multi-Modal Retrieval-Augmented Generation (RAG)** system built from first principles.

* **Primary Learner Goal:** Master every layer of the Multi-Modal RAG stack at an **AI Architect level** — moving beyond toy RAG tutorials to solve real-world problems presented by complex, dense enterprise documents.
* **Target Benchmark Document:** `input/jpmc_annualreport-2025.pdf` (JPMorgan Chase & Co. Annual Report 2025: 364 pages, 152 TOC bookmarks, 8 distinct page archetypes).

---

## 2. Why Traditional RAG Fails on Complex Enterprise PDFs

1. **Native Vector Graphics vs. Bitmaps:** Financial charts (e.g., 20-year net income progression, CET1 ratios, deposit trends) are rendered as native PDF vector operations (lines, fills, polygons, coordinates), not embedded JPEG/PNG files. Naïve image extractors extract 0 images, while text extractors pull isolated floating numbers (`$58.5`, `24%`, `$20.02`) with no axes, series labels, or legends.
2. **Borderless & Hierarchical Accounting Tables:** Tables often lack gridlines, feature multi-tier column headers (e.g., *Year ended Dec 31* spanning 2025/2024/2023), indent-based parent-child rows, and footnotes (`(a)`, `(b)`). Standard scrapers slice through rows, destroying relational integrity.
3. **Multi-Column Prose Flow:** Executive letters (e.g., Jamie Dimon's Shareholder Letter) use 2-column and 3-column newspaper formats with callout sidebars. Standard line-by-line scrapers interleave columns horizontally.
4. **Context Loss Across Modalities:** In pure text RAG, a user asking *"What was the net income trend over the last 20 years?"* fails because the answer is locked inside a visual vector chart.
5. **The Generative VLM "Silent Column Omission" Trap:** While Vision-Language Models (VLMs) excel at recognizing visual whitespace, free-form text generation can silently drop dense outer columns (e.g. dropping the 2023 column from 3-year summary tables).

---

## 3. Architecture Blueprint: The 5 RAG Lifecycle Pillars

```
+-----------------------------------------------------------------------------------+
| 1. MULTI-MODAL PARSING & DECOMPOSITION (Completed Benchmarks)                     |
|    - Pass-0 Document Profiler (PyMuPDF: 152 TOC bookmarks in < 50ms)             |
|    - Deterministic Page Classifier (Drawings count, text density, '$' indicators) |
|    - Table Extraction: Docling TableFormer (Immutable token binding to cells)     |
|    - Visual Chart Extraction: Local GLM-OCR (Titles, axes, trends, time horizons) |
|    - Dynamic Prose Extraction: PyMuPDF with auto x0 margin clustering (no cols arg)|
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. MODALITY-AWARE CHUNKING & ENRICHMENT (Completed)                               |
|    - Pass 1: Docling tree walk -> flat reading-order stream (4,814 elements)       |
|    - Pass 2: 152 TOC bookmarks -> 129 active hierarchical sections (364 pages)    |
|    - Pass 3: Modality-specific chunking (Text ~512 tok, Table row-parts, Figures) |
|    - Pass 4: Dual-anchor enrichment (TOC path, prev/next, footnotes, metrics)     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. HYBRID MULTI-VECTOR INDEXING (Next Focus)                                      |
|    - Dense Semantic Embeddings + Sparse Keyword (BM25 / SPLADE)                   |
|    - Visual Embeddings / Multi-vector anchors (links back to rendered page images)|
|    - Vector Database with rich metadata filtering                                |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. QUERY ROUTING & ADAPTIVE RETRIEVAL                                            |
|    - Intent Routing: Numerical trend -> Charts; Data lookup -> Tables; Narrative  |
|    - Multi-modal context assembly (Text snippets + Tables + Visual crops)        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. GROUNDED GENERATION & AUDITABLE CITATIONS                                     |
|    - Multi-modal LLM synthesis with exact page citations                          |
|    - Visual grounding & numerical hallucination checks                           |
+-----------------------------------------------------------------------------------+
```

---

## 4. Phase 1 Deep-Dive: Parsing & Decomposition Benchmark Findings

We conducted head-to-head empirical testing across **7 parsing frameworks** and executed a full 364-page document transduction.

### 4.1 Comparative Evaluation Summary

| Framework | Role in Production Architecture | Speed / Page | Core Finding |
|:---|:---|:---:|:---|
| **PyMuPDF (`fitz`)** | **Pass-0 Ingestion Backbone** | **1 – 5 ms** | Profiles 364 pages in 1.2s; extracts vector drawings; best 216 DPI renderer. |
| **Docling (IBM)** | **Primary Table & Layout Engine** | 15 – 30 s (GPU) | **TableFormer binds native PDF tokens to cells with zero omission**. Best for accounting tables. |
| **GLM-OCR (Ollama)** | **Primary Visual Transducer** | **3.92 s (Measured)** | **The only framework that semantically understands vector charts**. Processed 364 pages in 23.76 min. |
| **pdfplumber** | **Visual Debugger** | 250 – 600 ms | Character/curve inspection; `.to_image().draw_tables()` helps verify edge snapping. |
| **Camelot** | **Targeted Table Extractor** | 500 – 2000 ms | Built-in `parsing_report['accuracy']`; requires Ghostscript for lattice mode. |
| **Tabula-py** | **CPU Table Fallback** | 1500 – 3000 ms | Direct Pandas DataFrame extraction; requires Java and `latin1` encoding on Windows. |
| **pytesseract** | **Scanned PDF Fallback ONLY** | 2000 – 4500 ms | Anti-pattern for digital PDFs (500x slower, introduces OCR noise, destroys tables). |

### 4.2 Key Architectural Breakthroughs

1. **Table Integrity (TableFormer vs. Generative VLMs):**
   - On Page 2 (Financial Highlights), **GLM-OCR silently omitted the 2023 column** (`$158,104`, `$87,172`, `$70,932`) from the Selected Income Statement section by drifting into a bulleted list.
   - **Docling TableFormer** extracted all 4 columns with 100% precision because it maps immutable digital text tokens from the PDF into predicted grid cells. **Rule:** Primary financial tables must use Docling TableFormer.
2. **Semantic Understanding of Vector Charts:**
   - Financial charts on Pages 8, 9, 11–13, and 77 contain 0 bitmap images and over 6,200 PDF drawing paths. Docling detects only a bounding box.
   - **GLM-OCR via Ollama** transcribes axes, units, 20-year time horizons (2005–2025), and accounting adjustments (TCJA, CECL, Visa shares) directly into structured text. **Rule:** Pure vector charts must route to GLM-OCR.
3. **Dynamic Reading Order (Zero Hardcoded Columns):**
   - Instead of manually passing `num_cols`, PyMuPDF can cluster horizontal left-margin origins ($x_0$). It dynamically discovers 1-col, 2-col (Page 16), or 3-col (Pages 7, 52) layouts on the fly, eliminating column interleaving at 500 pages/sec on CPU.

---

## 5. Phase 2 Deep-Dive: Modality-Aware Chunking & Enrichment Architecture

We engineered a **4-Pass Hierarchical Chunking Pipeline** processing all 364 pages into 1,452–1,491 unified multimodal chunks stored in `unified_chunks.json` and `unified_chunks.jsonl`.

### 5.1 The 4-Pass Chunking Lifecycle

```
[Docling full_docling_parsed.json] ---> Pass 1: Tree Walk ---> Flat Stream (4,814 elements)
                                                                       |
[TOC Bookmarks: 152 items]          ---> Pass 2: Section Map ---> 129 Active Sections
                                                                       |
+----------------------------------------------------------------------+
| Pass 3: Modality-Aware Splitting & Context Injection
|  - Text: Cross-page healing + heading-aware micro-splits (~512 tokens)
|  - Tables: Repeated column headers on splits + inlined footnotes + surrounding context
|  - Figures: Dual-anchor (rendered image + GLM-OCR institutional metrics + context)
+----------------------------------------------------------------------+
                                   |
Pass 4: Graph Enrichment ---> TOC Path + Prev/Next Pointers + Sibling Links + Exact Citations
                                   |
                                   v
             [unified_chunks.json / unified_chunks.jsonl]
```

### 5.2 Modality-Specific Implementations

1. **Text (Prose) Chunking:**
   - **Cross-Page Healing:** Detects sentence splices across page breaks (e.g. line ending without terminal punctuation followed by lowercase token) and merges them to prevent semantic truncation.
   - **Heading-Aware Micro-Splitting:** Inspects section headers and bold subheadings to enforce logical semantic boundaries before applying 512-token windowing with 2–3 sentence overlaps.
2. **Table Chunking:**
   - **Header Persistence:** For tall financial tables split into row groups (e.g., Financial Highlights spanning 4 sub-chunks), **full column headers are replicated** on every chunk so numbers are never unanchored from column identities.
   - **Inlined Footnotes:** Footnote symbols (`(a)`, `(b)`, `*`) in cells are automatically resolved against the page footnote definitions and injected directly into the table chunk markdown.
   - **Surrounding Narrative Context:** Injects the preceding 1–2 sentences of prose into a `> **Surrounding Context**` blockquote.
3. **Figure & Vector Chart Chunking (Dual-Anchor):**
   - **Visual Anchor:** Exact filepath to rendered 216 DPI image crop (`parsing/output/figures/pic_*.png`).
   - **Structured Institutional Metrics:** High-fidelity quantitative data extraction (GLM-OCR) populating `quantitative_metrics` (e.g., 2005–2025 Net Income $8.5B $\rightarrow$ $57.0B, EPS $4.03 $\rightarrow$ $20.02, ROTCE 20%, 10-Yr Total Return 20.4% CAGR).
   - **Preceding Narrative Context:** Anchors the visual chart to the surrounding discussion in the shareholder/executive letters.

---

## 6. Execution Roadmap & Phase Status

1. **Step 1.1 — Representative Test Harness:** ✅ **Completed** (11 representative pages tested across 7 frameworks).
2. **Step 1.2 — Dedicated Framework Notebooks:** ✅ **Completed** (Notebooks `01` through `09` in `parsing/`).
3. **Step 1.3 — Full-Document Transduction:** ✅ **Completed** (364 pages processed via GLM-OCR in 23.76 min, Notebook `10`).
4. **Step 1.4 — Architectural Decision Matrix:** ✅ **Completed** ([`framework_comparison_and_decision_matrix.md`](parsing/framework_comparison_and_decision_matrix.md)).
5. **Step 2.1 — Production Hybrid Ingestion Pipeline:** ✅ **Completed** (PyMuPDF Profiler + Docling TableFormer + GLM-OCR).
6. **Step 2.2 — Modality-Aware Chunking & TOC Hierarchy:** ✅ **Completed** (Interactive notebook [`chunking/01_multimodal_hierarchical_chunking.ipynb`](chunking/01_multimodal_hierarchical_chunking.ipynb) & [`run_updated_pipeline.py`](chunking/run_updated_pipeline.py)).
7. **Step 2.3 — High-Fidelity Figure Metric Extraction:** ✅ **Completed** (All 34 informative figures enriched with explicit quantitative metrics in [`detailed_image_summaries.json`](chunking/output/detailed_image_summaries.json)).
8. **Step 3.1 — Hybrid Multi-Vector Indexing:** ✅ **Completed** (1024d Dense Qwen3-Embedding + FastEmbed BM25 Sparse + HNSW & Payload Indexes in Qdrant, Notebook [`embedding/01_qdrant_hybrid_embedding_indexing.ipynb`](embedding/01_qdrant_hybrid_embedding_indexing.ipynb)).
9. **Step 4.1 & 5.1 — Agentic Multi-Modal RAG with LangGraph:** ✅ **Completed** (LangGraph StateGraph agent with Qdrant hybrid retrieval tool, Groq grounded synthesis tool, multi-rewrite reasoning, checkpointers, and LangGraph Studio configuration).
