# Multi-Modal RAG: Enterprise Financial Document Intelligence

An enterprise-grade Multi-Modal Retrieval-Augmented Generation (RAG) system engineered to parse, index, and query dense, complex financial reports containing multi-column executive letters, borderless accounting tables, native PDF vector charts, and hierarchical footnotes.

**Benchmark Target:** `input/jpmc_annualreport-2025.pdf` (JPMorgan Chase & Co. 2025 Annual Report: 364 pages, 152 TOC bookmarks, 8 distinct page archetypes).

---

## 📌 Project Overview & Core Challenges

Standard RAG architectures fail when applied to real-world financial filings (Form 10-K, Annual Reports) due to four fundamental challenges:

1. **Native Vector Graphics vs. Bitmaps:** Financial trend charts (e.g. 20-year net income progression, CET1 capital ratios) are rendered via native PDF drawing commands (lines, curves, fills)—**not** embedded PNG/JPEG files. Standard image extractors find 0 images, while naive text scrapers extract unanchored numbers (`$58.5`, `24%`) stripped of series labels, legends, and axes.
2. **Borderless & Multi-Tier Accounting Tables:** Tables often lack visible gridlines, rely on whitespace alignment, and feature multi-tier column headers (e.g., *Year ended Dec 31* spanning *2025 \| 2024 \| 2023*). Rule-based parsers split rows, misalign columns, or detect 0 tables.
3. **Multi-Column Prose Flow:** Executive and shareholder letters use 2-column and 3-column formats. Line-by-line scrapers interleave columns horizontally, causing semantic corruption in RAG embeddings.
4. **The Generative VLM "Silent Column Omission" Trap:** While Vision-Language Models (VLMs) excel at borderless tables, free-form autoregressive text generation can silently drop dense outer columns (e.g. omitting the 2023 column from 3-year summary tables).

To solve these challenges, this repository evaluates **7 distinct parsing frameworks** and establishes a high-throughput **Hybrid Document Layout Analysis (DLA) + Vision-Language Model (VLM)** ingestion architecture.

---

## 🔬 Framework Benchmarks & Comparative Scorecard

We systematically tested 7 frameworks across the 8 page archetypes of the JPMC 2025 Annual Report, culminating in a full 364-page production run:

| Framework | Engine / Tech | Speed / Page | Multi-Col Prose | Borderless Tables | Bordered Tables | Vector Charts | Key Trade-Off / Role |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **PyMuPDF (`fitz`)** | MuPDF (C library) | **1 – 5 ms** | Good (auto-sort) | **Fails (0 found)** | High | Coordinates only | **Pass-0 Ingestion Backbone:** Blazing fast (~300 p/s) profiling, block sorting, and 216 DPI rendering. |
| **Docling** | LayoutLM + TableFormer | 15 – 30 s (GPU) | **Excellent** | **Exceptional** | **Exceptional** | Bounding box only | **Primary Accounting Table Engine:** Binds native PDF tokens to cells with zero hallucination or column drop. |
| **GLM-OCR** | Local VLM (Ollama) | **3.92 s (Actual avg)** | **Excellent** | Moderate (omission risk) | **Exceptional** | **Exceptional (Trends/Axes)** | **Primary Visual Transducer:** Transcribes visual charts, axes, and trends into Markdown. Full 364 pages in 23.76 min. |
| **pdfplumber** | pdfminer.six (Python) | 250 – 600 ms | **Fails (interleaves)** | Moderate (custom text) | High | Primitives only | **Visual Debugger:** Low-level bounding box inspection and coordinate-based cropping. |
| **Camelot** | Stream & Lattice | 500 – 2,000 ms | N/A (tables only) | Moderate (noisy) | High (needs GS) | N/A (tables only) | **Targeted Table Extractor:** Built-in accuracy metric report; requires Ghostscript for lattice mode. |
| **Tabula-py** | Java Apache PDFBox | 1,500 – 3,000 ms | N/A (tables only) | Moderate (Stream mode) | High (Lattice mode) | N/A (tables only) | **CPU Table Fallback:** Direct Pandas DataFrames; requires JRE and `encoding='latin1'` on Windows. |
| **pytesseract** | Tesseract 5.5 (LSTM) | 2,000 – 4,500 ms | Moderate (PSM mode) | **Fails (raw text)** | **Fails (raw text)** | Isolated numbers | **Scanned PDF Fallback ONLY:** Anti-pattern for digital PDFs (500x slower, introduces OCR typos). |

> 📖 **Full Technical Analysis & Decision Matrix:** See [`parsing/framework_comparison_and_decision_matrix.md`](parsing/framework_comparison_and_decision_matrix.md) for the complete 10-point evaluation report.

---

## 🏛️ Production Hybrid Ingestion Architecture

```
                             [Raw Input PDF: 364 Pages]
                                         |
                                         v
                         +-------------------------------+
                         |   PyMuPDF Pass-0 Profiler     |  (< 1 second total)
                         | - Extract 152 TOC bookmarks   |
                         | - Check selectable font stream|
                         +-------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |    Deterministic Page Classifier      |  (~2 ms / page)
                     |    - drawings > 50 & text < 2,000     |
                     |    - '$' count & table markers        |
                     |    - font signatures (Tiempos/Sons)   |
                     +---------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
[Archetype 5: Vector Charts]    [Archetype 2, 6, 7: Tables]    [Archetype 3, 4, 8: Prose]
(Pages 8, 9, 11-13, 77)         (Pages 2, 76, 85, 197-351)     (Pages 4-74, 352-364)
         |                               |                               |
         v                               v                               v
[High-DPI Render: 216 DPI]       [Docling TableFormer]         [PyMuPDF Block Sorter]
PyMuPDF get_pixmap()             Binds digital tokens to       Dynamic column clustering
         |                       grid cells (zero omission)              |
         v                               |                               v
[Local GLM-OCR via Ollama]               v                     [Clean Sequential Text]
Prompt: Chart Title +             [Structured Table]           Heading hierarchy
Axes + Trends + Series                   |                               |
         |                               |                               v
         v                               v                     [Hierarchical Chunker]
[Synthetic Chart Summary]        [Markdown / JSON Table]       Guided by 152 TOC items
         |                               |                               |
         \                               |                              /
          \                              |                             /
           v                             v                            v
         +---------------------------------------------------------------+
         |             Unified Multimodal Knowledge Store                |
         |  - Text Chunks (TOC metadata + section hierarchy)            |
         |  - Table Chunks (Markdown + Row JSONification)               |
         |  - Figure Chunks (Synthetic VLM summary + Image Anchor)       |
         +---------------------------------------------------------------+
```

---

## 📂 Repository Structure

```
.
├── input/
│   ├── jpmc_annualreport-2025.pdf        # Primary benchmark document (364 pages)
│   ├── NVIDIA-2024-Annual-Report.pdf     # NVIDIA 2024 Annual Report
│   └── apple_10-K-Q4-2023-As-Filed.pdf   # Apple 10-K Q4 2023 Filing
├── parsing/
│   ├── 01_docling_layout_analysis.ipynb  # Docling GPU layout analysis & TableFormer
│   ├── 02_image_classification_preprocessing.ipynb # Visual classification with local VLM (gemma4:26b)
│   ├── 03_pymupdf_parsing_analysis.ipynb # PyMuPDF block geometry, reading order, vector drawings
│   ├── 04_pdfplumber_parsing_analysis.ipynb # pdfplumber primitive inspection & visual table debug
│   ├── 05_camelot_table_extraction.ipynb # Camelot stream/lattice evaluation & quality reports
│   ├── 06_tabula_table_extraction.ipynb  # Tabula-py Java engine & financial table parsing
│   ├── 07_pytesseract_ocr_analysis.ipynb # Google Tesseract OCR 5.5 benchmarking & PSM modes
│   ├── 08_glm_ocr_ollama_analysis.ipynb  # Local VLM (glm-ocr) for borderless tables & vector charts
│   ├── 09_framework_comparison_benchmark.ipynb # Master 7-framework comparative benchmarking
│   ├── 10_full_document_glm_ocr_pipeline.ipynb # Full 364-page document pipeline & monitoring
│   ├── framework_comparison_and_decision_matrix.md # Comprehensive architectural report & decision matrix
│   └── scripts/
│       ├── run_classification.py         # Visual element classification runner
│       └── retry_failed.py               # Checkpoint recovery & retry utility
├── planning/
│   ├── multimodal_rag_approach.md        # Initial document profile & archetype strategy
│   └── scratch_analysis.py               # Exploratory PDF metrics inspection
├── pyproject.toml                        # Project configuration and dependencies
└── README.md
```

---

## 🔍 Document Anatomy: 8 Page Archetypes

| # | Archetype | Target Pages | Characteristics | Recommended Ingestion Strategy |
|:--|:---|:---:|:---|:---|
| 1 | **Cover & TOC** | 1, 3 | Title imagery, Table of Contents bookmarks | **PyMuPDF:** Extract 152 TOC bookmarks in < 50ms for chunk navigation hierarchy. |
| 2 | **Financial Highlights Table** | 2, 76 | Borderless summary tables, multi-tier headers | **Docling (TableFormer):** Exact cell bounding and token binding (avoids VLM column omission). |
| 3 | **Shareholder Letter (2-col)** | 4–51 | Dense 2-column serif prose, callout quotes | **PyMuPDF Block Sorter:** Dynamic horizontal margin clustering to prevent column interleaving. |
| 4 | **Senior Exec Letters (3-col)**| 52–74 | Dense 3-column prose essays | **PyMuPDF Block Sorter:** 3-column dynamic band sorting (~500 pages/sec on CPU). |
| 5 | **Pure Vector Charts & Infographics** | 8, 9, 11–13, 77 | 0 raster images, 50–305 vector paths, floating numbers | **PyMuPDF High-DPI Render (216 DPI) + GLM-OCR:** Transcribes axes, units, series, and trends. |
| 6 | **MD&A Mixed Text & Tables** | 78–192 | Prose narrative explaining financial variance tables | **Hybrid:** PyMuPDF for narrative text + Docling TableFormer for embedded accounting tables. |
| 7 | **Audited Statements & Notes** | 193–351 | Balance Sheets, Income Statements, Notes 1–34 | **Docling TableFormer or Tabula-py (Lattice):** Preserves nested rows and footnotes. |
| 8 | **Glossary & End Matter** | 352–364 | Definition pairs (`Term -> Definition`) | **PyMuPDF Regex Parser:** Direct key-value extraction for exact-match terminology resolution. |

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* [`uv`](https://github.com/astral-sh/uv) package manager
* (Optional for local VLM) [Ollama](https://ollama.com/) with `glm-ocr:latest` or `gemma4:26b`
* (Optional for GPU acceleration) NVIDIA GPU with CUDA support (e.g. RTX 4070 Ti SUPER)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/maseed260/Multi-modal-RAG.git
cd Multi-modal-RAG

# Checkout the active parsing development branch
git checkout parsing_dev

# Install dependencies using uv
uv sync
```

### Running the Parsing Benchmarks

Open any of the interactive notebooks in VS Code or Jupyter:
* **Master Comparison:** `parsing/09_framework_comparison_benchmark.ipynb`
* **Full-Document Pipeline:** `parsing/10_full_document_glm_ocr_pipeline.ipynb`
* **Docling Layout Analysis:** `parsing/01_docling_layout_analysis.ipynb`
* **PyMuPDF Extraction:** `parsing/03_pymupdf_parsing_analysis.ipynb`

---

## 🗺️ Roadmap & Current Progress

- [x] **Phase 0:** Document profiling, font signatures, vector drawing detection, and archetype classification.
- [x] **Phase 1.1:** Comparative framework evaluation across 7 parsers (PyMuPDF, pdfplumber, Camelot, Tabula, Tesseract, Docling, GLM-OCR).
- [x] **Phase 1.2:** Full 364-page end-to-end document visual transduction with GLM-OCR (23.76 min, 100% success).
- [x] **Phase 1.3:** Architectural decision matrix and failure mode analysis (TableFormer vs. VLM column omission).
- [x] **Phase 2:** Automated Hybrid Ingestion Engine implementation (Pass-0 Classifier + TableFormer + VLM chart captioner).
- [x] **Phase 3:** TOC-guided hierarchical chunking and metadata enrichment (linking footnote markers `(a)`, `(b)`).
- [x] **Phase 4:** Hybrid multi-vector indexing (Dense embeddings + Sparse BM25 + Visual crop anchors).
- [x] **Phase 5:** Multi-modal query routing, grounded generation, and visual source citations (LangGraph Agent + Qdrant hybrid retrieval + Groq synthesis).
