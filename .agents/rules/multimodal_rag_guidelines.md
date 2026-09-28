# Multi-Modal RAG Engineering Guidelines

## 1. Docling Schema & Pydantic Data Contracts
- **Bounding Box Typing**: Docling's `bbox` dictionary contains both coordinates (`l`, `t`, `r`, `b`) and coordinate frame metadata (`coord_origin: "BOTTOMLEFT"`).
  - **Rule**: Never type bounding boxes as `Dict[str, float]`. Always type as `Optional[Dict[str, Any]]` or use a dedicated model accepting `coord_origin: Optional[str]`.

## 2. Local VLM Selection for Financial Visuals
- **Primary Visual Transducer**: Always prefer `glm-ocr:latest` for financial charts, diagrams, and vector graphic pages.
  - Latency: 1–5 seconds per graphic (vs 30–180s for 26B+ parameter models).
  - Accuracy: Faithfully transcribes 20-year axes, units, legends, and series labels without hallucination.
- **Output Robustness**: Avoid forcing heavy generative models into raw JSON with strict delimiters when transcribing numbers containing commas (`$158,104`); prefer structured markdown or robustly post-processed sections.

## 3. Modality-Aware Chunking Invariants
- **Table Breakpoint Invariant**: Never merge prose text runs across a table or figure. Tables and figures act as structural barriers.
- **Footnote Absorption**: Text blocks immediately following a table that start with footnote markers (`(a)`, `(b)`, `*`, `1`, `2`) must be absorbed into table metadata/rows, never emitted as orphan text chunks.
- **Lead-In Sentence Folding**: Introductory text immediately preceding a table that is < 80 tokens (e.g., *"The following table presents..."*) must be folded into the table chunk as a preamble/caption.
- **Cross-Page Paragraph Healing**: When consecutive elements span a page boundary:
  - De-hyphenate words split at line ends (`word-` + `remainder`).
  - If the page ends without terminal punctuation (`. ! ? : ;`), stitch with a single space.
- **Table Row-Grouping**: Sliced tables (5–10 rows) must **always clone and repeat the full multi-tier column headers** and unit scales (`in millions`) into each chunk.
- **Provenance & Graph Navigation**: Every chunk must retain `page_range`, human-readable `citation`, `toc_path`, `prev_chunk_id`, `next_chunk_id`, and `section_sibling_ids`.
