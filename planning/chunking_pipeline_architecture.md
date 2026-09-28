# Multi-Modal Chunking Pipeline: Architecture & Data Flow

## Data Sources

| Source | File | What It Provides |
|---|---|---|
| **Docling JSON** | `parsing/output/full_docling_parsed.json` (115MB) | Reading-order tree of 6,355 text blocks + 295 tables + 73 pictures, each with type label, page number, and bounding box |
| **GLM-OCR Pages** | `parsing/output/glm_full/pages/page_*.md` (364 files) | Rich visual transcription per page — the only source that semantically understands vector charts (axes, trends, time horizons) |
| **Classified Images** | `parsing/output/classified_images.json` | VLM classification (category + summary + rag_action) for each of the 73 Docling-detected pictures |
| **TOC Bookmarks** | `parsing/output/toc_bookmarks.json` (extracted from PDF via PyMuPDF `get_toc()`) | 152 hierarchical section entries at 6 depth levels, covering all 364 pages |

---

## Pipeline: 4 Passes

### Pass 1: Walk Docling's Tree → Flat Element Stream

Docling's JSON (`full_docling_parsed.json`) contains a document tree: `body` → `children` → `$ref` pointers to `texts[]`, `tables[]`, `pictures[]`. Walking this tree produces a flat, **reading-order** sequence of labeled elements:

```
[section_header]  p.2:   "Financial Highlights"
[TABLE 32x4]      p.2:   (structured table with cell data)
[text]             p.2:   "(a) Pre-provision profit, TBVPS..."
...
[section_header]  p.79:  "EXECUTIVE OVERVIEW"
[text]             p.79:  "This executive overview of the MD&A..."
[TABLE 24x4]      p.79:  (financial performance table)
[text]             p.79:  "(a) As of January 1, 2025..."
[list_item]        p.80:  "The provision for credit losses was $14.2B..."
...
```

Each element knows its **type** (`text`, `table`, `picture`), **label** (`section_header`, `text`, `list_item`, `caption`, `page_footer`), **page number**, and **content** (text string or structured table cells).

**Skip** elements with label `page_footer` or `page_header` (running headers like "JPMorgan Chase & Co./2025 Form 10-K").

**Why Docling and not GLM-OCR pages?** Docling gives us typed, labeled elements in reading order. GLM-OCR flattens everything into markdown — you can't tell where a table ends and prose begins. However, GLM-OCR is used later for chart transcription and table summary generation.

---

### Pass 2: Group Elements by TOC Section

Use the 152 TOC bookmarks to group elements into **logical sections** (ignoring page boundaries):

- Each TOC entry defines a section: `[start_page, next_entry_start_page)`
- Assign every element to its **deepest (most specific)** matching TOC section
- A section spanning pages 79–82 simply has all its elements in sequence — the page boundary is invisible

**TOC coverage is uneven:**
- Pages 75–351 (financial sections): **130+ entries**, 1–5 pages each → TOC-guided chunking works perfectly
- Pages 4–51 (shareholder letter): **5 entries**, 12–16 pages each → needs heading detection fallback or pure semantic splitting
- Pages 352–364 (end matter): **4 entries**, sparse

**For long sections (>5 pages) without sub-TOC entries**, use a 3-tier fallback:
1. **Tier 1**: TOC section boundary (if ≤5 pages)
2. **Tier 2**: Detect in-content headings (`#` markdown headings, `**bold**` sub-headers, ALL-CAPS lines) from GLM-OCR output
3. **Tier 3**: Pure semantic splitting using embedding similarity valleys (for truly heading-less prose like pages 36–48)

---

### Pass 3: Modality-Specific Chunking (The Key Insight)

Within each section, walk the element sequence. **Tables and figures are breakpoints** that split the text stream:

```
text text text → merge into one text run → semantic split into ~512-token chunks
TABLE         → separate chunk(s): row-groups + LLM summary  
text text     → new text run → semantic split
PICTURE       → separate figure chunk (GLM-OCR text + image anchor)
text text text → another text run → semantic split
```

**Text before a table and text after a table never get merged together.** The table sits between them as its own entity.

#### Text Chunking
- Consecutive text elements (including `list_item`, `section_header`) within the same text run get **concatenated** into one stream
- Cross-page mid-sentence breaks are detected and merged (e.g., sentence ending without `.!?` followed by lowercase continuation)
- The merged text run is then **semantic-split** into chunks of ~512 tokens with 2–3 sentence overlap

#### Table Chunking (Three Representations)
- **Row-group chunks**: Split large tables into groups of 5–10 rows, **always repeating the full column headers** in each chunk (a row like `$321,596` is useless without "Deposits with banks" + "2025" + "(in millions)")
- **Executive summary**: LLM-generated natural-language summary of the table's key figures and trends (uses GLM-OCR transcription as context for the prompt)
- Tables < 512 tokens → single chunk as-is

#### Figure/Chart Chunking (Dual-Anchor)
- **Decorative** images (`rag_action: "discard"` in `classified_images.json`) → skip
- **Vector charts** (pages 8, 9, 11–13, 77): use GLM-OCR transcription directly as `text_for_embedding` (already contains axes, trends, time horizons)
- **Other figures**: use VLM classification summary from `classified_images.json`
- Each figure chunk carries an `image_path` anchor — at generation time, the LLM receives both the text summary AND the rendered image

---

### Pass 4: Enrichment

After chunking, enrich every chunk with metadata and cross-references:

1. **TOC ancestry path**: e.g., `["Financial:", "MD&A:", "Firmwide Risk Management", "Capital Risk Management"]`
2. **Prev/next pointers**: linked-list navigation for adjacent chunk retrieval at query time
3. **Cross-modal sibling IDs**: all chunks sharing the same `toc_section` are siblings — retrieving any one can pull the others (text + table + figure about the same topic)
4. **Footnote re-attachment**: detect `(a)`, `(b)`, `*` markers in table/text chunks and inline the footnote definitions

---

## Output Format

`unified_chunks.jsonl` — one JSON object per line:

```json
{
  "chunk_id": "text_exec_overview_02",
  "modality": "text",
  "toc_section": "Executive Overview",
  "toc_parent": "Management's discussion and analysis",
  "toc_path": ["Financial:", "MD&A:", "Executive Overview"],
  "toc_level": 3,
  "page_range": [79, 80],
  "chunk_index": 2,
  "total_chunks_in_section": 8,
  "prev_chunk_id": "table_exec_overview_01_summary",
  "next_chunk_id": "text_exec_overview_03",
  "token_count": 487,
  "text": "Apple Card transaction: On January 7, 2026... Firmwide overview: JPMorganChase reported net income of $57.0 billion...",
  "splitting_tier": "toc_guided",
  "section_sibling_ids": ["text_exec_overview_01", "table_exec_overview_01", "..."]
}
```

Table chunks add: `"modality": "table"` or `"modality": "table_summary"`
Figure chunks add: `"modality": "figure"`, `"image_path": "figures/page_008_render.png"`

---

## Estimated Scale

| Modality | Estimated Chunks | Rationale |
|---|---|---|
| Text (prose) | ~400–500 | ~200 prose pages × 2–3 chunks/page avg |
| Tables (row-groups) | ~200–300 | ~80 logical tables × 2–4 row-groups each |
| Table summaries | ~80–100 | 1 per logical table |
| Figures/Charts | ~20–30 | Non-decorative only (filtered by classifier) |
| **Total** | **~700–930** | Manageable for any vector DB |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Chunk size | ~512 tokens | Sweet spot for financial text density with modern embedding models |
| Overlap | 2–3 sentences at text chunk boundaries | Simple, preserves continuity |
| Table source | Docling for row-group structure, GLM-OCR for summary generation prompts | Docling has exact cell binding; GLM-OCR has richer narrative context |
| Storage | JSONL (one chunk per line) | Streamable, debuggable, easy to filter |
| Footnote handling | Inline expansion in chunk text | Avoids losing context at retrieval time |
