import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
PARSING_OUT = ROOT_DIR / "parsing" / "output"
DOCLING_JSON_PATH = PARSING_OUT / "full_docling_parsed.json"
TOC_JSON_PATH = PARSING_OUT / "toc_bookmarks.json"
CLASSIFIED_IMG_PATH = PARSING_OUT / "classified_images.json"
DETAILED_SUMMARIES_PATH = ROOT_DIR / "chunking" / "output" / "detailed_image_summaries.json"
OUTPUT_DIR = ROOT_DIR / "chunking" / "output"

class ChunkModality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    TABLE_SUMMARY = "table_summary"
    FIGURE = "figure"
    CHART = "chart"

class ChunkProvenance(BaseModel):
    doc_id: str = "jpmc_annualreport-2025"
    doc_title: str = "JPMorgan Chase & Co. 2025 Annual Report"
    page_start: int
    page_end: int
    page_range: List[int]
    citation: str
    bbox: Optional[Dict[str, Any]] = None

class TOCContext(BaseModel):
    toc_section: str
    toc_parent: Optional[str] = None
    toc_path: List[str] = Field(default_factory=list)
    toc_level: int = 1
    toc_bookmark_index: int = 0

class GraphNavigation(BaseModel):
    chunk_index_in_section: int = 0
    total_chunks_in_section: int = 0
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    section_sibling_ids: List[str] = Field(default_factory=list)
    companion_table_id: Optional[str] = None
    companion_figure_id: Optional[str] = None

class MultimodalAnchor(BaseModel):
    image_path: Optional[str] = None
    visual_category: Optional[str] = None
    quantitative_metrics: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class UnifiedChunk(BaseModel):
    chunk_id: str
    modality: ChunkModality
    text: str
    token_count: int
    char_count: int
    provenance: ChunkProvenance
    toc: TOCContext
    navigation: GraphNavigation
    multimodal: Optional[MultimodalAnchor] = None
    inlined_footnotes: List[str] = Field(default_factory=list)
    filter_metadata: Dict[str, Any] = Field(default_factory=dict)

def approximate_token_count(text: str) -> int:
    return max(1, len(text) // 4)

def heal_cross_page_text(last_text: str, next_text: str) -> str:
    last_text = last_text.rstrip()
    next_text = next_text.lstrip()
    if not last_text:
        return next_text
    if not next_text:
        return last_text
    if re.search(r'[a-zA-Z]-$', last_text) and re.match(r'^[a-z]', next_text):
        return last_text[:-1] + next_text
    terminal_punct = ('.', '!', '?', ':', ';')
    if not last_text.endswith(terminal_punct):
        return last_text + " " + next_text
    return last_text + "\n\n" + next_text

def is_footnote_element(text: str) -> bool:
    t = text.strip()
    return bool(re.match(r'^\([a-z0-9]\)', t) or t.startswith('*') or re.match(r'^[0-9]\s+[A-Z]', t))

def format_docling_table(table_raw: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    grid = table_raw.get("data", {}).get("grid", [])
    if not grid:
        return [], []
    header_rows = []
    data_rows = []
    for r_idx, row in enumerate(grid):
        row_cells = [cell.get("text", "").strip() for cell in row]
        if r_idx == 0:
            header_rows.append(row_cells)
        else:
            data_rows.append(row_cells)
    return header_rows[0] if header_rows else [], data_rows

def table_to_markdown(header: List[str], rows: List[List[str]]) -> str:
    if not header and not rows:
        return ""
    if not header and rows:
        header = [f"Col {i+1}" for i in range(len(rows[0]))]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for r in rows:
        cells = r + [""] * (len(header) - len(r))
        md += "| " + " | ".join(cells[:len(header)]) + " |\n"
    return md

def split_text_into_semantic_chunks(text: str, heading: str = "", max_tokens: int = 512, overlap_sentences: int = 2) -> List[Tuple[str, int]]:
    clean_text = text.strip()
    if not clean_text:
        return []
    heading_prefix = f"### {heading}\n\n" if heading else ""
    full_text = heading_prefix + clean_text
    total_tokens = approximate_token_count(full_text)
    if total_tokens <= max_tokens:
        return [(full_text, total_tokens)]
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if s.strip()]
    if not sentences:
        return [(full_text, total_tokens)]
    chunks = []
    current_sentences = []
    current_tok = approximate_token_count(heading_prefix)
    for s in sentences:
        s_tok = approximate_token_count(s)
        if current_sentences and (current_tok + s_tok > max_tokens):
            chunk_body = " ".join(current_sentences)
            part_suffix = f" (Part {len(chunks)+1})" if heading else ""
            c_text = f"### {heading}{part_suffix}\n\n{chunk_body}" if heading else chunk_body
            chunks.append((c_text, approximate_token_count(c_text)))
            current_sentences = current_sentences[-overlap_sentences:] if len(current_sentences) >= overlap_sentences else current_sentences
            current_tok = approximate_token_count(heading_prefix + " ".join(current_sentences))
        current_sentences.append(s)
        current_tok += s_tok
    if current_sentences:
        chunk_body = " ".join(current_sentences)
        part_suffix = f" (Part {len(chunks)+1})" if chunks and heading else ""
        c_text = f"### {heading}{part_suffix}\n\n{chunk_body}" if heading else chunk_body
        chunks.append((c_text, approximate_token_count(c_text)))
    return chunks

def emit_text_chunks(text: str, subheading: str, pages: set, b_idx: int, toc_info: dict, counter: int) -> Tuple[List[UnifiedChunk], int]:
    if not text.strip():
        return [], counter
    p_list = sorted(list(pages)) or [1]
    sub_chunks = split_text_into_semantic_chunks(text, heading=subheading, max_tokens=512)
    emitted = []
    for part_idx, (c_text, tok) in enumerate(sub_chunks):
        counter += 1
        part_str = f"_part{part_idx+1}" if len(sub_chunks) > 1 else ""
        c_id = f"chunk_sec{b_idx:03d}_text_{counter:04d}{part_str}"
        full_path = list(toc_info["path"])
        if subheading and subheading not in full_path:
            full_path.append(subheading)
        citation = f"JPMC 2025 Annual Report, pp. {p_list[0]}-{p_list[-1]} ({' > '.join(full_path)})"
        emitted.append(UnifiedChunk(
            chunk_id=c_id,
            modality=ChunkModality.TEXT,
            text=c_text,
            token_count=tok,
            char_count=len(c_text),
            provenance=ChunkProvenance(
                page_start=p_list[0],
                page_end=p_list[-1],
                page_range=p_list,
                citation=citation
            ),
            toc=TOCContext(
                toc_section=subheading or toc_info["active_title"],
                toc_parent=toc_info["active_title"] if subheading else toc_info["parent"],
                toc_path=full_path,
                toc_level=toc_info["level"] + (1 if subheading else 0),
                toc_bookmark_index=b_idx
            ),
            navigation=GraphNavigation(),
            filter_metadata={"subheading": subheading, "part": part_idx + 1, "total_parts": len(sub_chunks)}
        ))
    return emitted, counter

with open(DOCLING_JSON_PATH, "r", encoding="utf-8") as f:
    docling_data = json.load(f)
with open(TOC_JSON_PATH, "r", encoding="utf-8") as f:
    toc_bookmarks = json.load(f)
with open(DETAILED_SUMMARIES_PATH, "r", encoding="utf-8") as f:
    detailed_image_summaries = json.load(f)

texts_lookup = {f"#/texts/{i}": t for i, t in enumerate(docling_data.get("texts", []))}
tables_lookup = {f"#/tables/{i}": t for i, t in enumerate(docling_data.get("tables", []))}
pictures_lookup = {f"#/pictures/{i}": p for i, p in enumerate(docling_data.get("pictures", []))}
groups_lookup = {f"#/groups/{i}": g for i, g in enumerate(docling_data.get("groups", []))}

def resolve_ref(ref: str):
    if ref in texts_lookup:
        return ("text", texts_lookup[ref])
    elif ref in tables_lookup:
        return ("table", tables_lookup[ref])
    elif ref in pictures_lookup:
        return ("picture", pictures_lookup[ref])
    return None

def walk_children(children):
    flat_elements = []
    for c in children:
        ref = c.get("$ref")
        if not ref:
            continue
        if ref in groups_lookup:
            group = groups_lookup[ref]
            flat_elements.extend(walk_children(group.get("children", [])))
        else:
            resolved = resolve_ref(ref)
            if resolved:
                modality, data = resolved
                label = data.get("label", "")
                if label in ("page_header", "page_footer"):
                    continue
                prov = data.get("prov", [{}])[0]
                flat_elements.append({
                    "ref": ref,
                    "modality": modality,
                    "label": label,
                    "page_no": prov.get("page_no", 1),
                    "bbox": prov.get("bbox"),
                    "raw_data": data
                })
    return flat_elements

flat_stream = walk_children(docling_data.get("body", {}).get("children", []))
sorted_toc = sorted(toc_bookmarks, key=lambda x: x["page"])

def get_toc_ancestry(page_no: int):
    active = sorted_toc[0]
    active_idx = 0
    for idx, b in enumerate(sorted_toc):
        if b["page"] <= page_no:
            active = b
            active_idx = idx
        else:
            break
    path = [active["title"]]
    current_level = active["level"]
    parent = None
    for p in reversed(sorted_toc[:active_idx]):
        if p["level"] < current_level:
            if parent is None:
                parent = p["title"]
            path.insert(0, p["title"])
            current_level = p["level"]
            if current_level == 1:
                break
    return {
        "active_title": active["title"].strip(),
        "level": active["level"],
        "parent": parent.strip() if parent else None,
        "path": [p.strip() for p in path],
        "bookmark_index": active_idx
    }

sections_map = {}
for elem in flat_stream:
    toc_info = get_toc_ancestry(elem["page_no"])
    b_idx = toc_info["bookmark_index"]
    if b_idx not in sections_map:
        sections_map[b_idx] = {"toc_info": toc_info, "elements": []}
    sections_map[b_idx]["elements"].append(elem)

all_chunks = []
chunk_counter = 0

for b_idx, sec_data in tqdm(sorted(sections_map.items()), desc="Processing TOC Sections"):
    toc_info = sec_data["toc_info"]
    elements = sec_data["elements"]
    section_chunks = []
    current_text_run = ""
    current_subheading = ""
    current_pages = set()
    lead_in_text = ""
    
    i = 0
    while i < len(elements):
        elem = elements[i]
        modality = elem["modality"]
        page_no = elem["page_no"]
        raw = elem["raw_data"]
        label = elem.get("label", "")
        
        if modality == "text":
            text_str = raw.get("text", "").strip()
            if not text_str:
                i += 1
                continue
            is_subheading = (label in ("section_header", "title"))
            if is_subheading:
                if current_text_run:
                    flushed, chunk_counter = emit_text_chunks(current_text_run, current_subheading, current_pages, b_idx, toc_info, chunk_counter)
                    section_chunks.extend(flushed)
                    current_text_run = ""
                    current_pages = set()
                current_subheading = text_str
                current_pages.add(page_no)
                i += 1
                continue
            next_is_table = (i + 1 < len(elements) and elements[i+1]["modality"] == "table")
            if next_is_table and approximate_token_count(text_str) < 80 and not current_text_run:
                lead_in_text = text_str
                i += 1
                continue
            current_text_run = heal_cross_page_text(current_text_run, text_str)
            current_pages.add(page_no)
        elif modality == "table":
            if current_text_run:
                flushed, chunk_counter = emit_text_chunks(current_text_run, current_subheading, current_pages, b_idx, toc_info, chunk_counter)
                section_chunks.extend(flushed)
                current_text_run = ""
                current_pages = set()
                current_subheading = ""
            absorbed_footnotes = []
            j = i + 1
            while j < len(elements):
                next_elem = elements[j]
                if next_elem["modality"] == "text":
                    nxt_txt = next_elem["raw_data"].get("text", "").strip()
                    if is_footnote_element(nxt_txt):
                        absorbed_footnotes.append(nxt_txt)
                        j += 1
                        continue
                break
            i = j - 1
            header, rows = format_docling_table(raw)
            table_title = lead_in_text if lead_in_text else f"Table ({toc_info['active_title']})"
            lead_in_text = ""
            ROW_WINDOW = 10
            row_chunks = [rows[k:k+ROW_WINDOW] for k in range(0, max(len(rows), 1), ROW_WINDOW)]
            for r_idx, r_group in enumerate(row_chunks):
                chunk_counter += 1
                c_id = f"chunk_sec{b_idx:03d}_table_{chunk_counter:04d}_part{r_idx+1}"
                tbl_md = f"### {table_title} (Part {r_idx+1}/{len(row_chunks)})\n\n"
                tbl_md += table_to_markdown(header, r_group)
                if absorbed_footnotes:
                    tbl_md += "\n\n**Inlined Footnotes:**\n" + "\n".join(f"- {fn}" for fn in absorbed_footnotes)
                tok = approximate_token_count(tbl_md)
                citation = f"JPMC 2025 Annual Report, p. {page_no} ({' > '.join(toc_info['path'])})"
                section_chunks.append(UnifiedChunk(
                    chunk_id=c_id,
                    modality=ChunkModality.TABLE,
                    text=tbl_md,
                    token_count=tok,
                    char_count=len(tbl_md),
                    provenance=ChunkProvenance(
                        page_start=page_no,
                        page_end=page_no,
                        page_range=[page_no],
                        citation=citation,
                        bbox=elem.get("bbox")
                    ),
                    toc=TOCContext(
                        toc_section=toc_info["active_title"],
                        toc_parent=toc_info["parent"],
                        toc_path=toc_info["path"],
                        toc_level=toc_info["level"],
                        toc_bookmark_index=b_idx
                    ),
                    navigation=GraphNavigation(),
                    inlined_footnotes=absorbed_footnotes,
                    filter_metadata={"is_table_part": r_idx + 1, "total_table_parts": len(row_chunks)}
                ))
        elif modality == "picture":
            pic_idx = elem.get("ref", "").split("/")[-1]
            cls_info = detailed_image_summaries.get(pic_idx, {})
            if not cls_info.get("is_decorative") and cls_info.get("rag_action") != "discard":
                title = cls_info.get("title") or f"Visual Figure (Page {page_no})"
                category = cls_info.get("category") or "Diagram / Chart"
                detailed_sum = cls_info.get("detailed_summary") or cls_info.get("summary", "")
                takeaway = cls_info.get("business_takeaway", "")
                quant_data = cls_info.get("quantitative_data", [])
                tags = cls_info.get("tags", [category])
                chunk_counter += 1
                c_id = f"chunk_sec{b_idx:03d}_figure_{chunk_counter:04d}"
                fig_text = f"### Visual Figure: {title} ({category}, Page {page_no})\n\n"
                fig_text += f"{detailed_sum}\n\n"
                if takeaway:
                    fig_text += f"**Key Business Takeaway**: {takeaway}\n\n"
                if quant_data:
                    fig_text += f"**Extracted Metrics**: {quant_data}\n\n"
                fig_text += f"**Context**: {toc_info['active_title']} ({' > '.join(toc_info['path'])})"
                tok = approximate_token_count(fig_text)
                citation = f"JPMC 2025 Annual Report, p. {page_no} ({' > '.join(toc_info['path'])})"
                section_chunks.append(UnifiedChunk(
                    chunk_id=c_id,
                    modality=ChunkModality.FIGURE,
                    text=fig_text,
                    token_count=tok,
                    char_count=len(fig_text),
                    provenance=ChunkProvenance(
                        page_start=page_no,
                        page_end=page_no,
                        page_range=[page_no],
                        citation=citation,
                        bbox=elem.get("bbox")
                    ),
                    toc=TOCContext(
                        toc_section=toc_info["active_title"],
                        toc_parent=toc_info["parent"],
                        toc_path=toc_info["path"],
                        toc_level=toc_info["level"],
                        toc_bookmark_index=b_idx
                    ),
                    navigation=GraphNavigation(),
                    multimodal=MultimodalAnchor(
                        image_path=cls_info.get("path"),
                        visual_category=category,
                        quantitative_metrics=[str(m) for m in quant_data] if isinstance(quant_data, list) else [str(quant_data)],
                        tags=tags
                    )
                ))
        i += 1
    if current_text_run:
        flushed, chunk_counter = emit_text_chunks(current_text_run, current_subheading, current_pages, b_idx, toc_info, chunk_counter)
        section_chunks.extend(flushed)
    sibling_ids = [c.chunk_id for c in section_chunks]
    for s_idx, c in enumerate(section_chunks):
        c.navigation.chunk_index_in_section = s_idx + 1
        c.navigation.total_chunks_in_section = len(section_chunks)
        c.navigation.section_sibling_ids = sibling_ids
    all_chunks.extend(section_chunks)

for idx in range(len(all_chunks)):
    if idx > 0:
        all_chunks[idx].navigation.prev_chunk_id = all_chunks[idx - 1].chunk_id
    if idx < len(all_chunks) - 1:
        all_chunks[idx].navigation.next_chunk_id = all_chunks[idx + 1].chunk_id

jsonl_out = OUTPUT_DIR / "unified_chunks.jsonl"
json_out = OUTPUT_DIR / "unified_chunks.json"

with open(jsonl_out, "w", encoding="utf-8") as f_jl:
    for c in all_chunks:
        f_jl.write(c.model_dump_json() + "\n")

with open(json_out, "w", encoding="utf-8") as f_j:
    json.dump([c.model_dump() for c in all_chunks], f_j, indent=2)

print(f"\nGenerated {len(all_chunks)} chunks total.")
txt = [c for c in all_chunks if c.modality == ChunkModality.TEXT]
toks = [c.token_count for c in txt]
print(f"Text Chunks: {len(txt)}")
print(f"Token stats: min={min(toks)}, max={max(toks)}, avg={sum(toks)/len(toks):.1f}")
large = [c for c in txt if c.token_count > 512]
print(f"Text chunks > 512 tokens: {len(large)}")
if large:
    for c in sorted(large, key=lambda x: x.token_count, reverse=True)[:5]:
        print(f"  - {c.chunk_id}: {c.token_count} tok | {c.toc.toc_section[:40]}")
