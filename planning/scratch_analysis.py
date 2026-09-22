import pymupdf

doc = pymupdf.open('input/jpmc_annualreport-2025.pdf')

# Check which pages 85-86 have tables that pymupdf misses
# Look at page 85 raw text structure - it likely has a table
page = doc[84]
text = page.get_text()
print("=== Page 85 full text ===")
print(text[:600])
print()

# Check notes pages (202+) for table detection
print("=== Notes to Financial Statements table detection ===")
notes_with_tables = 0
notes_without_tables = 0
notes_with_dollars = 0
for pno in range(201, 347):
    page = doc[pno]
    tabs = page.find_tables()
    text = page.get_text()
    has_dollars = '$' in text
    if tabs.tables:
        notes_with_tables += 1
    elif has_dollars:
        notes_without_tables += 1
        notes_with_dollars += 1
    else:
        notes_without_tables += 1

print(f'  Pages 202-347 (Notes): pages with tables detected={notes_with_tables}')
print(f'  Pages with $ but no table detected={notes_with_dollars}')
print(f'  Total pages without table detection={notes_without_tables}')

# Check a specific notes page for quality
print()
print("=== Sample Notes page (Page 210) ===")
page = doc[209]
text = page.get_text()
print(text[:500])
print()

# Check the glossary section
print("=== Glossary section (Pages 352-359) ===")
page = doc[351]
text = page.get_text()
print(f'Page 352 preview: {text[:200]!r}')

# Check a senior exec letter page with potential 3-column layout
print()
print("=== Senior Exec Letter 3-column test (Page 52) ===")
page = doc[51]
blocks = page.get_text("blocks")
text_blocks = [b for b in blocks if b[6] == 0]
x_starts = sorted(set(round(b[0]) for b in text_blocks))
print(f'  X-coordinate starts: {x_starts}')
print(f'  Suggests {len([x for x in x_starts if x < 500])} distinct column origins')

# Render a chart page to see dimensions
print()
print("=== Page rendering dimensions ===")
page = doc[7]
mat = pymupdf.Matrix(2, 2)  # 2x zoom = ~144 DPI
pix = page.get_pixmap(matrix=mat)
print(f'  Page 8 at 2x zoom: {pix.width}x{pix.height} pixels')
mat3 = pymupdf.Matrix(3, 3)  # 3x zoom = ~216 DPI
pix3 = page.get_pixmap(matrix=mat3)
print(f'  Page 8 at 3x zoom: {pix3.width}x{pix3.height} pixels')

# Check how many pages have ONLY drawings and very little text (pure chart pages)
print()
print("=== Pure chart pages (drawings > 50, text_len < 2000) ===")
pure_chart_pages = []
for pno in range(len(doc)):
    page = doc[pno]
    draws = page.get_drawings()
    text = page.get_text()
    if len(draws) > 50 and len(text) < 2000:
        pure_chart_pages.append(pno + 1)
print(f'  Found {len(pure_chart_pages)} pure chart pages: {pure_chart_pages}')
