import os, glob, json
from pypdf import PdfReader

reports_dir = r"d:\nckh\Reports"
pdf_files = sorted(glob.glob(os.path.join(reports_dir, "**", "*.pdf"), recursive=True))

analysis = {}

for p in pdf_files:
    rel = os.path.relpath(p, reports_dir)
    fname = os.path.basename(rel)
    reader = PdfReader(p)
    num_pages = len(reader.pages)
    
    # Read text across first 3 pages and last page (conclusion)
    intro_text = ""
    for i in range(min(3, num_pages)):
        intro_text += "\n" + (reader.pages[i].extract_text() or "")
    
    concl_text = ""
    if num_pages > 3:
        concl_text = reader.pages[-1].extract_text() or ""
        if len(concl_text) < 300 and num_pages > 4:
            concl_text = (reader.pages[-2].extract_text() or "") + "\n" + concl_text
            
    analysis[rel] = {
        "rel": rel,
        "filename": fname,
        "pages": num_pages,
        "intro_preview": " ".join(intro_text.split())[:1500],
        "concl_preview": " ".join(concl_text.split())[:1000]
    }

with open(r"d:\nckh\Reports\deep_analysis.json", "w", encoding="utf-8") as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)

print(f"Deep analysis extracted for {len(analysis)} papers.")
