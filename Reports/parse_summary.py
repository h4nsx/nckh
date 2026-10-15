import os, json

with open(r'd:\nckh\Reports\extracted_meta.json', encoding='utf-8') as f:
    data = json.load(f)

with open(r'd:\nckh\Reports\summary_list.txt', 'w', encoding='utf-8') as out:
    for i, item in enumerate(data):
        folder = item.get('folder', '')
        fname = item.get('filename', '')
        pages = item.get('pages', 0)
        abstract = " ".join(item.get('abstract', '').split())
        out.write(f"[{i+1}] [{folder}] {fname} ({pages} pages)\n")
        out.write(f"    Abstract: {abstract[:400]}\n\n")

print(f"Summary generated for {len(data)} papers.")
