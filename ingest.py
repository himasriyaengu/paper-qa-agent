import json, pathlib
from pypdf import PdfReader

CHUNK_WORDS, OVERLAP = 180, 30
chunks = []
for pdf in sorted(pathlib.Path("docs").glob("*.pdf")):
    reader = PdfReader(pdf)
    for page_no, page in enumerate(reader.pages, start=1):
        words = (page.extract_text() or "").split()
        step = CHUNK_WORDS - OVERLAP
        for i in range(0, max(len(words), 1), step):
            piece = " ".join(words[i:i + CHUNK_WORDS])
            if len(piece.split()) < 20:
                continue
            chunks.append({"id": len(chunks), "doc": pdf.stem,
                           "page": page_no, "text": piece})
pathlib.Path("chunks.json").write_text(json.dumps(chunks))
print(f"{len(chunks)} chunks from {len(set(c['doc'] for c in chunks))} documents")
