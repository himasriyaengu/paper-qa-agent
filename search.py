import json, re
from rank_bm25 import BM25Okapi

chunks = json.load(open("chunks.json"))
tok = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in {"the","a","an","of","to","in","is","are","was","what","which","how","does","do","did","and","or","for","on","with","by","at","as","it","its","this","that"}]
bm25 = BM25Okapi([tok(c["text"]) for c in chunks])

def search_docs(query: str, k: int = 4):
    """Search the paper collection. Returns the top matching passages with sources."""
    scores = bm25.get_scores(tok(query))
    top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"source": f'{chunks[i]["doc"]} p.{chunks[i]["page"]}',
             "score": round(float(scores[i]), 2),
             "text": chunks[i]["text"]} for i in top]
