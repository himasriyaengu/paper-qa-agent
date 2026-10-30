from dense import search_dense
from search import search_docs as search_bm25

NONE_BELOW, SURE_ABOVE = 0.70, 0.74

def _key(h): return h["source"] + h["text"][:40]

def retrieve(query: str, k: int = 4):
    hits = search_dense(query, k=k)
    top = hits[0]["score"]
    if top < NONE_BELOW:
        return [], "none"
    if top < SURE_ABOVE:
        have = {_key(h) for h in hits}
        hits = hits + [h for h in search_bm25(query, k=2) if _key(h) not in have]
        return hits, "low"
    return hits, "high"
