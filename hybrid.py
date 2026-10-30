from search import search_docs
from dense import search_dense

def search_hybrid(query: str, k: int = 4, pool: int = 10, c: int = 60):
    scores, keep = {}, {}
    for fn in (search_docs, search_dense):
        for rank, r in enumerate(fn(query, k=pool), start=1):
            key = r["source"] + r["text"][:40]
            scores[key] = scores.get(key, 0) + 1 / (c + rank)
            keep[key] = r
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**keep[t], "score": round(scores[t], 4)} for t in top]
