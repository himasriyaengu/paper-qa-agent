import json
import numpy as np
from embed import embed

_chunks = json.load(open("chunks.json"))
_vecs = np.load("chunk_vecs.npy")

def search_dense(query: str, k: int = 4):
    qv = embed([query], "RETRIEVAL_QUERY")[0]
    scores = _vecs @ qv
    top = np.argsort(-scores)[:k]
    return [{"source": f'{_chunks[i]["doc"]} p.{_chunks[i]["page"]}',
             "score": round(float(scores[i]), 3),
             "text": _chunks[i]["text"]} for i in top]
