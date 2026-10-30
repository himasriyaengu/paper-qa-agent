import json, time
import numpy as np
from google import genai
from google.genai import types

client = genai.Client(http_options=types.HttpOptions(timeout=60000))
EMBED_MODEL = "gemini-embedding-001"

def embed(texts, task):
    out = []
    for i in range(0, len(texts), 50):
        for attempt in range(4):
            try:
                resp = client.models.embed_content(
                    model=EMBED_MODEL, contents=texts[i:i + 50],
                    config=types.EmbedContentConfig(task_type=task))
                break
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    time.sleep(20 * (attempt + 1))
                    continue
                raise
        out += [e.values for e in resp.embeddings]
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

if __name__ == "__main__":
    chunks = json.load(open("chunks.json"))
    vecs = embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    np.save("chunk_vecs.npy", vecs)
    print("saved", vecs.shape)
