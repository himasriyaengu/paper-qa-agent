import json, time

class _T:
    def reset(self):
        self.llm_calls = self.embed_calls = 0
        self.in_tok = self.out_tok = self.total = 0
        self.llm_s = self.embed_s = 0.0
T = _T(); T.reset()
_installed = False

def install():
    global _installed
    if _installed:
        return
    from agent import client
    from embed import client as eclient
    gen, emb = client.models.generate_content, eclient.models.embed_content
    def g(*a, **kw):
        t = time.time(); r = gen(*a, **kw)
        T.llm_s += time.time() - t; T.llm_calls += 1
        u = getattr(r, "usage_metadata", None)
        if u:
            T.in_tok += u.prompt_token_count or 0
            T.out_tok += u.candidates_token_count or 0
            T.total += u.total_token_count or 0
        return r
    def e(*a, **kw):
        t = time.time(); r = emb(*a, **kw)
        T.embed_s += time.time() - t; T.embed_calls += 1
        return r
    client.models.generate_content, eclient.models.embed_content = g, e
    _installed = True

def tracked_run(question):
    install()
    from rag import rag_run
    T.reset(); t0 = time.time(); err = None
    try:
        answer, seen, conf, info = rag_run(question)
    except Exception as ex:
        answer, conf, info, err = "", "error", {}, type(ex).__name__
    review = info.get("critique") or ("degraded" if info.get("degraded") else "-")
    rec = {"q": question, "latency_s": round(time.time() - t0, 2),
           "llm_calls": T.llm_calls, "searches": T.embed_calls,
           "in_tok": T.in_tok, "out_tok": T.out_tok, "total_tok": T.total,
           "llm_s": round(T.llm_s, 2), "embed_s": round(T.embed_s, 2),
           "confidence": conf, "review": review, "error": err}
    with open("metrics.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return answer, rec
