import json, statistics as st
from collections import Counter

rows = [json.loads(l) for l in open("metrics.jsonl")]
ok = [r for r in rows if not r["error"]]

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]

lat = [r["latency_s"] for r in ok]
print(f"runs: {len(rows)}  errors: {len(rows) - len(ok)}")
print(f"latency s   p50 {st.median(lat):.1f}  p95 {pct(lat, .95):.1f}  max {max(lat):.1f}")
print(f"time in LLM calls: {100 * sum(r['llm_s'] for r in ok) / sum(lat):.0f}%   in embedding calls: {100 * sum(r['embed_s'] for r in ok) / sum(lat):.0f}%")
print(f"LLM calls/question  mean {st.mean(r['llm_calls'] for r in ok):.1f}")
print(f"searches/question   mean {st.mean(r['searches'] for r in ok):.1f}")
print(f"tokens/question     in {st.mean(r['in_tok'] for r in ok):.0f}  out {st.mean(r['out_tok'] for r in ok):.0f}  total {st.mean(r['total_tok'] for r in ok):.0f}")
print("confidence:", dict(Counter(r["confidence"] for r in ok)))
print("review:", dict(Counter(r["review"] for r in ok)))
