import ast
from search import search_docs
from dense import search_dense
from hybrid import search_hybrid

def load(path):
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, ast.Assign) and n.targets[0].id == "CASES":
            return ast.literal_eval(n.value)

def score(fn, cases, k=4):
    rr, hits = [], 0
    for q, term in cases:
        found = [i for i, r in enumerate(fn(q, k=k)) if term in r["text"].lower()]
        hits += bool(found)
        rr.append(1 / (found[0] + 1) if found else 0)
    return hits, sum(rr) / len(rr)

for name, path in [("easy", "ret_eval.py"), ("hard", "ret_eval_hard.py")]:
    cases = load(path)
    for label, fn in [("bm25", search_docs), ("dense", search_dense), ("hybrid", search_hybrid)]:
        h, mrr = score(fn, cases)
        print(f"{name:5} {label:7} hit@4 {h}/{len(cases)}  MRR {mrr:.2f}")
