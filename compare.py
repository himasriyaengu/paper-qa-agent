import ast
from search import search_docs
from dense import search_dense

def load(path):
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, ast.Assign) and n.targets[0].id == "CASES":
            return ast.literal_eval(n.value)

def ranks(fn, cases, k=4):
    out = []
    for q, term in cases:
        found = [i for i, r in enumerate(fn(q, k=k)) if term in r["text"].lower()]
        out.append(found[0] + 1 if found else None)
    return out

for name, path in [("easy", "ret_eval.py"), ("hard", "ret_eval_hard.py")]:
    cases = load(path)
    b, d = ranks(search_docs, cases), ranks(search_dense, cases)
    print(f"\n{name} set (rank of first matching chunk, '-' = miss)")
    for (q, _), rb, rd in zip(cases, b, d):
        print(f"  bm25={rb or '-'}  dense={rd or '-'}  {q[:70]}")
    print(f"  hit@4   bm25 {sum(r is not None for r in b)}/{len(cases)}   dense {sum(r is not None for r in d)}/{len(cases)}")
