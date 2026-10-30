import ast
from dense import search_dense

def load(path):
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, ast.Assign) and n.targets[0].id == "CASES":
            return [q for q, _ in ast.literal_eval(n.value)]

IN = load("ret_eval.py") + load("ret_eval_hard.py")
OUT = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the 2018 football world cup?",
    "How does AlphaFold predict protein structures?",
    "What learning rate is used to train ResNet-50 on ImageNet?",
    "What is the boiling point of water at sea level?",
    "How do I fix a flat bicycle tire?",
    "What is the GDP of Brazil?",
]

def top(q): return search_dense(q, k=1)[0]["score"]
a = sorted(top(q) for q in IN)
b = sorted(top(q) for q in OUT)
print("in-scope  top-1 scores: min %.3f  median %.3f" % (a[0], a[len(a)//2]))
print("out-scope top-1 scores: median %.3f  max %.3f" % (b[len(b)//2], b[-1]))
print("lowest in-scope:", [round(x, 3) for x in a[:4]])
print("highest out-of-scope:", [round(x, 3) for x in b[-4:]])
