import ast, time, logging
logging.disable(logging.CRITICAL)
from metrics import tracked_run

def load(path):
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, ast.Assign) and n.targets[0].id == "CASES":
            return [q for q, _ in ast.literal_eval(n.value)]

OUT = ["What is the capital of France?", "How do I bake sourdough bread?",
       "Who won the 2018 football world cup?", "How does AlphaFold predict protein structures?",
       "What learning rate is used to train ResNet-50 on ImageNet?",
       "What is the boiling point of water at sea level?",
       "How do I fix a flat bicycle tire?", "What is the GDP of Brazil?"]

QS = load("ret_eval.py") + load("ret_eval_hard.py") + OUT
for i, q in enumerate(QS, 1):
    for attempt in range(3):
        answer, rec = tracked_run(q)
        if rec["error"] == "ClientError" and attempt < 2:
            time.sleep(30)
            continue
        break
    print(f"{i:2}/{len(QS)} {rec['latency_s']:5.1f}s calls={rec['llm_calls']} conf={rec['confidence']:5} {rec['error'] or ''} {q[:50]}")
    time.sleep(3)
