from search import search_docs

K = 4
CASES = [
    ("How does the architecture know the order of tokens when it has no recurrence?", "sinusoid"),
    ("Which learning algorithm updates the weights when training the original translation model?", "adam"),
    ("Which household-task simulator was used to test the reasoning-and-acting approach?", "alfworld"),
    ("Which neural bi-encoder finds relevant passages in the generator-plus-search-index system?", "dpr"),
    ("How does the agent turn failed attempts into lessons it can reuse on later tries?", "self-reflection"),
    ("What fraction of coding problems did the self-improving agent solve on its first try?", "91"),
    ("What lets the generator use a different retrieved document for each output word?", "rag-token"),
    ("Why does combining reasoning with tool use make fewer made-up facts?", "hallucination"),
]

hits = 0
for q, term in CASES:
    results = search_docs(q, k=K)
    found = [i for i, r in enumerate(results) if term in r["text"].lower()]
    hits += bool(found)
    print(f"{'HIT ' if found else 'MISS'} ({'rank ' + str(found[0] + 1) if found else 'MISS'}) {q}")
print(f"\nBM25 hard set hit@{K}: {hits}/{len(CASES)}")
