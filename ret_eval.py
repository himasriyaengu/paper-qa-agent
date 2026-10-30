from search import search_docs

K = 4
# (question, a term that must appear in the top-K chunks if retrieval worked)
CASES = [
    ("What is scaled dot-product attention?", "scaled dot-product"),
    ("How many identical layers are in the Transformer encoder?", "identical layers"),
    ("Which optimizer was used to train the Transformer?", "adam"),
    ("What positional encoding does the Transformer use?", "sinusoid"),
    ("What does the ReAct paradigm combine?", "reasoning traces"),
    ("Which interactive benchmarks was ReAct tested on?", "alfworld"),
    ("Which knowledge source does ReAct query for question answering?", "wikipedia"),
    ("What are the two RAG formulations?", "rag-token"),
    ("Which retriever does RAG use?", "dpr"),
    ("What is the non-parametric memory in RAG?", "non-parametric"),
    ("What components make up Reflexion?", "self-reflection"),
    ("What pass@1 did Reflexion reach on HumanEval?", "91"),
]

hits = 0
for q, term in CASES:
    results = search_docs(q, k=K)
    found = [i for i, r in enumerate(results) if term in r["text"].lower()]
    ok = bool(found)
    hits += ok
    rank = f"rank {found[0] + 1}" if ok else "MISS"
    print(f"{'HIT ' if ok else 'MISS'} ({rank}) {q}")
print(f"\nBM25 hit@{K}: {hits}/{len(CASES)}")
