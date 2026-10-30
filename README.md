# Paper Q&A Agent

A small question-answering agent over four ML papers (Attention Is All You Need, ReAct, Retrieval-Augmented Generation, Reflexion). The agent loop is written by hand against the Gemini API, with no agent framework, so every step is visible and logged.

It answers only from retrieved passages, cites sources as `[doc p.N]`, checks in code that every citation was actually retrieved, and refuses when the documents do not contain the answer.

## What is in the repo

| File | Purpose |
|---|---|
| `agent.py`, `tools.py` | Basic tool-calling loop: step limit, tool errors returned to the model, step-by-step log. Tools: calculator, sandboxed file reader. |
| `structured.py` | Validated JSON output with Pydantic, logging and retrying on validation failure. |
| `ingest.py` | Splits PDF pages into overlapping 180-word chunks (`chunks.json`). |
| `search.py` | BM25 keyword retrieval with stopword filtering. |
| `embed.py`, `dense.py` | Embedding retrieval (`gemini-embedding-001`), vectors cached to disk. |
| `hybrid.py` | BM25 + dense merged with reciprocal rank fusion. |
| `rag.py` | The RAG agent: search tool, grounded answers, citation check. Uses dense retrieval. |
| `citations.py` | Parses citations, including shorthand like `[reflexion p.4, 6]`. |
| `ret_eval.py`, `ret_eval_hard.py`, `report.py` | Retrieval evaluation and comparison report. |
| `test_structured.py` | Unit tests (`python -m pytest -q`). |

## Retrieval results

Relevance label: a chunk counts as relevant if it contains a known key term from the answer. "Easy" questions reuse the paper's wording. "Hard" questions are paraphrased so they share few words with the text. hit@4 means the relevant chunk appeared in the top 4. MRR is mean reciprocal rank (higher means the right chunk ranks nearer the top).

| Retriever | Easy hit@4 | Easy MRR | Hard hit@4 | Hard MRR |
|---|---|---|---|---|
| BM25 | 12/12 | 0.78 | 6/8 | 0.51 |
| Dense embeddings | 12/12 | 0.94 | 7/8 | 0.62 |
| Hybrid (RRF) | 12/12 | 0.96 | 4/8 | 0.44 |

Findings:

- Dense retrieval ranked the right chunk higher than BM25 on both sets, so the agent uses it.
- Hybrid fusion helped on the easy set and hurt on the hard set. A likely cause is that BM25 returns mostly noise for paraphrased queries and rank fusion lets that noise outvote a good dense result. I did not verify this by inspecting individual queries.
- BM25 returned junk pages (repeated attention-visualization text) until stopwords were removed. Stopword filtering took the easy set from 11/12 to 12/12.

## Limitations

- The evaluation is small (12 easy and 8 hard questions) and written by me, so differences of one or two questions are within noise. This is a small experiment, not a benchmark.
- The stopword list was adjusted while looking at the easy set, so the easy set is not an independent test of that change.
- The relevance label is a keyword match, which can mark a chunk wrong when it describes the answer in different words.
- The citation check confirms that a cited page was among the retrieved results. It does not confirm that the page supports the claim.
- Refusal behavior was checked on one out-of-scope question, not a test set.
- Dense search always returns something, so refusing relies on the model noticing that the passages do not answer the question.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install google-genai pydantic pypdf rank_bm25 numpy pytest
export GEMINI_API_KEY=your_key      # free tier from Google AI Studio
bash get_docs.sh                    # downloads the four PDFs from arXiv
python ingest.py                    # build chunks.json
python embed.py                     # embed chunks once, saves chunk_vecs.npy
python rag.py                       # ask questions
python report.py                    # reproduce the results table
python -m pytest -q
```

Models used: `gemini-3.5-flash-lite` (set `AGENT_MODEL` to override) and `gemini-embedding-001`. Model availability changes over time, so these names may need updating. On the free tier, Google may use inputs and outputs to improve its products. The documents here are public papers.

## Possible next steps

- Larger, independently written evaluation set with page-level relevance labels.
- Test whether weighting dense results higher in fusion fixes the hybrid regression, using questions not used for tuning.
- Check that each cited passage supports the claim it is attached to, for example with a second model call.
- Measure refusal rate on a set of out-of-scope questions.
