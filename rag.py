import logging
from google.genai import types
from agent import client, MODEL, MAX_STEPS
from guard import retrieve
from citations import parse_cites

log = logging.getLogger("agent")

SYSTEM = ("Answer only from the search_docs tool results, never from outside knowledge. "
          "Search first (you may search more than once), then answer in 2 to 4 sentences "
          "and cite sources like [react p.3]. Each result set has a confidence field. "
          "If confidence is 'none', do not answer from memory: say you could not find it "
          "in the documents. If it is 'low', say your answer is uncertain. "
          "If the results do not contain the answer, say you could not find it.")

decl = types.FunctionDeclaration(
    name="search_docs",
    description="Search the paper collection (Attention, ReAct, RAG, Reflexion). "
                "Returns top passages with their sources and a confidence level.",
    parameters={"type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]},
)
config = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=[types.Tool(function_declarations=[decl])],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

from critique import review, degrade

ORDER = {"high": 0, "low": 1, "none": 2}

def _key(h): return h["source"] + h["text"][:40]

def rag_run(question: str):
    seen, best, evidence = set(), "none", {}
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    for step in range(1, MAX_STEPS + 1):
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        contents.append(resp.candidates[0].content)
        if not resp.function_calls:
            final, info = review(question, resp.text, list(evidence.values()), best)
            return final, seen, best, info
        parts = []
        for call in resp.function_calls:
            q = (call.args or {}).get("query", "")
            hits, conf = retrieve(q)
            log.info(f"[step {step}] search_docs({q!r}) confidence={conf}")
            seen.update(h["source"] for h in hits)
            evidence.update({_key(h): h for h in hits})
            if ORDER[conf] < ORDER[best]:
                best = conf
            parts.append(types.Part.from_function_response(
                name=call.name, response={"confidence": conf, "results": hits}))
        contents.append(types.Content(role="user", parts=parts))
    log.info("[degrade] step limit reached, answering from evidence so far")
    return degrade(question, list(evidence.values())), seen, best, {"degraded": True}

def check_citations(text: str, seen: set):
    cited = parse_cites(text)
    return cited, sorted(c for c in cited if c not in seen)

if __name__ == "__main__":
    answer, seen, conf, info = rag_run(input("Ask: "))
    cited, bad = check_citations(answer, seen)
    print("\n" + answer)
    print(f"\nRetrieval confidence: {conf}   Review: {info}")
    print(f"Citations: {len(cited)} cited, {len(bad)} not found in retrieved results {bad}")
