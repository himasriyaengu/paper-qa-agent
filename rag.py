import logging, re
from google.genai import types
from agent import client, MODEL, MAX_STEPS
from dense import search_dense as search_docs

log = logging.getLogger("agent")

SYSTEM = ("Answer only from the search_docs tool results, never from outside knowledge. "
          "Search first (you may search more than once), then answer in 2 to 4 sentences "
          "and cite sources like [react p.3]. If the results do not contain the answer, "
          "say you could not find it in the documents.")

decl = types.FunctionDeclaration(
    name="search_docs",
    description="Search the paper collection (Attention, ReAct, RAG, Reflexion). "
                "Returns top passages with their sources.",
    parameters={"type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]},
)
config = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=[types.Tool(function_declarations=[decl])],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

def rag_run(question: str):
    seen, contents = set(), [types.Content(role="user", parts=[types.Part(text=question)])]
    for step in range(1, MAX_STEPS + 1):
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        contents.append(resp.candidates[0].content)
        if not resp.function_calls:
            return resp.text, seen
        parts = []
        for call in resp.function_calls:
            q = (call.args or {}).get("query", "")
            log.info(f"[step {step}] search_docs({q!r})")
            hits = search_docs(q)
            seen.update(h["source"] for h in hits)
            parts.append(types.Part.from_function_response(
                name=call.name, response={"results": hits}))
        contents.append(types.Content(role="user", parts=parts))
    return "Stopped: reached the step limit.", seen

def check_citations(text: str, seen: set):
    cited = {f"{d} p.{p}" for d, p in re.findall(r"\b([a-z]+) p\.(\d+)", text.lower())}
    return cited, sorted(c for c in cited if c not in seen)

if __name__ == "__main__":
    answer, seen = rag_run(input("Ask: "))
    cited, bad = check_citations(answer, seen)
    print("\n" + answer)
    print(f"\nCitations: {len(cited)} cited, {len(bad)} not found in retrieved results {bad}")
