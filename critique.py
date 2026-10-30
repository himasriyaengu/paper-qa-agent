import logging, time
from pydantic import BaseModel, Field
from google.genai import types
from agent import client, MODEL

log = logging.getLogger("agent")

class Critique(BaseModel):
    supported: bool = Field(description="True if every claim is backed by the passages")
    unsupported_claims: list[str]

def _passages(evidence):
    return "\n\n".join(f"[{h['source']}] {h['text']}" for h in evidence)

def _ask(prompt, schema=None):
    cfg = None
    if schema:
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema)
    for attempt in range(3):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
        except Exception as e:
            log.info(f"[critique] call failed ({type(e).__name__}), attempt {attempt + 1}")
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))


def review(question, answer, evidence, confidence):
    if confidence == "none" or not evidence:
        return answer, {"critique": "skipped"}
    prompt = (f"Passages:\n{_passages(evidence)}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
              "Check every factual claim in the answer against the passages only. "
              "List any claim the passages do not support.")
    try:
        c = Critique.model_validate_json(_ask(prompt, Critique).text)
    except Exception as e:
        log.info(f"[critique] failed: {type(e).__name__}: {str(e)[:150]}")
        return answer, {"critique": "error"}
    log.info(f"[critique] supported={c.supported} unsupported={c.unsupported_claims}")
    if c.supported or not c.unsupported_claims:
        return answer, {"critique": "passed"}
    fix = (f"Passages:\n{_passages(evidence)}\n\nQuestion: {question}\nDraft answer: {answer}\n"
           f"Unsupported claims: {c.unsupported_claims}\n"
           "Rewrite the answer using only the passages. Remove or soften the unsupported "
           "claims and keep citations like [react p.3].")
    return _ask(fix).text, {"critique": "revised", "removed": c.unsupported_claims}

def degrade(question, evidence):
    if not evidence:
        return "I ran out of search steps and found nothing relevant."
    prompt = (f"Passages:\n{_passages(evidence)}\n\nQuestion: {question}\n"
              "I ran out of search steps. Give the best partial answer using only these "
              "passages, cite sources like [react p.3], keep it under 120 words with no headings, and state in one sentence what is still missing.")
    return _ask(prompt).text
