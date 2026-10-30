import logging
from pydantic import BaseModel, Field, ValidationError
from google.genai import types
from agent import run, client, MODEL

log = logging.getLogger("agent")

class Answer(BaseModel):
    answer: str = Field(description="Short final answer in plain text")
    value: float | None = Field(default=None, description="Main numeric result, if there is one")
    refused: bool = Field(description="True if the request could not be completed")
    confidence: float = Field(ge=0, le=1, description="0 to 1")

def structured_run(question: str, retries: int = 2) -> Answer:
    raw = run(question)
    prompt = f"Question: {question}\nAgent answer: {raw}\nExtract the result."
    last_err = None
    for attempt in range(retries + 1):
        resp = client.models.generate_content(
            model=MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Answer))
        try:
            return Answer.model_validate_json(resp.text)
        except ValidationError as e:
            last_err = e
            log.info(f"[validation] attempt {attempt + 1} failed: {e.error_count()} errors")
            prompt += f"\nYour previous output was invalid: {e}. Fix it."
    raise last_err

if __name__ == "__main__":
    print(structured_run(input("Ask: ")).model_dump_json(indent=2))
