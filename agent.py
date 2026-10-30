import logging, os
from google import genai
from google.genai import types
from tools import TOOLS, DECLARATIONS

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.FileHandler("agent.log"), logging.StreamHandler()])
log = logging.getLogger("agent")

MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash-lite")
MAX_STEPS = 6
client = genai.Client(http_options=types.HttpOptions(timeout=30000))

config = types.GenerateContentConfig(
    system_instruction="You are a careful assistant. Use tools when needed. "
                       "If a tool returns an error, fix your input or explain the problem.",
    tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**d) for d in DECLARATIONS])],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

def run(question: str) -> str:
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    for step in range(1, MAX_STEPS + 1):
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        contents.append(resp.candidates[0].content)
        if not resp.function_calls:
            log.info(f"[step {step}] final answer")
            return resp.text
        parts = []
        for call in resp.function_calls:
            log.info(f"[step {step}] call {call.name}({dict(call.args)})")
            try:
                out = {"result": TOOLS[call.name](**call.args)}
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
            log.info(f"[step {step}] result {out}")
            parts.append(types.Part.from_function_response(name=call.name, response=out))
        contents.append(types.Content(role="user", parts=parts))
    return "Stopped: reached the step limit without a final answer."

if __name__ == "__main__":
    print(run(input("Ask: ")))
