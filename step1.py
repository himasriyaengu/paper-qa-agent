from google import genai
from google.genai import types

client = genai.Client()
MODEL = "gemini-3.5-flash-lite"
MAX_STEPS = 5

def get_word_count(text: str) -> int:
    return len(text.split())

TOOLS = {"get_word_count": get_word_count}

decl = types.FunctionDeclaration(
    name="get_word_count",
    description="Count the words in a piece of text.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
config = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[decl])],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

contents = [types.Content(role="user", parts=[types.Part(
    text="How many words are in: 'agents are just loops'?")])]

for step in range(MAX_STEPS):
    resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
    contents.append(resp.candidates[0].content)
    if not resp.function_calls:
        print(resp.text)
        break
    parts = []
    for call in resp.function_calls:
        result = TOOLS[call.name](**call.args)
        parts.append(types.Part.from_function_response(
            name=call.name, response={"result": result}))
    contents.append(types.Content(role="user", parts=parts))
else:
    print("Stopped: hit max steps without a final answer.")
