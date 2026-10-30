import ast, operator, pathlib

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")

def calculate(expression: str):
    """Evaluate basic arithmetic like '12 * (3 + 4)'."""
    return _eval(ast.parse(expression, mode="eval").body)

def read_file(path: str) -> str:
    """Read a text file from the data/ folder."""
    base = pathlib.Path("data").resolve()
    target = (base / path).resolve()
    if base not in target.parents:
        raise ValueError("access outside data/ is not allowed")
    return target.read_text()[:5000]

TOOLS = {"calculate": calculate, "read_file": read_file}

DECLARATIONS = [
    {"name": "calculate",
     "description": "Evaluate a basic arithmetic expression.",
     "parameters": {"type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]}},
    {"name": "read_file",
     "description": "Read a text file from the data folder by file name.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]}},
]
