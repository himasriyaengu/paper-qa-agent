import re

def parse_cites(text: str) -> set:
    cited = set()
    for group in re.findall(r"\[([^\]]+)\]", text.lower()):
        doc = None
        for part in group.split(","):
            part = part.strip()
            m = re.match(r"([a-z]+) p\.(\d+)", part)
            if m:
                doc = m.group(1)
                cited.add(f"{doc} p.{m.group(2)}")
            elif doc and (n := re.fullmatch(r"(?:p\.)?(\d+)", part)):
                cited.add(f"{doc} p.{n.group(1)}")
    return cited
