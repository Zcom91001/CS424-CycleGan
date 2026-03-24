from pathlib import Path


def _parse_scalar(value: str):
    v = value.strip()
    if not v:
        return ""
    lower = v.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        pass
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v


def load_yaml(path):
    text = Path(path).read_text(encoding="utf-8")
    root = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            continue
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if value == "":
            node = {}
            current[key] = node
            stack.append((indent, node))
        else:
            current[key] = _parse_scalar(value)

    return root
