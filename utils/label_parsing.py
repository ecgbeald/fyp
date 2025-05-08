import ast
import re


def parse_label_string(s):  # parses "[1,2,3]", "4", "2,3", etc.
    s = s.strip()
    if not s:
        return [0]

    try:
        # Try parsing as a literal (handles "[1,2,3]", "4", etc.)
        parsed = ast.literal_eval(s)
        if isinstance(parsed, int):
            return [parsed]
        elif isinstance(parsed, list):
            if not parsed:
                return [0]
            return [int(x) for x in parsed]
    except Exception:
        pass

    # Handle comma-separated values like "2,3"
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return []


def multi_label(response):
    matched = False
    for line in response.split("\n"):
        if "reason" in line:
            matched = True
            match = re.search(r'"reason":\s*"([^"]*)"', line)
            if match:
                reason_str = match.group(1)
                return parse_label_string(reason_str)
            else:
                return [0]
    if not matched:
        return [0]


def parse_explain(response):
    matched = False
    for line in response.split("\n"):
        if "explanation" in line:
            matched = True
            match = re.search(r'"explanation":\s*"([^}]*)"', line)
            if match:
                reason_str = match.group(1)
                return reason_str
            else:
                return ""
    if not matched:
        return ""
