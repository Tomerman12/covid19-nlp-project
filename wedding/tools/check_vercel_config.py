"""Validate vercel.json before it reaches a deploy.

This exists because a `"//"` key was once used to hold a comment inside a
headers rule. JSON has no comments, Vercel's schema rejects any property it
does not know, and the deploy failed at build time with

    headers[0] should NOT have additional property `//`

The local test server at the time skipped that key, so it was more permissive
than the real validator and the mistake sailed through. This encodes the
constraint that actually failed: nothing anywhere in the file may carry a
property the schema does not define.

    python3 tools/check_vercel_config.py

Exits non-zero and prints the offending path on failure. Vercel's published
schema lives at https://openapi.vercel.sh/vercel.json and is the authority;
this is a strict local subset covering the keys this project uses.
"""
import json
import pathlib
import sys

from jsonschema import Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT / "vercel.json"

HEADER = {
    "type": "object",
    "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
    "required": ["key", "value"],
    "additionalProperties": False,
}

RULE = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "headers": {"type": "array", "items": HEADER, "minItems": 1},
        "has": {"type": "array"},
        "missing": {"type": "array"},
    },
    "required": ["source", "headers"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "$schema": {"type": "string"},
        "headers": {"type": "array", "items": RULE},
        "redirects": {"type": "array"},
        "rewrites": {"type": "array"},
        "cleanUrls": {"type": "boolean"},
        "trailingSlash": {"type": "boolean"},
    },
    "additionalProperties": False,
}


def main() -> int:
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"vercel.json is not valid JSON: {e}")
        return 1

    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(config), key=lambda e: list(e.path))
    for e in errors:
        where = "".join(f"[{p!r}]" if isinstance(p, str) else f"[{p}]" for p in e.path) or "(root)"
        print(f"  {where}  {e.message}")
    if errors:
        print(f"\n{len(errors)} problem(s) in {CONFIG.relative_to(ROOT.parent)}")
        return 1

    rules = config.get("headers", [])
    keys = sum(len(r["headers"]) for r in rules)
    print(f"vercel.json valid: {len(rules)} header rule(s), {keys} header(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
