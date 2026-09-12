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

# Two configs, because which one Vercel reads depends on the project's Root
# Directory setting, which lives in the dashboard and cannot be read from here.
# Root Directory "wedding" makes it wedding/vercel.json; the default "/" makes
# it the one at the repository root, whose outputDirectory points back here.
# Both have to be valid, so both are checked.
CONFIGS = [ROOT.parent / "vercel.json", ROOT / "vercel.json"]

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
        "outputDirectory": {"type": "string"},
        "headers": {"type": "array", "items": RULE},
        "redirects": {"type": "array"},
        "rewrites": {"type": "array"},
        "cleanUrls": {"type": "boolean"},
        "trailingSlash": {"type": "boolean"},
    },
    "additionalProperties": False,
}


def check(path: pathlib.Path) -> int:
    name = path.relative_to(ROOT.parent)
    if not path.exists():
        print(f"  {name} is missing")
        return 1
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"  {name} is not valid JSON: {e}")
        return 1

    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(config), key=lambda e: list(e.path))
    for e in errors:
        where = "".join(f"[{p!r}]" if isinstance(p, str) else f"[{p}]" for p in e.path) or "(root)"
        print(f"  {name}{where}  {e.message}")
    if errors:
        return len(errors)

    rules = config.get("headers", [])
    keys = sum(len(r["headers"]) for r in rules)
    out = config.get("outputDirectory")
    extra = f", serving {out}/" if out else ""
    print(f"  {name} valid: {len(rules)} header rule(s), {keys} header(s){extra}")
    return 0


def main() -> int:
    problems = sum(check(p) for p in CONFIGS)
    if problems:
        print(f"\n{problems} problem(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
