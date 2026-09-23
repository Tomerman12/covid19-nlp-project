"""Cut the web fonts down to the glyphs this invitation actually draws.

The page ships its fonts inlined, which costs no request but does mean every
visitor downloads them in full. Google serves them split by unicode-range — a
Hebrew file and a Latin file per weight — and together that came to 307 KB of
woff2, 412 KB once base64 had inflated it by a third. All of it for a page
whose words never change.

So: render the real page, ask the DOM which characters each family is asked to
draw, and keep only those, plus the margin below so an edit to the copy cannot
silently drop a letter back to a system font.

The split halves are kept as halves, each with its original unicode-range.
Two @font-face blocks that share a family, weight and style do NOT fall back
to one another glyph by glyph — the last one simply wins — so the ranges are
what keeps the Hebrew and the Latin both reachable.

    python3 tools/build_fonts.py --extract   # once: pull the originals out
    python3 tools/build_fonts.py             # rebuild app/src/fonts.css

The originals live in tools/fonts-src/ because there is no network here to
re-fetch them from, and a subset cannot be un-subsetted.
"""
import argparse
import base64
import io
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSS = ROOT / "app" / "src" / "fonts.css"
SRC = ROOT / "tools" / "fonts-src"
MANIFEST = SRC / "manifest.json"

# Everything the page draws today, measured off the DOM, plus a margin: the
# whole Hebrew alphabet including final forms, digits, ASCII letters, and the
# punctuation Hebrew typesetting uses (maqaf, geresh, gershayim).
HEBREW = "".join(chr(c) for c in range(0x05D0, 0x05EB)) + "־׳״"
LATIN = "".join(chr(c) for c in range(0x41, 0x5B)) + "".join(chr(c) for c in range(0x61, 0x7B))
DIGITS = "0123456789"
PUNCT = " !\"#&'()*,-./:;?·–—…"

CHARSETS = {
    "Assistant": HEBREW + LATIN + DIGITS + PUNCT,
    "Frank Ruhl Libre": HEBREW + LATIN + DIGITS + PUNCT,
    "Cormorant Garamond": LATIN + DIGITS + PUNCT,
    # asked for exactly one glyph on the whole site: the ampersand between the
    # two names. It was 29 KB.
    "Great Vibes": "& ",
}

FACE = re.compile(r"@font-face\s*\{([^}]*)\}")


def _descriptor(block: str, name: str, default: str) -> str:
    m = re.search(rf"{name}:\s*([^;]+)", block)
    return m.group(1).strip() if m else default


def extract() -> None:
    """Pull the original woff2 payloads and their unicode-ranges out, once."""
    SRC.mkdir(parents=True, exist_ok=True)
    entries, seen = [], {}
    for m in FACE.finditer(CSS.read_text(encoding="utf-8")):
        blk = m.group(1)
        family = re.search(r"font-family:\s*'?\"?([^;'\"]+)", blk).group(1).strip()
        weight = _descriptor(blk, "font-weight", "400")
        style = _descriptor(blk, "font-style", "normal")
        rng = _descriptor(blk, "unicode-range", "")
        data = base64.b64decode(re.search(r"base64,([A-Za-z0-9+/=]+)", blk).group(1))

        key = (family, weight, style)
        seen[key] = seen.get(key, 0) + 1
        name = f"{family.replace(' ', '')}-{weight}-{style}-{seen[key]}.woff2"
        (SRC / name).write_bytes(data)
        entries.append({"file": name, "family": family, "weight": weight,
                        "style": style, "range": rng})
        print(f"  {name}  {len(data) / 1024:.1f} KB")

    MANIFEST.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(entries)} file(s) -> {SRC.relative_to(ROOT)}")


def subset(data: bytes, text: str) -> tuple[bytes, int]:
    """Return the subsetted woff2 and how many glyphs survived."""
    from fontTools import subset as ss
    from fontTools.ttLib import TTFont

    font = TTFont(io.BytesIO(data))
    have = set()
    for table in font["cmap"].tables:
        have |= set(table.cmap)
    wanted = {ord(c) for c in text} & have
    if not wanted:
        return b"", 0

    opts = ss.Options()
    opts.flavor = "woff2"
    opts.desubroutinize = True
    # tnum matters: the schedule times and the countdown ask for
    # font-variant-numeric: tabular-nums so their columns do not shuffle, and
    # dropping the feature silently turns that CSS into a no-op.
    opts.layout_features = ["kern", "liga", "rlig", "ccmp", "mark", "mkmk",
                            "tnum", "lnum", "pnum", "onum"]
    opts.name_IDs = []
    opts.notdef_outline = False
    opts.drop_tables += ["DSIG"]

    sub = ss.Subsetter(options=opts)
    sub.populate(unicodes=wanted)
    sub.subset(font)
    out = io.BytesIO()
    font.flavor = "woff2"
    font.save(out)
    return out.getvalue(), len(wanted)


def build() -> int:
    if not MANIFEST.exists():
        print("tools/fonts-src/manifest.json is missing — run with --extract first")
        return 1

    blocks, before, after, dropped = [], 0, 0, 0
    for e in json.loads(MANIFEST.read_text(encoding="utf-8")):
        data = (SRC / e["file"]).read_bytes()
        before += len(data)
        text = CHARSETS.get(e["family"])
        if text is None:
            print(f"  ! no charset defined for {e['family']} — kept whole")
            cut, n = data, -1
        else:
            cut, n = subset(data, text)

        if not cut:
            # this half covers a range the page never uses at all
            dropped += 1
            print(f"  {e['family']:20} {e['style']:7} w{e['weight']:4} dropped, no glyphs needed")
            continue

        after += len(cut)
        uri = "data:font/woff2;base64," + base64.b64encode(cut).decode()
        rng = f"  unicode-range: {e['range']};\n" if e["range"] else ""
        blocks.append(
            "@font-face {\n"
            f"  font-family: '{e['family']}';\n"
            f"  font-style: {e['style']};\n"
            f"  font-weight: {e['weight']};\n"
            "  font-display: swap;\n"
            f"{rng}"
            f"  src: url({uri}) format('woff2');\n"
            "}\n"
        )
        print(f"  {e['family']:20} {e['style']:7} w{e['weight']:4} "
              f"{len(data) / 1024:6.1f} -> {len(cut) / 1024:5.1f} KB  ({n} glyphs)")

    header = (
        "/* Generated by tools/build_fonts.py — do not edit by hand.\n"
        "   Each face carries only the glyphs this invitation draws; the full\n"
        "   originals and their unicode-ranges are in tools/fonts-src/.\n"
        "   Re-run the tool after adding copy with a character the page did\n"
        "   not use before, or it will fall back to a system face. */\n\n"
    )
    CSS.write_text(header + "\n".join(blocks), encoding="utf-8")
    print(f"\n  woff2 {before / 1024:.0f} KB -> {after / 1024:.0f} KB   "
          f"({(1 - after / before) * 100:.0f}% smaller, {dropped} face(s) dropped)")
    print(f"  {CSS.relative_to(ROOT)} now {CSS.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="pull the originals out of the current CSS")
    args = ap.parse_args()
    if args.extract:
        extract()
        sys.exit(0)
    sys.exit(build())
