"""Draw the site icon: the invitation's five-petal blossom on its cream paper.

Without an icon every browser asks for /favicon.ico and gets a 404, and the tab
and any bookmark show a blank page. iOS asks for /apple-touch-icon.png as well
when someone adds the invitation to their home screen, so that one is a real
file; the favicon itself is inlined into the page as a data URI, which costs no
request at all.

    python3 tools/build_icons.py

Prints the data URI to paste into app/index.html.
"""
import math
import pathlib
import urllib.parse

from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = "#f8f2e4"
ROSE = "#c73a72"
GOLD = "#d9a94f"

# five petals on a circle, plus the centre — the same flower the garlands use
PETALS = [(16 + 5.6 * math.sin(math.radians(a)), 16 - 5.6 * math.cos(math.radians(a))) for a in range(0, 360, 72)]

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    f'<rect width="32" height="32" rx="7" fill="{PAPER}"/>'
    f'<g fill="{ROSE}">'
    + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5"/>' for x, y in PETALS)
    + "</g>"
    f'<circle cx="16" cy="16" r="3.6" fill="{GOLD}"/>'
    "</svg>"
)


def png(size: int, path: pathlib.Path) -> None:
    ss = 4  # supersample, then shrink, so the curves stay smooth
    im = Image.new("RGB", (size * ss, size * ss), PAPER)
    d = ImageDraw.Draw(im)
    k = size * ss / 32
    for x, y in PETALS:
        d.ellipse([(x - 5) * k, (y - 5) * k, (x + 5) * k, (y + 5) * k], fill=ROSE)
    d.ellipse([(16 - 3.6) * k, (16 - 3.6) * k, (16 + 3.6) * k, (16 + 3.6) * k], fill=GOLD)
    im.resize((size, size), Image.LANCZOS).save(path)
    print(f"  {path.relative_to(ROOT)}  {size}px  {path.stat().st_size / 1024:.1f} KB")


def ico(path: pathlib.Path) -> None:
    """A real file at /favicon.ico too: the page carries an inline icon so a
    browser never asks, but anything that goes straight for the well-known path
    would otherwise get a 404."""
    ss = 8
    im = Image.new("RGB", (32 * ss, 32 * ss), PAPER)
    d = ImageDraw.Draw(im)
    for x, y in PETALS:
        d.ellipse([(x - 5) * ss, (y - 5) * ss, (x + 5) * ss, (y + 5) * ss], fill=ROSE)
    d.ellipse([(16 - 3.6) * ss, (16 - 3.6) * ss, (16 + 3.6) * ss, (16 + 3.6) * ss], fill=GOLD)
    im.resize((64, 64), Image.LANCZOS).save(path, sizes=[(16, 16), (32, 32), (48, 48)])
    print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    (ROOT / "media").mkdir(exist_ok=True)
    png(180, ROOT / "media" / "icon-180.png")
    ico(ROOT / "favicon.ico")
    uri = "data:image/svg+xml," + urllib.parse.quote(SVG, safe="")
    print(f"\nfavicon data URI ({len(uri)} chars):\n{uri}")


if __name__ == "__main__":
    main()
