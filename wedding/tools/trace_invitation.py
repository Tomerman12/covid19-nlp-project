"""Turn the printed invitation into the vector artwork the paper page uses.

The couple, the hand lettering and the asterisk on wedding/paper/ are not
drawings of the printed invitation — they are the invitation, traced. This is
the tool that does it, so none of those SVGs is a magic file nobody can
regenerate.

The scan is only 594x840, which puts the illustration at 157x236 pixels: far
too small for a screen and soft at any enlargement. Rather than ship it as a
raster, each piece is lifted to vector:

  1. Sample the two colours off the scan itself rather than matching by eye.
  2. Recover ink coverage per pixel by projecting the pixel onto the
     paper->ink line. That keeps the anti-aliasing the artwork was drawn with,
     so edges stay smooth instead of going to a 1px staircase.
  3. Supersample 4-8x, threshold, and hand that to potrace, which then follows
     the smooth edge rather than the pixel grid.
  4. Re-emit as a compact SVG with the ink baked in.

The sub-motifs — the couple on their own, the burst strokes beside a raised
glass — are cut from the traced paths by index, not by cropping pixels. A
pixel crop slices through whatever crosses its edge: the first attempt at
"the cake without the couple" cut a tier in half, because the couple sits on
the cake and the two overlap. Paths do not have that problem.

    python3 tools/trace_invitation.py              # everything
    python3 tools/trace_invitation.py --list       # path bounding boxes only

Needs `potrace` on PATH (apt-get install potrace) plus Pillow and numpy.
"""
import argparse
import pathlib
import re
import subprocess
import sys
import tempfile
import xml.dom.minidom
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN = ROOT / "tools" / "invitation-src" / "invitation.png"
OUT = ROOT / "paper"

# Regions of the scan, in its own 594x840 pixel space. Measured by scanning for
# bands of ink down the sheet, not guessed.
REGIONS = {
    "cake":     dict(box=(180, 60, 430, 300), scale=4, label="איור: שחף ותומר יושבים על עוגת חתונה ומרימים כוסית"),
    "asterisk": dict(box=(262, 636, 330, 695), scale=6, pad=3, label=""),
    # pad is tighter on the lettering: its viewBox sets the aspect-ratio the
    # page reserves for it, so a looser margin here silently stretches the type
    "names":    dict(box=(45, 330, 556, 392), scale=4, pad=4, label="SHACHAF & TOMER"),
    "date":     dict(box=(183, 415, 406, 490), scale=5, pad=4, label="28.10.26"),
}

# Sub-motifs, cut out of cake.svg by path index once it has been traced.
# The indices are stable for a given trace; --list prints them with their
# bounding boxes so they can be re-checked after any change above.
PARTS = {
    # the two figures, their glasses and the burst strokes. NOT paths 10 and
    # 11: by bounding box they look like dangling legs, but they are the top
    # tier's two side edges, and including them leaves a stub of cake floating
    # under the couple.
    "couple": dict(paths=range(0, 10), box=(22, 22, 592, 482), label="שחף ותומר מרימים כוסית"),
    # the three short strokes that fly off the raised glass on the right
    "clink":  dict(paths=(1, 3, 5), box=(417, 33, 67, 55), label=""),
}


def coverage(img: Image.Image) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Per-pixel ink coverage, 0 where the pixel is bare paper and 1 where it
    is solid ink, plus the ink colour it found."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)

    flat = a.reshape(-1, 3)
    colours, counts = np.unique(flat.astype(np.uint8), axis=0, return_counts=True)
    paper = colours[counts.argmax()].astype(np.float32)

    # the ink is the median of everything far from the paper; a plain "most
    # common other colour" picks up compression noise instead
    far = flat[np.abs(flat - paper).sum(axis=1) > 200]
    ink = np.median(far, axis=0)

    d = ink - paper
    cov = np.clip(((a - paper) @ d) / (d @ d), 0, 1)
    return cov, tuple(int(c) for c in ink)


def trace(cov: np.ndarray, box, scale: int, pad: int) -> tuple[str, int, int]:
    """Crop, supersample, threshold, potrace. Returns the raw SVG."""
    x0, y0, x1, y1 = box
    sub = cov[y0:y1, x0:x1]
    ys, xs = np.where(sub > 0.12)
    sub = sub[max(0, ys.min() - pad):ys.max() + pad + 1,
              max(0, xs.min() - pad):xs.max() + pad + 1]

    big = Image.fromarray((np.clip(sub * 1.14, 0, 1) * 255).astype(np.uint8), "L")
    big = big.resize((big.width * scale, big.height * scale), Image.LANCZOS)
    bw = np.where(np.asarray(big) > 118, 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        pgm = pathlib.Path(tmp) / "in.pgm"
        svg = pathlib.Path(tmp) / "out.svg"
        Image.fromarray(bw, "L").save(pgm)
        subprocess.run(["potrace", "-s", "-o", str(svg), "--turdsize", "2",
                        "--alphamax", "1.1", "--opttolerance", "0.28", str(pgm)],
                       check=True)
        return svg.read_text(), big.width, big.height


def emit(raw: str, ink: str, label: str, box=None) -> str:
    """Repack potrace's output as one compact SVG with the ink baked in."""
    w = float(re.search(r'width="([\d.]+)pt"', raw).group(1))
    h = float(re.search(r'height="([\d.]+)pt"', raw).group(1))
    transform = re.search(r'<g transform="([^"]+)"', raw).group(1)
    body = "".join(f'<path d="{p.strip()}"/>' for p in re.findall(r'<path d="(.*?)"/>', raw, re.S))

    view = f"{box[0]} {box[1]} {box[2]} {box[3]}" if box else f"0 0 {w:.0f} {h:.0f}"
    # escape(): an unescaped & in aria-label ("SHACHAF & TOMER") is a parse
    # error. A separate .svg file is read as XML, so the whole image silently
    # fails to load — it does not degrade, it disappears.
    a11y = f' role="img" aria-label="{escape(label)}"' if label else ' aria-hidden="true"'
    out = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}"{a11y}>'
           f'<g transform="{transform}" fill="{ink}" stroke="none">{body}</g></svg>')
    xml.dom.minidom.parseString(out)          # never write something that will not parse
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print the traced cake's path bounding boxes and stop")
    args = ap.parse_args()

    if not SCAN.exists():
        print(f"missing {SCAN.relative_to(ROOT)}")
        return 1

    img = Image.open(SCAN)
    cov, ink_rgb = coverage(img)
    ink = "#{:02x}{:02x}{:02x}".format(*ink_rgb)
    print(f"scan {img.size[0]}x{img.size[1]}   ink {ink}\n")

    OUT.mkdir(parents=True, exist_ok=True)
    cake_raw = None
    for name, cfg in REGIONS.items():
        raw, w, h = trace(cov, cfg["box"], cfg["scale"], cfg.get("pad", 6))
        if name == "cake":
            cake_raw = raw
        svg = emit(raw, ink, cfg["label"])
        (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
        print(f"  {name + '.svg':16} {len(svg) / 1024:6.1f} KB   traced at {w}x{h}")

    paths = re.findall(r'<path d="(.*?)"/>', cake_raw, re.S)
    transform = re.search(r'<g transform="([^"]+)"', cake_raw).group(1)

    if args.list:
        print(f"\n{len(paths)} paths in the illustration — check indices here "
              f"before changing PARTS above")
        return 0

    print()
    for name, cfg in PARTS.items():
        body = "".join(f'<path d="{paths[i].strip()}"/>' for i in cfg["paths"])
        b = cfg["box"]
        label = cfg["label"]
        a11y = f' role="img" aria-label="{escape(label)}"' if label else ' aria-hidden="true"'
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{b[0]} {b[1]} {b[2]} {b[3]}"{a11y}>'
               f'<g transform="{transform}" fill="{ink}" stroke="none">{body}</g></svg>')
        xml.dom.minidom.parseString(svg)
        (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
        print(f"  {name + '.svg':16} {len(svg) / 1024:6.1f} KB   cut from the illustration by path")

    return 0


if __name__ == "__main__":
    sys.exit(main())
