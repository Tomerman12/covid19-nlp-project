"""Prepare the couple's snapshots for the opening screen.

They sit on the very first screen, which is also the payload-critical one, so
each is cropped square around the two of them, resized small and written as
WebP. At 300px and quality 72 the whole set costs about as much as one of the
machine's lever frames.

Crops are given as (centre-x, centre-y, size) in fractions of the transposed
image, picked by eye off a contact sheet rather than centre-cropped: the two of
them are rarely in the middle of their own photos.

    python3 tools/build_photos.py
"""
import pathlib

from PIL import Image, ImageOps

SRC = pathlib.Path("/root/.claude/uploads/fa936e26-aa08-561d-b474-6800259fef64")
OUT = pathlib.Path(__file__).resolve().parent.parent / "app" / "src" / "assets" / "photos"
SIZE = 300
QUALITY = 72

# key -> (name, centre-x, centre-y, crop size as a fraction of the short side)
PHOTOS = [
    ("b999b87f", "pizza", 0.50, 0.40, 1.00),
    ("74243081", "party", 0.53, 0.50, 1.00),
    ("1cfe186f", "times", 0.55, 0.62, 0.58),
    ("e6ec05b5", "rooftop", 0.44, 0.42, 1.00),
    # kept out of the default set on the page: a rifle is visible at the hip.
    # Built anyway so it can be dropped in without re-running anything.
    ("5736a73e", "elevator", 0.52, 0.45, 1.00),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for key, name, cx, cy, frac in PHOTOS:
        src = next(SRC.glob(key + "*"))
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        side = int(min(im.width, im.height) * frac)
        x = round(cx * im.width - side / 2)
        y = round(cy * im.height - side / 2)
        # keep the box on the canvas without moving the subject more than needed
        x = max(0, min(x, im.width - side))
        y = max(0, min(y, im.height - side))
        crop = im.crop((x, y, x + side, y + side)).resize((SIZE, SIZE), Image.LANCZOS)
        dst = OUT / f"{name}.webp"
        crop.save(dst, "WEBP", quality=QUALITY, method=6)
        kb = dst.stat().st_size / 1024
        total += kb
        print(f"  {name:9} {side}px -> {SIZE}px  {kb:5.1f} KB")
    print(f"\n{len(PHOTOS)} photos, {total:.0f} KB total, in {OUT.relative_to(OUT.parents[3])}")


if __name__ == "__main__":
    main()
