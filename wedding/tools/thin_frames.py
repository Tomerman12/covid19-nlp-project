"""Halve the lever frames and recompress them.

The pull was exported at 37 frames and about 23 KB each — 838 KB that every
visitor downloads before the machine stops being dim, on a page whose whole
job at that moment is to show a lever. Only 5.9% of the pixels change across
the pull, but they are spread over 80% of the frame, so there is no crop to
take; the two levers that do work are frame count and quality.

19 frames is roughly 5% of lever travel each. A drag covers that in a few
pointer moves, so the step is not visible in the hand; a slow deliberate pull
is where you would see it, and at 19 it still reads as continuous.

    python3 tools/thin_frames.py --dry-run
    python3 tools/thin_frames.py

Rewrites media/machine/pull/ in place, renumbered from 01, and prints the new
PULL_FRAMES to set in SlotIntro.tsx.
"""
import argparse
import pathlib
import shutil

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
PULL = ROOT / "media" / "machine" / "pull"
BACKUP = ROOT / "tools" / "pull-original"
KEEP = 19
QUALITY = 75


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = sorted(BACKUP.glob("*.webp")) or sorted(PULL.glob("*.webp"))
    if not src:
        print("no frames found")
        return 1

    # keep the originals once, so this can be re-run at a different setting
    if not BACKUP.exists():
        if args.dry_run:
            print(f"would copy {len(src)} originals to {BACKUP.relative_to(ROOT)}")
        else:
            shutil.copytree(PULL, BACKUP)
            src = sorted(BACKUP.glob("*.webp"))

    # evenly spaced, and always including the first and last frame — the rest
    # position the lever, but those two are the two the eye actually checks
    idx = [round(i * (len(src) - 1) / (KEEP - 1)) for i in range(KEEP)]
    before = sum(f.stat().st_size for f in src)

    total = 0
    for n, i in enumerate(idx, start=1):
        im = Image.open(src[i]).convert("RGB")
        out = PULL / f"{n:02d}.webp"
        if args.dry_run:
            import io
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=QUALITY, method=6)
            total += len(buf.getvalue())
        else:
            im.save(out, "WEBP", quality=QUALITY, method=6)
            total += out.stat().st_size

    if not args.dry_run:
        for stale in sorted(PULL.glob("*.webp")):
            if int(stale.stem) > KEEP:
                stale.unlink()

    print(f"  {len(src)} frames, {before / 1024:.0f} KB")
    print(f"  {KEEP} frames, {total / 1024:.0f} KB at quality {QUALITY}")
    print(f"  {(1 - total / before) * 100:.0f}% smaller")
    print(f"\n  set PULL_FRAMES = {KEEP} in app/src/components/invite/SlotIntro.tsx")
    if args.dry_run:
        print("  (dry run — nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
