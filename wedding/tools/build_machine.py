"""Cut the machine out of the Veo clip and lay it on the invitation's paper.

The motion in the clip is the real thing — the lever swinging back, the bulbs
coming up, the drums settling one by one, the confetti. None of that is worth
re-inventing in code. The only problem was ever the rectangle it arrived in.

So every frame is matted off the studio backdrop and composited onto the exact
cream the page uses, together with the machine's own ground shadow. The output
is ordinary opaque media whose background *is* the page: no alpha channel, no
special codec, and no visible frame.

Two things come out, and they are pixel-aligned because they share this crop:

    media/machine/pull/01..NN.webp  the lever coming down, one still per step,
                                    scrubbed under the finger
    media/machine/spin.webm|.mp4    everything from the release onwards

Usage:
    MACHINE_CLIP=/path/to/clip.mp4 python3 tools/build_machine.py
"""
import hashlib
import json
import os
import pathlib
import re
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy import ndimage

FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC = os.environ.get("MACHINE_CLIP", "machine-source.mp4")
OUT = pathlib.Path(__file__).resolve().parent.parent / "media" / "machine"

# the paper the invitation is printed on
CREAM = np.array([248, 242, 228], float)

# where the lever swings down in the source clip; the video takes over at PULL_TO
PULL_FROM = float(os.environ.get("PULL_FROM", 22 / 24))
PULL_TO = float(os.environ.get("PULL_TO", 36 / 24))
END_T = float(os.environ.get("END_T", 9.9))

W = H = 0
RING = KEEP = PINV = BALL = None
FLOOR_Y = 0


def probe():
    """Read the clip's dimensions off ffmpeg."""
    out = subprocess.run([FF, "-i", SRC], capture_output=True, text=True).stderr
    m = re.search(r"Video:.*?, (\d+)x(\d+)", out)
    if not m:
        raise SystemExit("could not read the clip's dimensions")
    return int(m.group(1)), int(m.group(2))


def build_model():
    """The backdrop fit only needs solving once — the ring never moves."""
    global RING, PINV, BALL
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    yy /= H
    xx /= W
    RING = np.zeros((H, W), bool)
    RING[:, : int(W * 0.16)] = True
    RING[:, int(W * 0.88):] = True
    RING[: int(H * 0.06), :] = True
    basis = lambda X, Y: np.stack(
        [np.ones_like(X), X, Y, X * X, X * Y, Y * Y, X * X * Y, X * Y * Y,
         X * X * X, Y * Y * Y], 1
    )
    PINV = np.linalg.pinv(basis(xx[RING], yy[RING]))
    BALL = basis(xx.ravel(), yy.ravel())


def backdrop(f):
    pred = np.empty_like(f)
    for c in range(3):
        pred[..., c] = (BALL @ (PINV @ f[..., c][RING])).reshape(H, W)
    return pred


def find_machine(f):
    """Locate the machine so its own cast shadow can be excluded.

    No per-pixel rule separates the two — the shadow at the base scores the same
    against the backdrop as the machine's darker panels do. But the machine is a
    tall object and the shadow is a low, wide smear, so the columns it occupies
    give it away.
    """
    global KEEP, FLOOR_Y
    # A high threshold for the bootstrap: the machine differs strongly from the
    # backdrop, while whatever the light throws on the back wall is only a mild
    # darkening and would otherwise be picked up as part of the subject.
    rough = ndimage.binary_opening(np.abs(f - backdrop(f)).max(2) > 55, np.ones((4, 4)))

    tall = rough.sum(0) > H * 0.30
    if not tall.any():
        raise SystemExit("could not find the machine in the first frame")
    lab, n = ndimage.label(tall)
    which = int(lab[W // 2])
    if which == 0:  # the machine is off-centre — take the widest run instead
        which = int(np.argmax(ndimage.sum(tall, lab, range(1, n + 1)))) + 1
    core = np.where(lab == which)[0]

    # the body is the one big blob inside those columns; taking the component
    # rather than a per-row count keeps the narrow top of the arch
    sub = rough.copy()
    sub[:, :core.min()] = False
    sub[:, core.max() + 1:] = False
    l2, n2 = ndimage.label(sub)
    big = int(np.argmax(ndimage.sum(sub, l2, range(1, n2 + 1)))) + 1
    ys = np.where(ndimage.binary_fill_holes(l2 == big).any(1))[0]
    y0, y1 = int(ys.min()), int(ys.max())

    # the lever reaches outside the body's columns, so measure the width across
    # the machine's own height, stopping short of the floor and its shadow
    band = rough[y0:y0 + int((y1 - y0) * 0.85)]
    wide = np.where(band.sum(0) > (y1 - y0) * 0.02)[0]
    x0, x1 = int(wide.min()), int(wide.max())

    KEEP = np.zeros((H, W), bool)
    KEEP[max(0, y0 - 6):y1 + 10, max(0, x0 - 6):x1 + 10] = True
    FLOOR_Y = y0 + int((y1 - y0) * 0.78)   # the contact shadow lives below this
    print(f"machine at x {x0}..{x1}  y {y0}..{y1}   floor band from y={FLOOR_Y}")


def matte(f):
    """Return (alpha, shadow) for one frame of the clip."""
    pred = backdrop(f)

    solid = (np.abs(f - pred).max(2) > 26) & KEEP
    solid = ndimage.binary_closing(solid, np.ones((5, 5)))
    solid = ndimage.binary_fill_holes(solid)
    solid = ndimage.binary_opening(solid, np.ones((4, 4)))
    lab, n = ndimage.label(solid)
    if n:
        sizes = ndimage.sum(solid, lab, range(1, n + 1))
        solid = lab == (int(np.argmax(sizes)) + 1)
    solid = ndimage.binary_fill_holes(solid)

    # a soft, slightly generous edge — a hard one reads as a sticker
    alpha = ndimage.gaussian_filter(solid.astype(float), 1.5)
    alpha = np.clip((alpha - 0.20) / 0.55, 0, 1)

    # The shadow darkens the paper rather than lifting the clip's tan backdrop.
    # Only the contact shadow on the floor is wanted — anything the light throws
    # on the back wall is part of the studio, not of the machine, and would show
    # up as streaks across the invitation.
    ratio = np.clip(f.sum(2) / np.maximum(pred.sum(2), 1), 0, 1)
    shadow = np.clip((1 - ratio) * 1.5, 0, 1) * (1 - alpha)
    floor = np.zeros_like(shadow)
    floor[FLOOR_Y:] = 1
    floor = ndimage.gaussian_filter(floor, 18)
    shadow = ndimage.gaussian_filter(shadow * floor, 3)
    return alpha[..., None], shadow


def composite(f):
    a, shadow = matte(f)
    ground = CREAM * (1 - shadow * 0.5)[..., None]
    return ground * (1 - a) + f[..., :3] * a


def decode(ss, to, extra=None):
    vf = ["-vf", extra] if extra else []
    p = subprocess.run(
        [FF, "-ss", str(ss), "-to", str(to), "-i", SRC, *vf,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-", "-loglevel", "error"],
        capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3).astype(float)


def main():
    global W, H
    OUT.mkdir(parents=True, exist_ok=True)
    W, H = probe()
    print(f"clip {W}x{H}")
    build_model()

    # ---- the pull: 12 source frames interpolated up so a slow drag doesn't step
    pull = decode(PULL_FROM, PULL_TO,
                  "minterpolate=fps=72:mi_mode=mci:mc_mode=aobmc:vsbmc=1")
    print(f"pull frames: {len(pull)}")

    find_machine(pull[0])
    a, _ = matte(pull[0])
    m = a[..., 0] > 0.5
    cols, rows = np.where(m.any(0))[0], np.where(m.any(1))[0]
    cx, cy = max(0, int(cols.min()) - 40), max(0, int(rows.min()) - 30)
    cw = min(int(cols.max()) + 40, W) - cx
    ch = min(int(rows.max()) + 40, H) - cy
    cw -= cw % 2   # h.264 wants even dimensions
    ch -= ch % 2
    assert cx + cw <= W and cy + ch <= H, "crop runs past the frame"
    print(f"machine at x {cols.min()}..{cols.max()} y {rows.min()}..{rows.max()}"
          f"   crop {cw}x{ch} at ({cx},{cy})")

    (OUT / "pull").mkdir(exist_ok=True)
    for old in (OUT / "pull").glob("*"):
        old.unlink()
    total = 0
    for i, f in enumerate(pull):
        out = composite(f)[cy:cy + ch, cx:cx + cw]
        p = OUT / "pull" / f"{i + 1:02d}.webp"
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
            p, "WEBP", quality=90, method=5)
        total += p.stat().st_size
    print(f"  {len(pull)} stills, {total / 1024:.0f} KB")

    # ---- the spin: everything from the release onwards, same crop
    read = subprocess.Popen(
        [FF, "-ss", str(PULL_TO), "-to", str(END_T), "-i", SRC,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-", "-loglevel", "error"],
        stdout=subprocess.PIPE)
    enc = {}
    for name, args in {
        "spin.webm": ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0",
                      "-row-mt", "1", "-cpu-used", "2", "-g", "24"],
        "spin.mp4": ["-c:v", "libx264", "-profile:v", "high", "-preset", "slow",
                     "-crf", "20", "-g", "24", "-movflags", "+faststart"],
    }.items():
        enc[name] = subprocess.Popen(
            [FF, "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cw}x{ch}",
             "-r", "24", "-i", "-", "-an", *args, "-pix_fmt", "yuv420p",
             str(OUT / name), "-y", "-loglevel", "error"],
            stdin=subprocess.PIPE)

    n, need = 0, H * W * 3
    while True:
        buf = read.stdout.read(need)
        if len(buf) < need:
            break
        out = composite(np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(float))
        data = np.clip(out[cy:cy + ch, cx:cx + cw], 0, 255).astype(np.uint8).tobytes()
        for p in enc.values():
            p.stdin.write(data)
        n += 1
        if n % 50 == 0:
            print(f"  {n} frames")
    read.stdout.close()
    for p in enc.values():
        p.stdin.close()
        p.wait()
    print(f"  video {n} frames ({n / 24:.2f}s)")
    for name in enc:
        print(f"    {name}: {(OUT / name).stat().st_size / 1024:.0f} KB")

    # A fingerprint of what was just written, stamped onto every media URL.
    # The filenames themselves never change, so without this a browser holding
    # yesterday's cache shows yesterday's machine inside today's page — which
    # looks like a bug in the site rather than a stale file.
    digest = hashlib.sha1()
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.suffix != ".json":
            digest.update(f.read_bytes())
    version = digest.hexdigest()[:10]

    (OUT / "meta.json").write_text(json.dumps({
        "version": version,
        "size": [cw, ch],
        "pullFrames": len(pull),
        "videoStartsAtSourceT": PULL_TO,
        "videoFrames": n,
        "videoDuration": round(n / 24, 3),
    }, indent=2))

    ts = OUT.parent.parent / "app" / "src" / "lib" / "machineVersion.ts"
    ts.write_text(
        "/* Written by tools/build_machine.py — do not edit.\n"
        " * Stamped onto the media URLs so a rebuild can never be served from a\n"
        " * stale cache under the same filenames. */\n"
        f"export const MACHINE_VERSION = '{version}'\n", encoding="utf-8")
    print(f"  version {version} -> {ts.relative_to(OUT.parent.parent)}")
    for stale in ("lit.webp", "lit.png", "shadow.png"):
        (OUT / stale).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
