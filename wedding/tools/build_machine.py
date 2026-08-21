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
import json
import os
import pathlib
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy import ndimage

FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC = os.environ.get("MACHINE_CLIP", "machine-source.mp4")
OUT = pathlib.Path(__file__).resolve().parent.parent / "media" / "machine"
H, W = 720, 1280

# The clip is locked off and the machine never leaves this box. The box is what
# separates it from its own cast shadow — no per-pixel rule can, because the
# shadow at the base scores the same against the backdrop as the gold panel.
MBOX = (470, 44, 892, 658)

# the paper the invitation is printed on
CREAM = np.array([248, 242, 228], float)

# the lever swings down over these source frames; the video takes over at PULL_TO
PULL_FROM, PULL_TO = 14 / 24, 26 / 24
END_T = 9.0

yy, xx = np.mgrid[0:H, 0:W].astype(float)
yy /= H
xx /= W
RING = np.zeros((H, W), bool)
RING[:, :380] = True
RING[:, 930:] = True
RING[:70, :] = True
basis = lambda X, Y: np.stack(
    [np.ones_like(X), X, Y, X * X, X * Y, Y * Y, X * X * Y, X * Y * Y], 1
)
# the ring and the basis never change, so solve the backdrop fit once
PINV = np.linalg.pinv(basis(xx[RING], yy[RING]))
BALL = basis(xx.ravel(), yy.ravel())

KEEP = np.zeros((H, W), bool)
KEEP[MBOX[1]:MBOX[3], MBOX[0]:MBOX[2]] = True


def matte(f):
    """Return (alpha, shadow) for one frame of the clip."""
    pred = np.empty_like(f)
    for c in range(3):
        pred[..., c] = (BALL @ (PINV @ f[..., c][RING])).reshape(H, W)

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

    # the shadow darkens the paper rather than lifting the clip's tan backdrop
    ratio = np.clip(f.sum(2) / np.maximum(pred.sum(2), 1), 0, 1)
    shadow = ndimage.gaussian_filter(np.clip((1 - ratio) * 1.5, 0, 1) * (1 - alpha), 3)
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
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- the pull: 12 source frames interpolated up so a slow drag doesn't step
    pull = decode(PULL_FROM, PULL_TO,
                  "minterpolate=fps=72:mi_mode=mci:mc_mode=aobmc:vsbmc=1")
    print(f"pull frames: {len(pull)}")

    a, _ = matte(pull[0])
    m = a[..., 0] > 0.5
    cols, rows = np.where(m.any(0))[0], np.where(m.any(1))[0]
    cx, cy = int(cols.min()) - 40, int(rows.min()) - 30
    cw, ch = int(cols.max()) + 40 - cx, int(rows.max()) + 40 - cy
    cw -= cw % 2   # h.264 wants even dimensions
    ch -= ch % 2
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

    (OUT / "meta.json").write_text(json.dumps({
        "size": [cw, ch],
        "pullFrames": len(pull),
        "videoStartsAtSourceT": PULL_TO,
        "videoFrames": n,
        "videoDuration": round(n / 24, 3),
    }, indent=2))
    for stale in ("lit.webp", "lit.png", "shadow.png"):
        (OUT / stale).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
