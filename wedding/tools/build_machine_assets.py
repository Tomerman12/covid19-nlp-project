"""Cut the machine out of the Veo clip so it can sit on the invitation's paper
with no frame around it, and knock a hole where the reels go so code can drive
them.

Out:  machine/pull/NN.webp   lever positions, RGBA, window knocked out
      machine/lit.webp       same pose, bulbs on
      machine/shadow.webp    the ground shadow as black + alpha
"""
import os, subprocess, pathlib, json
import numpy as np, imageio_ffmpeg
from scipy import ndimage

FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC = os.environ.get("MACHINE_CLIP", "machine-source.mp4")  # the Veo render this was cut from
OUT = pathlib.Path(__file__).resolve().parent.parent / "media" / "machine"
H, W = 720, 1280

# the ivory drum faces, measured off the frame — this becomes a transparent hole
WIN = dict(x0=526, x1=755, y0=261, y1=409)
# the machine, lever included, never moves outside this box
MBOX = (470, 44, 892, 658)   # the base sits at ~y=650; leave it room or it slices flat
# the lever swings down over these source frames
PULL_FROM, PULL_TO = 14 / 24, 26 / 24
LIT_T = 6.25            # bulbs on, lever back at rest, before the confetti arrives


def frames(ss, to, extra=""):
    """Decode a slice of the clip to RGB float arrays."""
    vf = extra or "null"
    p = subprocess.run(
        [FF, "-ss", f"{ss}", "-to", f"{to}", "-i", SRC, "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-", "-loglevel", "error"],
        capture_output=True, check=True)
    a = np.frombuffer(p.stdout, np.uint8)
    return a.reshape(-1, H, W, 3).astype(float)


yy, xx = np.mgrid[0:H, 0:W].astype(float)
yy /= H; xx /= W
RING = np.zeros((H, W), bool)
RING[:, :380] = True; RING[:, 930:] = True; RING[:70, :] = True
basis = lambda X, Y: np.stack([np.ones_like(X), X, Y, X * X, X * Y, Y * Y,
                               X * X * Y, X * Y * Y], 1)
BM = basis(xx[RING], yy[RING])
BALL = basis(xx.ravel(), yy.ravel())


def background_of(f):
    """Fit the smooth studio backdrop behind the machine."""
    pred = np.empty_like(f)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(BM, f[..., c][RING], rcond=None)
        pred[..., c] = (BALL @ coef).reshape(H, W)
    return pred


def cut(f):
    """Split a frame into (machine RGBA, shadow alpha).

    The machine differs from the backdrop in colour; its shadow only darkens the
    backdrop while keeping its hue. Separating on that keeps a real, soft contact
    shadow instead of a hard blob.
    """
    pred = background_of(f)
    resid = np.abs(f - pred).max(2)

    solid = resid > 26
    # The clip is locked off and the machine never leaves this box, so the box
    # is what separates it from its own cast shadow — no per-pixel rule can,
    # since the shadow at the base scores the same as the gold panel.
    keep = np.zeros_like(solid)
    keep[MBOX[1]:MBOX[3], MBOX[0]:MBOX[2]] = True
    solid &= keep
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

    # shadow = how much darker than the predicted backdrop, outside the machine
    ratio = np.clip(f.sum(2) / np.maximum(pred.sum(2), 1), 0, 1)
    shadow = np.clip((1 - ratio) * 1.5, 0, 1) * (1 - alpha)
    shadow = ndimage.gaussian_filter(shadow, 3)
    shadow[shadow < 0.02] = 0

    rgba = np.dstack([f, alpha * 255])
    return rgba, shadow


def machine_box(alpha):
    m = alpha > 0.5
    cols = np.where(m.sum(0) > 4)[0]
    rows = np.where(m.sum(1) > 4)[0]
    return cols.min(), cols.max(), rows.min(), rows.max()


def punch_window(rgba, scale, cx, cy, ref):
    """Clear the drum faces so the code-drawn reels show through."""
    x0, x1, y0, y1 = (WIN["x0"], WIN["x1"], WIN["y0"], WIN["y1"])
    # follow the same scaling the frame got
    sx0 = (x0 - ref[0]) * scale + cx
    sx1 = (x1 - ref[0]) * scale + cx
    sy0 = (y0 - ref[1]) * scale + cy
    sy1 = (y1 - ref[1]) * scale + cy
    rgba[int(round(sy0)):int(round(sy1)), int(round(sx0)):int(round(sx1)), 3] = 0
    return [sx0, sy0, sx1 - sx0, sy1 - sy0]


def write_png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape[:2]
    fmt = "rgba" if arr.shape[2] == 4 else "rgb24"
    subprocess.run([FF, "-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{w}x{h}",
                    "-i", "-", str(path), "-y", "-loglevel", "error"],
                   input=np.clip(arr, 0, 255).astype(np.uint8).tobytes(), check=True)


if __name__ == "__main__":
    # ---- the pull, motion-interpolated: 12 source frames become ~36 steps, so
    #      a slow drag doesn't step visibly from one lever position to the next
    pull = frames(PULL_FROM, PULL_TO,
                  "minterpolate=fps=72:mi_mode=mci:mc_mode=aobmc:vsbmc=1")
    print(f"pull frames decoded: {len(pull)}")

    rest_rgba, rest_shadow = cut(pull[0])
    bx0, bx1, by0, by1 = machine_box(rest_rgba[..., 3] / 255)
    print(f"machine bbox at rest: x {bx0}..{bx1}  y {by0}..{by1}  (w {bx1-bx0})")

    # crop with room for the lever's full swing and the shadow
    CX0, CX1 = bx0 - 40, bx1 + 40
    CY0, CY1 = by0 - 30, by1 + 40
    print(f"crop: x {CX0}..{CX1}  y {CY0}..{CY1}  -> {CX1-CX0} x {CY1-CY0}")
    meta = {"crop": [CX0, CY0, CX1 - CX0, CY1 - CY0]}

    for i, f in enumerate(pull):
        rgba, _ = cut(f)
        rgba[WIN["y0"]:WIN["y1"], WIN["x0"]:WIN["x1"], 3] = 0
        write_png(OUT / "pull" / f"{i+1:02d}.png", rgba[CY0:CY1, CX0:CX1])
    meta["pullFrames"] = len(pull)

    # ---- lit frame: same pose, bulbs on. The camera creeps in across the clip,
    #      so it has to be registered onto the pull frames or the machine visibly
    #      changes size the instant the reels lock — which reads as a cut.
    #      Registering on the SILHOUETTES sidesteps the bulbs, which are the one
    #      thing that legitimately differs between the two frames.
    lit = frames(LIT_T, LIT_T + 1 / 24)[0]
    lit_rgba, _ = cut(lit)
    ref_mask = rest_rgba[..., 3] > 128
    lit_mask = lit_rgba[..., 3] > 128

    def place(arr, scale, dx, dy):
        """Scale about the origin, then shift, into a frame-sized canvas."""
        z = ndimage.zoom(arr, (scale, scale) + ((1,) if arr.ndim == 3 else ()), order=1)
        out = np.zeros_like(arr, dtype=float)
        sy, sx = max(0, -dy), max(0, -dx)
        ty, tx = max(0, dy), max(0, dx)
        h = min(z.shape[0] - sy, out.shape[0] - ty)
        w = min(z.shape[1] - sx, out.shape[1] - tx)
        if h > 0 and w > 0:
            out[ty:ty + h, tx:tx + w] = z[sy:sy + h, sx:sx + w]
        return out

    def centroid(m):
        ys, xs = np.nonzero(m)
        return xs.mean(), ys.mean()

    rcx, rcy = centroid(ref_mask)
    best = None
    for scale in np.linspace(0.965, 1.035, 57):
        zm = ndimage.zoom(lit_mask.astype(float), scale, order=1) > 0.5
        if not zm.any():
            continue
        zcx, zcy = centroid(zm)
        for ddx in (-2, -1, 0, 1, 2):
            for ddy in (-2, -1, 0, 1, 2):
                dx = int(round(rcx - zcx)) + ddx
                dy = int(round(rcy - zcy)) + ddy
                cand = place(lit_mask.astype(float), scale, dx, dy) > 0.5
                err = int(np.logical_xor(cand, ref_mask).sum())
                if best is None or err < best[0]:
                    best = (err, scale, dx, dy)
    err, scale, dx, dy = best
    print(f"lit registered: scale {scale:.4f} shift ({dx},{dy}) — "
          f"{err} px differ from the rest frame silhouette "
          f"({err / max(ref_mask.sum(), 1) * 100:.2f}%)")

    canvas = place(lit_rgba, scale, dx, dy)
    canvas[WIN["y0"]:WIN["y1"], WIN["x0"]:WIN["x1"], 3] = 0
    write_png(OUT / "lit.png", canvas[CY0:CY1, CX0:CX1])

    # ---- the ground shadow, as black with alpha
    sh = np.zeros((H, W, 4))
    sh[..., 3] = rest_shadow * 255
    write_png(OUT / "shadow.png", sh[CY0:CY1, CX0:CX1])

    meta["window"] = [WIN["x0"] - CX0, WIN["y0"] - CY0,
                      WIN["x1"] - WIN["x0"], WIN["y1"] - WIN["y0"]]
    meta["size"] = [CX1 - CX0, CY1 - CY0]
    meta = json.loads(json.dumps(meta, default=int))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
