"""Build media/share.jpg — the 1200×630 card that WhatsApp / iMessage / Facebook
show when the invitation's link is pasted into a chat.

It reuses what the site already has, so the card cannot drift from the page:
the machine comes out of media/machine/spin.mp4 at the moment the reels lock on
28 · 10 · 26, the fonts come out of app/src/fonts.css (they are embedded there as
data URIs), and the colours are the page's own tokens.

    python3 tools/build_share_card.py

Prints the version hash to paste into the og:image URL in app/index.html — the
media folder is served immutable for a year, so the ?v= is what lets a new card
actually reach anyone.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile

import imageio_ffmpeg
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLIP = ROOT / "media" / "machine" / "spin.mp4"
FONTS = ROOT / "app" / "src" / "fonts.css"
OUT = ROOT / "media" / "share.jpg"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

# seconds into spin.mp4: reels locked, bulbs lit, lever back at rest
FRAME_T = 7.0
SIZE = (1200, 630)
QUALITY = 92

CARD = """<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="fonts.css">
<style>
  :root {
    --bg: #f8f2e4;
    --ivory: #57503a;
    --ink: #4a4331;
    --muted: #837a60;
    --champ: #c73a72;
    --champ2: #a72d67;
    --line: rgba(191, 152, 70, 0.45);
    --display: 'Frank Ruhl Libre', serif;
    --body: 'Assistant', sans-serif;
    --serif: 'Cormorant Garamond', Georgia, serif;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 1200px; height: 630px; }
  body {
    background:
      radial-gradient(120% 90% at 12% 18%, rgba(255, 157, 190, 0.16), transparent 62%),
      radial-gradient(110% 90% at 88% 84%, rgba(147, 168, 239, 0.14), transparent 60%),
      var(--bg);
    font-family: var(--body);
    color: var(--ink);
    overflow: hidden;
    position: relative;
  }
  .frame { position: absolute; inset: 22px; border: 1px solid var(--line); border-radius: 4px; }
  .frame::after {
    content: ''; position: absolute; inset: 7px;
    border: 1px solid rgba(191, 152, 70, 0.22); border-radius: 2px;
  }
  .row { position: absolute; inset: 22px; display: flex; align-items: center; }

  .machine {
    flex: 0 0 auto; width: 300px;
    display: flex; justify-content: center; align-items: center;
    margin-inline-start: 48px;
  }
  .machine img { height: 500px; display: block; }

  .copy { flex: 1 1 auto; text-align: center; padding-inline: 46px 40px; padding-bottom: 6px; }
  .eyebrow {
    font-family: var(--serif); font-weight: 600; font-size: 19px;
    letter-spacing: 0.42em; color: var(--champ2); direction: ltr;
  }
  .names {
    font-family: var(--serif); direction: ltr; font-weight: 600;
    font-size: 62px; line-height: 1.06; letter-spacing: 0.13em;
    color: var(--ivory); margin-top: 16px;
  }
  .marry {
    font-family: var(--serif); direction: ltr; font-weight: 600;
    font-size: 27px; letter-spacing: 0.3em; text-transform: uppercase;
    color: var(--champ); margin-top: 16px;
  }
  .rule {
    display: flex; align-items: center; justify-content: center;
    gap: 16px; margin: 24px auto 0; max-width: 460px;
  }
  .rule i { flex: 1 1 auto; height: 1px; background: var(--line); }
  .rule b {
    flex: 0 0 auto; width: 7px; height: 7px; border-radius: 50%;
    background: var(--champ); opacity: 0.6;
  }
  .date {
    font-family: var(--serif); direction: ltr; font-weight: 700;
    font-size: 76px; line-height: 1; letter-spacing: 0.06em;
    color: var(--ink); margin-top: 20px;
    font-variant-numeric: lining-nums tabular-nums;
  }
  .where { font-size: 27px; font-weight: 600; color: var(--ink); margin-top: 22px; }
  .when { font-size: 22px; font-weight: 500; color: var(--muted); margin-top: 8px; }
  .cta {
    display: inline-block; margin-top: 30px; padding: 12px 26px;
    border: 1px solid var(--line); border-radius: 999px;
    background: rgba(255, 253, 244, 0.72);
    font-size: 21px; font-weight: 700; color: var(--champ2);
  }
</style>
</head>
<body>
  <div class="frame"></div>
  <div class="row">
    <div class="machine"><img src="machine.png" alt=""></div>
    <div class="copy">
      <div class="eyebrow">SAVE THE DATE</div>
      <div class="names">SHACHAF<br>&amp; TOMER</div>
      <div class="marry">are getting married</div>
      <div class="rule"><i></i><b></b><i></i></div>
      <div class="date">28.10.2026</div>
      <div class="where">אולם אביגדור · בן אביגדור 22, תל אביב</div>
      <div class="when">יום רביעי · קבלת פנים ב־19:30</div>
      <div class="cta">כל הפרטים בהזמנה · משכו את הידית</div>
    </div>
  </div>
</body>
</html>
"""

SHOOT = """
const {{ chromium }} = require({modules!r} + '/playwright-core');
(async () => {{
  const browser = await chromium.launch({{ executablePath: {chrome!r} }});
  const page = await browser.newPage({{ viewport: {{ width: 1200, height: 630 }}, deviceScaleFactor: 2 }});
  await page.goto('file://{dir}/card.html');
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(400);
  await page.screenshot({{ path: '{dir}/card@2x.png' }});
  await browser.close();
}})();
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)

        # the machine, straight out of the clip the page plays
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(FRAME_T),
             "-i", str(CLIP), "-frames:v", "1", str(d / "machine.png"), "-y"],
            check=True,
        )
        shutil.copy(FONTS, d / "fonts.css")
        (d / "card.html").write_text(CARD, encoding="utf-8")
        (d / "shoot.js").write_text(
            SHOOT.format(modules=str(ROOT / "app" / "node_modules"), chrome=CHROME, dir=d),
            encoding="utf-8",
        )
        subprocess.run(["node", str(d / "shoot.js")], check=True)

        # rendered at 2× and resampled down, so the serif hairlines survive
        Image.open(d / "card@2x.png").convert("RGB").resize(SIZE, Image.LANCZOS).save(
            OUT, quality=QUALITY, optimize=True, subsampling=1
        )

    data = OUT.read_bytes()
    version = hashlib.sha1(data).hexdigest()[:10]
    print(json.dumps({
        "file": str(OUT.relative_to(ROOT)),
        "size": list(SIZE),
        "kb": round(len(data) / 1024),
        "version": version,
    }, indent=2))
    print(f"\nog:image  ->  /media/share.jpg?v={version}   (update app/index.html)")


if __name__ == "__main__":
    main()
