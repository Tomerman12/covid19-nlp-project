/**
 * The three drums behind the machine's window.
 *
 * The machine itself is a photograph with the drum faces cut out of it; these
 * are drawn live underneath, so the spin is real code — any length, any speed,
 * stoppable mid-flight — instead of a recording that always plays out the same.
 *
 * Colours are sampled from the render so the drawn drums sit inside the
 * photographed bezel without a seam.
 */

const FACE_BG = '#ded3bd'
const FACE_EDGE = '#c6b9a2'
const INK = '#231810'
const GAP = 'rgba(46,32,22,0.92)'

/**
 * Each drum rests on face 0 — the zeroed odometer the machine starts on — and
 * lands on face TARGET, which spells the date.
 */
const STRIPS = [
  ['00', '07', '19', '03', '28', '11', '05', '16'],
  ['00', '04', '12', '01', '10', '06', '02', '08'],
  ['00', '13', '21', '02', '26', '30', '09', '24'],
]
const TARGET = 4 // 28 · 10 · 26

const EASE_OUT = (t: number) => 1 - Math.pow(1 - t, 3)
const EASE_IN = (t: number) => t * t
const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v)

type Plan = { dur: number; end: number; back: number; stopped: boolean }

export type Reels = ReturnType<typeof createReels>

export function createReels(
  canvas: HTMLCanvasElement,
  opts: { onStop?: (i: number) => void; onFinish?: () => void } = {},
) {
  const ctx = canvas.getContext('2d')!
  let W = 1
  let H = 1
  let dpr = 1

  let spinning = false
  let done = false
  let elapsed = 0
  let rate = 1
  let plans: Plan[] = []
  const pos = [0, 0, 0] // face units; integer = that face centred
  const vel = [0, 0, 0]

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 3)
    const r = canvas.getBoundingClientRect()
    W = Math.max(1, Math.round(r.width))
    H = Math.max(1, Math.round(r.height))
    canvas.width = Math.round(W * dpr)
    canvas.height = Math.round(H * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    draw()
  }

  /** one drum's worth of numbers, wrapped around a cylinder */
  function drawDrum(i: number, x: number, w: number) {
    const strip = STRIPS[i]
    const n = strip.length
    const cy = H / 2
    const radius = H * 0.6
    const step = 0.95 // radians between numbers — neighbours only peek in
    const p = pos[i]

    ctx.save()
    ctx.beginPath()
    ctx.rect(x, 0, w, H)
    ctx.clip()

    const grad = ctx.createLinearGradient(0, 0, 0, H)
    grad.addColorStop(0, FACE_EDGE)
    grad.addColorStop(0.5, FACE_BG)
    grad.addColorStop(1, FACE_EDGE)
    ctx.fillStyle = grad
    ctx.fillRect(x, 0, w, H)

    const speed = Math.abs(vel[i])
    const smear = clamp(speed / 26, 0, 1) // fast drums blur, like the real thing
    ctx.fillStyle = INK
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.font = `700 ${Math.round(H * 0.3)}px "Frank Ruhl Libre", Georgia, serif`

    const first = Math.floor(p) - 2
    for (let j = first; j <= first + 4; j++) {
      const theta = (j - p) * step
      if (Math.abs(theta) > Math.PI / 2) continue
      // project onto the drum: faces near the lip sit lower, squashed and dimmer
      const y = cy + Math.sin(theta) * radius
      const squash = Math.cos(theta)
      const label = strip[((j % n) + n) % n]

      ctx.save()
      ctx.translate(x + w / 2, y)
      ctx.scale(1, Math.max(squash, 0.05))
      ctx.globalAlpha = 0.2 + 0.8 * squash
      if (smear < 0.04) {
        ctx.fillText(label, 0, 0)
      } else {
        // a few offset passes read as vertical motion blur
        const spread = H * 0.16 * smear
        ctx.globalAlpha *= 0.36
        for (let k = -2; k <= 2; k++) ctx.fillText(label, 0, (k / 2) * spread)
      }
      ctx.restore()
    }
    ctx.globalAlpha = 1

    // the drums sit in a recess: dark at the lip, light across the belly
    const shade = ctx.createLinearGradient(0, 0, 0, H)
    shade.addColorStop(0, 'rgba(38,26,18,0.70)')
    shade.addColorStop(0.13, 'rgba(38,26,18,0.22)')
    shade.addColorStop(0.44, 'rgba(255,248,236,0.16)')
    shade.addColorStop(0.7, 'rgba(38,26,18,0.14)')
    shade.addColorStop(1, 'rgba(38,26,18,0.66)')
    ctx.fillStyle = shade
    ctx.fillRect(x, 0, w, H)
    ctx.restore()
  }

  function draw() {
    if (!W || !H) return
    ctx.clearRect(0, 0, W, H)
    const w = W / 3
    for (let i = 0; i < 3; i++) drawDrum(i, i * w, w)
    // the seams between drums
    ctx.fillStyle = GAP
    for (let i = 1; i < 3; i++) ctx.fillRect(Math.round(i * w) - 1, 0, 2, H)
  }

  function start(power = 1) {
    if (spinning || done) return
    rate = 1
    elapsed = 0
    const p = clamp(power, 0.85, 1.6)
    plans = [0, 1, 2].map((i) => {
      const turns = Math.round((5 + i * 1.5) * p)
      return {
        dur: (2100 + i * 560) / clamp(p, 0.9, 1.3),
        end: turns * STRIPS[i].length + TARGET,
        back: -0.4,
        stopped: false,
      }
    })
    spinning = true
  }

  /** advance the spin; returns true while anything is still moving */
  function step(dt: number) {
    if (!spinning) return false
    elapsed += dt * rate
    let moving = false

    for (let i = 0; i < 3; i++) {
      const pl = plans[i]
      const k = Math.min(elapsed / pl.dur, 1)
      if (k < 1) moving = true
      const before = pos[i]
      let a: number

      if (k < 0.05) {
        a = pl.back * EASE_OUT(k / 0.05) // wind back before it goes
      } else if (k < 0.3) {
        a = pl.back + (pl.end * 0.22 - pl.back) * EASE_IN((k - 0.05) / 0.25)
      } else if (k < 0.75) {
        a = pl.end * 0.22 + (pl.end * 0.78 - pl.end * 0.22) * ((k - 0.3) / 0.45)
      } else if (k < 0.94) {
        const u = (k - 0.75) / 0.19
        a = pl.end * 0.78 + (pl.end + 0.45 - pl.end * 0.78) * EASE_OUT(u)
      } else {
        const u = (k - 0.94) / 0.06 // the detent catching
        a = pl.end + 0.45 * (1 - EASE_OUT(u))
      }

      pos[i] = a
      vel[i] = dt > 0 ? (pos[i] - before) / (dt / 16.7) : 0
      if (!pl.stopped && k >= 0.945) {
        pl.stopped = true
        opts.onStop?.(i)
      }
    }

    draw()
    if (!moving) {
      spinning = false
      done = true
      vel[0] = vel[1] = vel[2] = 0
      draw()
      opts.onFinish?.()
    }
    return true
  }

  /** a touch mid-spin hurries it along rather than being ignored */
  function hurry() {
    if (spinning) rate = 2.8
  }

  /** reduced motion: the result, without the journey */
  function showResult() {
    done = true
    spinning = false
    for (let i = 0; i < 3; i++) {
      pos[i] = TARGET
      vel[i] = 0
    }
    draw()
  }

  return {
    resize,
    draw,
    start,
    step,
    hurry,
    showResult,
    isSpinning: () => spinning,
    isDone: () => done,
  }
}
