import { useCallback, useEffect, useRef, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { COUPLE_EN, DATE_LABEL } from '@/lib/wedding'
import { createReels, type Reels } from '@/lib/reels'

/* ---------------------------------------------------------------------------
 * מסך הפתיחה: מכונת המזל.
 *
 * המכונה היא צילום — רינדור פוטוריאליסטי שנוצר ב-Veo — אבל **לא סרטון**. היא
 * נחתכה מהרקע שלה לתמונות עם שקיפות, וחלון הגלגלים נוקב בהן. כך:
 *
 *   media/machine/pull/01..31.webp   הידית, פריים לכל שלב במשיכה
 *   media/machine/lit.webp           אותה תנוחה, עם הנורות דולקות
 *
 * הידית עוקבת אחרי האצבע אחת־לאחת, והגלגלים מצוירים בקוד מתחת לחור — כלומר
 * הסיבוב אמיתי: כל אורך, כל מהירות, ואפשר לזרז אותו באמצע. אין וידאו, אין
 * מסגרת, והמכונה יושבת ישירות על נייר הקרם של ההזמנה.
 * ------------------------------------------------------------------------- */

const inlined: Record<string, string> | undefined = (window as any).__WEDDING_MEDIA__
const media = (name: string) => inlined?.[name] ?? `media/${name}`

const PULL_FRAMES = 31
const frameSrc = (i: number) => media(`machine/pull/${String(i + 1).padStart(2, '0')}.webp`)

/* המידות של תמונות המכונה, וחלון הגלגלים בתוכן (מ-meta.json של הפריקה) */
const IMG_W = 482
const IMG_H = 668
const WIN = { x: 96, y: 232, w: 229, h: 148 }
const pct = (v: number, total: number) => `${(v / total) * 100}%`

/** כמה פיקסלים של גרירה שווים משיכה מלאה */
const TRAVEL = 118
/** מתחת לזה זו נגיעה, לא גרירה */
const TAP_SLOP = 8
/** כמה זמן התאריך נשאר על המסך לפני שההזמנה נפתחת */
const HOLD_MS = 1900

type Phase = 'pull' | 'spinning' | 'revealed'

export default function SlotIntro({ onDone }: { onDone: () => void }) {
  const machineRef = useRef<HTMLCanvasElement | null>(null)
  const reelsRef = useRef<HTMLCanvasElement | null>(null)
  const reels = useRef<Reels | null>(null)
  const framesRef = useRef<HTMLImageElement[]>([])
  const litRef = useRef<HTMLImageElement | null>(null)
  const audioRef = useRef<AudioContext | null>(null)
  const timersRef = useRef<number[]>([])
  const rafRef = useRef<number>(0)
  const doneRef = useRef(false)
  const revealedRef = useRef(false)

  const later = (fn: () => void, ms: number) => {
    timersRef.current.push(window.setTimeout(fn, ms))
  }

  const pull = useRef(0) // 0 = מנוחה, 1 = משוכה עד הסוף
  const litMix = useRef(0) // 0 = נורות כבויות, 1 = דולקות
  const shownFrame = useRef(-1)
  const lastTickFrame = useRef(0)
  const anim = useRef<{ from: number; to: number; t0: number; dur: number; then?: () => void } | null>(null)
  const drag = useRef<{ id: number; y0: number; moved: number; hist: { t: number; y: number }[] } | null>(null)

  const [phase, setPhase] = useState<Phase>('pull')
  const [ready, setReady] = useState(false)
  const [pulling, setPulling] = useState(false)
  const rm = useReducedMotion()

  /* ---------- קליק מכני קצר, נוצר ב-Web Audio ---------- */
  const tick = useCallback((freq: number, vol: number, len: number) => {
    try {
      const AC = window.AudioContext || (window as any).webkitAudioContext
      if (!AC) return
      const ac = (audioRef.current ||= new AC())
      if (ac.state === 'suspended') ac.resume()
      const osc = ac.createOscillator()
      const gain = ac.createGain()
      osc.type = 'triangle'
      osc.frequency.setValueAtTime(freq, ac.currentTime)
      osc.frequency.exponentialRampToValueAtTime(freq * 0.55, ac.currentTime + len)
      gain.gain.setValueAtTime(vol, ac.currentTime)
      gain.gain.exponentialRampToValueAtTime(0.0001, ac.currentTime + len)
      osc.connect(gain).connect(ac.destination)
      osc.start()
      osc.stop(ac.currentTime + len + 0.02)
    } catch {
      /* בלי סאונד — לא נורא */
    }
  }, [])

  const buzz = (pattern: number | number[]) => {
    try {
      navigator.vibrate?.(pattern)
    } catch {
      /* לא נתמך */
    }
  }

  /* ---------- ציור המכונה עצמה ---------- */
  const sizeMachine = useCallback(() => {
    const c = machineRef.current
    if (!c) return
    const dpr = Math.min(window.devicePixelRatio || 1, 3)
    const r = c.getBoundingClientRect()
    c.width = Math.max(1, Math.round(r.width * dpr))
    c.height = Math.max(1, Math.round(r.height * dpr))
    shownFrame.current = -1
  }, [])

  const paintMachine = useCallback(
    (force = false) => {
      const c = machineRef.current
      if (!c || !c.width) return
      const i = Math.max(0, Math.min(PULL_FRAMES - 1, Math.round(rubber(pull.current) * (PULL_FRAMES - 1))))
      if (i === shownFrame.current && !force) return
      const img = framesRef.current[i]
      if (!img?.complete || !img.naturalWidth) return
      const ctx = c.getContext('2d')!
      ctx.clearRect(0, 0, c.width, c.height)
      ctx.drawImage(img, 0, 0, c.width, c.height)
      const lit = litRef.current
      if (litMix.current > 0 && lit?.complete) {
        ctx.globalAlpha = litMix.current
        ctx.drawImage(lit, 0, 0, c.width, c.height)
        ctx.globalAlpha = 1
      }
      if (i !== shownFrame.current && Math.abs(i - lastTickFrame.current) >= 3) {
        lastTickFrame.current = i
        tick(660 + i * 12, 0.028, 0.035) // רַצֶ'ט
      }
      shownFrame.current = i
    },
    [tick],
  )

  const setPull = useCallback(
    (p: number) => {
      pull.current = p
      paintMachine()
    },
    [paintMachine],
  )

  const animateTo = (to: number, dur: number, then?: () => void) => {
    anim.current = { from: pull.current, to, t0: performance.now(), dur, then }
  }

  /* ---------- סוף ---------- */
  const handoff = useCallback(() => {
    if (doneRef.current) return
    doneRef.current = true
    onDone()
  }, [onDone])

  const reveal = useCallback(() => {
    if (revealedRef.current) return
    revealedRef.current = true
    setPhase('revealed')
    tick(320, 0.06, 0.25)
    buzz([18, 60, 18])
    // הנורות נדלקות ברגע שהתאריך נעול
    const t0 = performance.now()
    const glow = () => {
      const k = Math.min((performance.now() - t0) / 420, 1)
      litMix.current = k
      paintMachine(true)
      if (k < 1) requestAnimationFrame(glow)
    }
    glow()
    later(handoff, HOLD_MS)
  }, [handoff, paintMachine, tick])

  const startSpin = useCallback(
    (power: number) => {
      setPhase('spinning')
      buzz(24)
      reels.current?.start(power)
    },
    [],
  )

  /* ---------- לולאה אחת לכל התנועה ---------- */
  useEffect(() => {
    let last = performance.now()
    const loop = (now: number) => {
      const dt = Math.min(now - last, 250)
      last = now
      const a = anim.current
      if (a) {
        const k = Math.min((now - a.t0) / a.dur, 1)
        setPull(a.from + (a.to - a.from) * easeOutCubic(k))
        if (k >= 1) {
          anim.current = null
          a.then?.()
        }
      }
      reels.current?.step(dt)
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(rafRef.current)
  }, [setPull])

  /* ---------- טעינה ---------- */
  useEffect(() => {
    let alive = true
    let loaded = 0
    const need = PULL_FRAMES + 1
    const imgs: HTMLImageElement[] = []
    framesRef.current = imgs

    const settle = () => {
      if (!alive) return
      loaded++
      if (loaded >= need) {
        setReady(true)
        paintMachine(true)
      }
    }
    for (let i = 0; i < PULL_FRAMES; i++) {
      const img = new Image()
      img.src = frameSrc(i)
      img.onload = () => {
        if (i === 0) paintMachine(true)
        settle()
      }
      img.onerror = settle
      imgs.push(img)
    }
    const lit = new Image()
    lit.src = media('machine/lit.webp')
    lit.onload = settle
    lit.onerror = settle
    litRef.current = lit

    const t = window.setTimeout(() => alive && setReady(true), 6000)
    return () => {
      alive = false
      window.clearTimeout(t)
    }
  }, [paintMachine])

  /* ---------- הרכבת הגלגלים ---------- */
  useEffect(() => {
    const canvas = reelsRef.current
    if (!canvas) return
    const r = createReels(canvas, {
      onStop: (i) => tick(150 + i * 26, 0.07, 0.09),
      onFinish: reveal,
    })
    reels.current = r

    const onResize = () => {
      sizeMachine()
      r.resize()
      paintMachine(true)
    }
    // הפונט של הספרות חייב להיות טעון לפני הציור הראשון
    ;(document as any).fonts?.ready?.then(() => r.draw())
    onResize()
    window.addEventListener('resize', onResize)

    if (rm) {
      r.showResult()
      litMix.current = 1
      paintMachine(true)
      revealedRef.current = true
      setPhase('revealed')
      later(handoff, 2600)
    }

    return () => {
      window.removeEventListener('resize', onResize)
      reels.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(
    () => () => {
      timersRef.current.forEach(window.clearTimeout)
      timersRef.current = []
    },
    [],
  )

  /* ---------- משיכת הידית ---------- */

  const onPointerDown = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (phase === 'revealed') {
      handoff() // התאריך כבר על המסך — נגיעה מקצרת את ההמתנה
      return
    }
    if (phase === 'spinning') {
      reels.current?.hurry()
      return
    }
    e.currentTarget.setPointerCapture(e.pointerId)
    anim.current = null
    drag.current = { id: e.pointerId, y0: e.clientY, moved: 0, hist: [{ t: performance.now(), y: e.clientY }] }
    setPulling(true)
    setPull(Math.max(pull.current, 0.045)) // תגובה כבר על הלחיצה
  }

  const onPointerMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    const d = drag.current
    if (!d || e.pointerId !== d.id) return
    const dy = e.clientY - d.y0
    d.moved = Math.max(d.moved, Math.abs(dy))
    d.hist.push({ t: performance.now(), y: e.clientY })
    if (d.hist.length > 6) d.hist.shift()
    setPull(Math.max(0, 0.045 + dy / TRAVEL))
  }

  const endDrag = (e: React.PointerEvent<HTMLButtonElement>) => {
    const d = drag.current
    if (!d || e.pointerId !== d.id) return
    drag.current = null
    setPulling(false)

    const now = performance.now()
    const recent = d.hist.filter((p) => now - p.t < 90)
    const a = recent[0] ?? d.hist[0]
    const b = d.hist[d.hist.length - 1]
    const velocity = ((b.y - a.y) / Math.max(b.t - a.t, 1)) * 1000
    const v = velocity / TRAVEL

    // הידית חוזרת למנוחה בזמן שהגלגלים כבר מסתובבים, כמו במכונה אמיתית
    const release = (power: number) => {
      startSpin(power)
      animateTo(0, 420)
    }
    if (d.moved < TAP_SLOP) {
      animateTo(1, 210, () => release(1))
      return
    }
    if (pull.current + v * 0.12 >= 0.5) {
      const remaining = Math.max(0, 1 - pull.current)
      const dur = Math.max(70, Math.min(240, (remaining / Math.max(Math.abs(v), 2.2)) * 1000))
      animateTo(1, dur, () => release(clamp(0.85 + Math.abs(v) * 0.16, 0.85, 1.6)))
    } else {
      animateTo(0, 380) // לא הספיק — חוזרת בשקט
    }
  }

  const revealed = phase === 'revealed'

  return (
    <motion.div className="slot-screen" exit={{ y: '-100%' }} transition={{ type: 'spring', bounce: 0, duration: 0.55 }}>
      <div className="slot-col">
        <header className="slot-head">
          <p className="slot-eyebrow font-serif2 italic" dir="ltr">
            the wedding of
          </p>
          <h1 className="slot-names font-script" dir="ltr">
            {COUPLE_EN.one} <span className="slot-amp">&amp;</span> {COUPLE_EN.two}
          </h1>
        </header>

        <div className="slot-caption">
          {revealed ? (
            <motion.p
              className="slot-date font-serif2 tabular"
              dir="ltr"
              initial={{ opacity: 0, y: 12, scale: 0.92 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ type: 'spring', bounce: 0.28, duration: 0.5 }}
            >
              {DATE_LABEL}
            </motion.p>
          ) : (
            <motion.p
              className="slot-hint"
              animate={{ opacity: !ready ? 0.28 : pulling || phase === 'spinning' ? 0.3 : 1 }}
              transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
            >
              משכו בידית לגילוי התאריך
            </motion.p>
          )}
        </div>

        <button
          className="slot-stage"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          aria-label="משכו בידית לגילוי תאריך החתונה"
          style={{ cursor: revealed ? 'default' : pulling ? 'grabbing' : 'grab' }}
        >
          <span className="slot-ground" aria-hidden="true" />
          <canvas
            ref={reelsRef}
            className="slot-reels"
            aria-hidden="true"
            style={{ left: pct(WIN.x, IMG_W), top: pct(WIN.y, IMG_H), width: pct(WIN.w, IMG_W), height: pct(WIN.h, IMG_H) }}
          />
          <canvas ref={machineRef} className="slot-machine" aria-hidden="true" />
        </button>
      </div>
    </motion.div>
  )
}

/** התנגדות גוברת מעבר לסוף המהלך — הידית לא נעצרת כמו קיר */
function rubber(x: number) {
  if (x <= 1) return x
  const over = x - 1
  return 1 + over / (1 + over * 2.6)
}

function easeOutCubic(t: number) {
  return 1 - Math.pow(1 - t, 3)
}

function clamp(v: number, a: number, b: number) {
  return v < a ? a : v > b ? b : v
}
