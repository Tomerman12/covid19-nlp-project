import { useCallback, useEffect, useRef, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { COUPLE, DATE_LABEL } from '@/lib/wedding'

/* ---------------------------------------------------------------------------
 * מסך הפתיחה: מכונת המזל.
 *
 * המכונה עצמה היא צילום — רינדור פוטוריאליסטי שנוצר ב-Google Veo. היא מגיעה
 * בשני חלקים שממשיכים אחד את השני בדיוק על אותו פריים:
 *
 *   media/pull/01..24.jpg  משיכת הידית. פריים לכל שלב, נצבע לקנבס לפי האצבע —
 *                          ככה המשיכה נשארת 1:1 בַּיד ולא תסריט קבוע.
 *   media/machine.mp4      מהרגע שמשחררים: הסיבוב, הנעילה על 28 · 10 · 26,
 *                          הנורות והקונפטי.
 *
 * הטכניקה של רצף תמונות מוכן־מראש שנצבע לקנבס היא זו שאפל משתמשת בה בדפי
 * המוצר שלה — היא מבטיחה מעקב מיידי בלי המתנה לפענוח או ל-seek של וידאו.
 * ------------------------------------------------------------------------- */

const PULL_FRAMES = 24

/**
 * נתיבי קבצי המכונה. כברירת מחדל הם יחסיים ל-`index.html`; גרסת התצוגה
 * המוטמעת (עמוד יחיד בלי קבצים לצידו) מזריקה מפה של data-URI במקומם.
 */
const inlined: Record<string, string> | undefined = (window as any).__WEDDING_MEDIA__
const media = (name: string) => inlined?.[name] ?? `media/${name}`
const frameSrc = (i: number) => media(`pull/${String(i + 1).padStart(2, '0')}.jpg`)

const W = 960
const H = 720

/** כמה פיקסלים של גרירה שווים משיכה מלאה */
const TRAVEL = 118
/** מתחת לזה זו נגיעה, לא גרירה */
const TAP_SLOP = 8

/* ציוני הדרך בתוך machine.mp4 (שניות) — נמדדו מהקליפ עצמו */
const T_STOP = [4.4, 4.66, 4.88] // שלוש החבטות של הגלגלים
const T_REVEAL = 5.02 // הגלגלים נעולים על 28 · 10 · 26
const T_HANDOFF = 6.55 // עוברים להזמנה, בזמן שהקונפטי יורד

type Phase = 'pull' | 'spinning' | 'revealed'

export default function SlotIntro({ onDone }: { onDone: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const framesRef = useRef<HTMLImageElement[]>([])
  const audioRef = useRef<AudioContext | null>(null)
  const timersRef = useRef<number[]>([])
  const rafRef = useRef<number>(0)
  const doneRef = useRef(false)
  const revealedRef = useRef(false)

  /** setTimeout שמתנקה לבד ביציאה מהמסך */
  const later = (fn: () => void, ms: number) => {
    timersRef.current.push(window.setTimeout(fn, ms))
  }

  /** מיקום הידית: 0 = מנוחה, 1 = משוכה עד הסוף */
  const pull = useRef(0)
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

  /* ---------- ציור פריים המשיכה ---------- */
  const paint = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const i = Math.max(0, Math.min(PULL_FRAMES - 1, Math.round(rubber(pull.current) * (PULL_FRAMES - 1))))
    if (i === shownFrame.current) return
    const img = framesRef.current[i]
    if (!img?.complete || !img.naturalWidth) return
    canvas.getContext('2d')?.drawImage(img, 0, 0, W, H)
    shownFrame.current = i

    // רַצֶ'ט: קליק קטן כל שלושה שלבים, כמו מכונה אמיתית
    if (Math.abs(i - lastTickFrame.current) >= 3) {
      lastTickFrame.current = i
      tick(660 + i * 12, 0.028, 0.035)
    }
  }, [tick])

  const setPull = useCallback(
    (p: number) => {
      pull.current = p
      paint()
    },
    [paint],
  )

  /* ---------- אנימציה קצרה של הידית (השלמת משיכה / חזרה למנוחה) ---------- */
  const runAnim = useCallback(
    (now: number) => {
      const a = anim.current
      if (!a) return false
      const k = Math.min((now - a.t0) / a.dur, 1)
      setPull(a.from + (a.to - a.from) * easeOutCubic(k))
      if (k >= 1) {
        anim.current = null
        a.then?.()
      }
      return true
    },
    [setPull],
  )

  const animateTo = (to: number, dur: number, then?: () => void) => {
    anim.current = { from: pull.current, to, t0: performance.now(), dur, then }
  }

  /* ---------- מעבר להזמנה ---------- */
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
  }, [tick])

  /** נפילה חזרה לתמונה דוממת: מציגים את התאריך ועוברים הלאה, בלי להחמיץ את הרגע */
  const staticFinish = useCallback(
    (delay = 240) => {
      later(reveal, delay)
      later(handoff, delay + 2400)
    },
    [handoff, reveal],
  )

  /* ---------- הסיבוב: הווידאו מתנגן, ואנחנו קוראים ממנו את הזמן ---------- */
  const startSpin = useCallback(() => {
    const v = videoRef.current
    setPhase('spinning')
    buzz(24)
    if (!v) {
      staticFinish()
      return
    }
    v.currentTime = 0
    v.playbackRate = 1
    void v.play().catch(() => {})
    // רשת ביטחון אחת לכל התקלות: קודק חסר, רשת שנפלה, נגינה שנחסמה.
    // אם הווידאו לא זז בפועל — מציגים את התאריך על התמונה הדוממת.
    later(() => {
      const el = videoRef.current
      if (!el || el.paused || el.currentTime < 0.1) staticFinish(0)
    }, 900)
  }, [staticFinish])

  /* ---------- לולאה אחת: אנימציית ידית + מעקב אחרי זמן הווידאו ---------- */
  useEffect(() => {
    let stopped = false
    let cueIndex = 0

    const loop = (now: number) => {
      if (stopped) return
      runAnim(now)
      const v = videoRef.current
      if (v && !v.paused && !rm) {
        const t = v.currentTime
        while (cueIndex < T_STOP.length && t >= T_STOP[cueIndex]) {
          tick(150 + cueIndex * 26, 0.07, 0.09)
          cueIndex++
        }
        if (t >= T_REVEAL) reveal()
        if (t >= T_HANDOFF) handoff()
      }
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => {
      stopped = true
      cancelAnimationFrame(rafRef.current)
    }
  }, [handoff, reveal, rm, runAnim, tick])

  /* ---------- טעינה מוקדמת של פריימי המשיכה ---------- */
  useEffect(() => {
    let alive = true
    let loaded = 0
    const imgs: HTMLImageElement[] = []
    framesRef.current = imgs
    for (let i = 0; i < PULL_FRAMES; i++) {
      const img = new Image()
      img.decoding = 'sync'
      img.src = frameSrc(i)
      img.onload = () => {
        if (!alive) return
        loaded++
        if (i === 0) paint() // הפריים הראשון על המסך מיד
        if (loaded === PULL_FRAMES) setReady(true)
      }
      img.onerror = () => {
        if (!alive) return
        loaded++
        if (loaded === PULL_FRAMES) setReady(true)
      }
      imgs.push(img)
    }
    // גם אם משהו נתקע ברשת — לא נועלים את המשתמש בחוץ
    const t = window.setTimeout(() => alive && setReady(true), 6000)
    // תנועה מצומצמת לא דורשת נגיעה, אז אם הווידאו לא נטען בכלל — ממשיכים בלעדיו
    if (rm) later(() => staticFinish(0), 1600)
    return () => {
      alive = false
      window.clearTimeout(t)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paint])

  useEffect(
    () => () => {
      timersRef.current.forEach(window.clearTimeout)
      timersRef.current = []
    },
    [],
  )

  /* ---------- תנועה מצומצמת: התוצאה בלי הסיבוב ---------- */
  const onLoadedData = () => {
    const v = videoRef.current
    if (!v) return
    if (rm) {
      v.currentTime = T_REVEAL + 0.35
      reveal()
      later(handoff, 2600)
      return
    }
    // מחממים את הווידאו כדי שהנגיעה הראשונה תתחיל מיד, בלי המתנה לרשת
    void v
      .play()
      .then(() => {
        v.pause()
        v.currentTime = 0
      })
      .catch(() => {
        /* הדפדפן יחכה לנגיעה — זה בסדר */
      })
  }

  const onVideoError = () => {
    if (phase !== 'pull') staticFinish(200)
  }

  /* ---------- משיכת הידית ---------- */

  const onPointerDown = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (phase === 'revealed') {
      handoff() // התאריך כבר על המסך — נגיעה מקצרת את ההמתנה
      return
    }
    if (phase === 'spinning') {
      const v = videoRef.current
      if (v) v.playbackRate = 2.6 // לא מתעלמים ממגע באמצע הסיבוב — מזרזים
      return
    }
    e.currentTarget.setPointerCapture(e.pointerId)
    anim.current = null
    drag.current = { id: e.pointerId, y0: e.clientY, moved: 0, hist: [{ t: performance.now(), y: e.clientY }] }
    setPulling(true)
    setPull(Math.max(pull.current, 0.045)) // תגובה כבר על הלחיצה, לפני שזזו
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

    // מהירות מתוך היסטוריית התנועה האחרונה, לא מהנקודה האחרונה בלבד
    const now = performance.now()
    const recent = d.hist.filter((p) => now - p.t < 90)
    const a = recent[0] ?? d.hist[0]
    const b = d.hist[d.hist.length - 1]
    const velocity = ((b.y - a.y) / Math.max(b.t - a.t, 1)) * 1000 // פיקסלים לשנייה
    const v = velocity / TRAVEL

    if (d.moved < TAP_SLOP) {
      animateTo(1, 210, startSpin) // נגיעה קצרה: המכונה מושכת בעצמה
      return
    }
    // מתחייבים לפי הכיוון שאליו הלכה היד, לא רק לפי המיקום
    if (pull.current + v * 0.12 >= 0.5) {
      const remaining = Math.max(0, 1 - pull.current)
      animateTo(1, Math.max(70, Math.min(240, (remaining / Math.max(Math.abs(v), 2.2)) * 1000)), startSpin)
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
          <h1 className="slot-names font-display">
            {COUPLE.one} <span className="font-script slot-amp">&amp;</span> {COUPLE.two}
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
          <canvas ref={canvasRef} width={W} height={H} className="slot-media" aria-hidden="true" />
          <video
            ref={videoRef}
            className="slot-media"
            style={{ opacity: phase === 'pull' ? 0 : 1 }}
            poster={media('poster.jpg')}
            preload="auto"
            muted
            playsInline
            onLoadedData={onLoadedData}
            onError={onVideoError}
            aria-hidden="true"
          >
            {/* VP9 קטן יותר ומתנגן בכל דפדפן מודרני; ה-mp4 הוא הרשת הביטחון לספארי ותיק */}
            <source src={media('machine.webm')} type="video/webm" />
            <source src={media('machine.mp4')} type="video/mp4" />
          </video>
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
