import { useCallback, useEffect, useRef, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { COUPLE_EN, DATE_LABEL } from '@/lib/wedding'
import { MACHINE_VERSION } from '@/lib/machineVersion'
import { ConfettiEngine } from '@/lib/confetti'

/* ---------------------------------------------------------------------------
 * מסך הפתיחה: מכונת המזל.
 *
 * המכונה היא רינדור פוטוריאליסטי שנוצר ב-Veo, והתנועה שרואים היא **התנועה
 * האמיתית מהקליפ** — הידית שחוזרת, הנורות שנדלקות בזו אחר זו, התופים שנתפסים
 * אחד־אחד, הקונפטי. אין טעם להמציא את זה מחדש בקוד; הבעיה הייתה תמיד רק
 * המלבן שהקליפ הגיע בתוכו.
 *
 * לכן כל פריים נגזר מרקע האולפן והורכב על **בדיוק הקרם של העמוד**, יחד עם הצל
 * של המכונה עצמה. מה שיוצא הוא מדיה אטומה רגילה שהרקע שלה *הוא* העמוד: בלי
 * ערוץ אלפא, בלי קודק מיוחד, ובלי מסגרת שנראית.
 *
 *   media/machine/pull/01..37.webp   הידית יורדת, פריים לכל שלב
 *   media/machine/spin.webm|.mp4     מרגע השחרור: הסיבוב, הנעילה, הקונפטי
 *
 * שני החלקים חתוכים באותו crop וממשיכים זה את זה על אותו פריים בדיוק.
 * ------------------------------------------------------------------------- */

const inlined: Record<string, string> | undefined = (window as any).__WEDDING_MEDIA__
/* the version stamp makes a rebuild a different URL, so a stale cache can never
   serve last week's machine inside this week's page */
const media = (name: string) => inlined?.[name] ?? `media/${name}?v=${MACHINE_VERSION}`

const PULL_FRAMES = 37
const frameSrc = (i: number) => media(`machine/pull/${String(i + 1).padStart(2, '0')}.webp`)

/* ציוני הדרך בתוך spin.webm (שניות) — נמדדו מהקליפ החתוך עצמו */
const T_STOP = [2.5, 3.42, 4.55] // שלושת התופים נתפסים
const T_REVEAL = 4.75 // הגלגלים נעולים על 28 · 10 · 26
const T_HANDOFF = 7.0 // אחרי שהנורות נדלקות (5.65), לפני סוף הקליפ (8.42)

/** כמה פיקסלים של גרירה שווים משיכה מלאה */
const TRAVEL = 118
/** מתחת לזה זו נגיעה, לא גרירה */
const TAP_SLOP = 8

type Phase = 'pull' | 'spinning' | 'revealed'

export default function SlotIntro({ onDone, leaving = false }: { onDone: () => void; leaving?: boolean }) {
  const machineRef = useRef<HTMLCanvasElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const stageRef = useRef<HTMLDivElement | null>(null)
  const confettiRef = useRef<HTMLCanvasElement | null>(null)
  const confetti = useRef<ConfettiEngine | null>(null)
  const framesRef = useRef<HTMLImageElement[]>([])
  const audioRef = useRef<AudioContext | null>(null)
  const timersRef = useRef<number[]>([])
  const rafRef = useRef<number>(0)
  const doneRef = useRef(false)
  const revealedRef = useRef(false)
  const unlockedRef = useRef(false)
  const startedRef = useRef(false)
  /** מתי הקליפ התקדם בפעם האחרונה — כך מזהים גם "לא התחיל" וגם "נתקע באמצע" */
  const progressRef = useRef({ t: -1, at: 0 })

  const later = (fn: () => void, ms: number) => {
    timersRef.current.push(window.setTimeout(fn, ms))
  }

  const pull = useRef(0) // 0 = מנוחה, 1 = משוכה עד הסוף
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
  const sizeCanvas = useCallback(() => {
    const c = machineRef.current
    if (!c) return
    const dpr = Math.min(window.devicePixelRatio || 1, 3)
    const r = c.getBoundingClientRect()
    if (!r.width) return
    c.width = Math.round(r.width * dpr)
    c.height = Math.round(r.height * dpr)
    shownFrame.current = -1
  }, [])

  const paint = useCallback(() => {
    const c = machineRef.current
    if (!c || !c.width) return
    const i = Math.max(0, Math.min(PULL_FRAMES - 1, Math.round(rubber(pull.current) * (PULL_FRAMES - 1))))
    if (i === shownFrame.current) return
    const img = framesRef.current[i]
    if (!img?.complete || !img.naturalWidth) return
    c.getContext('2d')!.drawImage(img, 0, 0, c.width, c.height)
    if (Math.abs(i - lastTickFrame.current) >= 3) {
      lastTickFrame.current = i
      tick(660 + i * 12, 0.028, 0.035) // רַצֶ'ט
    }
    shownFrame.current = i
  }, [tick])

  const setPull = useCallback(
    (p: number) => {
      pull.current = p
      paint()
    },
    [paint],
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

    // קונפטי סביב המכונה ברגע שהתאריך ננעל. הקליפ עצמו לא יכול לספק אותו —
    // הגזירה שומרת רק את הגוף המחובר, אז פתיתים שלא נוגעים במכונה נופלים בדרך.
    const e = confetti.current
    const r = stageRef.current?.getBoundingClientRect()
    if (e && r && !rm) {
      e.burst(r.left + r.width * 0.12, r.top + r.height * 0.22, 55)
      e.burst(r.right - r.width * 0.12, r.top + r.height * 0.22, 55)
      e.burst(r.left + r.width * 0.5, Math.max(r.top - 8, 8), 45)
    }
  }, [rm, tick])

  /** נפילה חזרה לתמונה דוממת: מציגים את התאריך ועוברים הלאה */
  const staticFinish = useCallback(
    (delay = 240) => {
      later(reveal, delay)
      later(handoff, delay + 2400)
    },
    [handoff, reveal],
  )

  /* ---------- השחרור: הקליפ ממשיך מהפריים שבו האצבע עצרה ---------- */
  const startSpin = useCallback(() => {
    const v = videoRef.current
    setPhase('spinning')
    buzz(24)
    if (!v) {
      staticFinish()
      return
    }
    startedRef.current = true

    // אחרון החבל: מה שלא יקרה לקליפ או ללולאה, התאריך מופיע וההזמנה נפתחת.
    // שתי הפעולות אידמפוטנטיות, אז במסלול התקין זה פשוט לא עושה כלום.
    later(() => reveal(), 9000)
    later(() => handoff(), 11500)

    const go = () => {
      progressRef.current = { t: -1, at: performance.now() }
      v.currentTime = 0
      v.playbackRate = 1
      void v.play().catch(() => {})
    }

    // לא מתחילים לנגן לתוך באפר ריק. או שהקליפ מוכן והסיבוב רץ חלק, או
    // שעוברים ישר לתאריך — קפיאה של ארבע שניות באמצע הסיבוב היא הדבר
    // הגרוע מבין השלושה, וזה מה שנראה במכשיר.
    if (v.readyState >= 3) {
      go()
      return
    }
    progressRef.current = { t: -1, at: performance.now() }
    const onReady = () => {
      v.removeEventListener('canplay', onReady)
      window.clearTimeout(wait)
      go()
    }
    v.addEventListener('canplay', onReady)
    const wait = window.setTimeout(() => {
      v.removeEventListener('canplay', onReady)
      startedRef.current = false
      staticFinish(0)
    }, 1500)
    timersRef.current.push(wait)
    v.load()
  }, [staticFinish])

  /* ---------- לולאה אחת: אנימציית הידית + מעקב אחרי זמן הקליפ ---------- */
  useEffect(() => {
    let stopped = false
    let cue = 0
    const loop = (now: number) => {
      if (stopped) return
      const a = anim.current
      if (a) {
        const k = Math.min((now - a.t0) / a.dur, 1)
        setPull(a.from + (a.to - a.from) * easeOutCubic(k))
        if (k >= 1) {
          anim.current = null
          a.then?.()
        }
      }
      const v = videoRef.current

      if (v && !v.paused && !rm) {
        const t = v.currentTime
        while (cue < T_STOP.length && t >= T_STOP[cue]) {
          tick(150 + cue * 26, 0.07, 0.09)
          buzz(12)
          cue++
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
  }, [handoff, reveal, rm, setPull, staticFinish, tick])

  /* ---------- רשת ביטחון שלא תלויה בלולאת הפריימים ----------
   * השומר ישב בתוך לולאת ה-requestAnimationFrame, וזה המקום היחיד שבו הוא
   * אסור: טלפון שהקליפ נתקע עליו בדרך כלל מרעיב גם את לולאת הפריימים, אז
   * השומר הפסיק לרוץ בדיוק ברגע שהיה צריך אותו — והמסך נשאר קפוא לתמיד.
   * טיימר ממשיך לפעול גם כשפריימים לא מצוירים, אז הוא מחזיק גם את זיהוי
   * התקיעה וגם את ציוני הדרך של הקליפ.
   */
  useEffect(() => {
    if (rm) return
    const id = window.setInterval(() => {
      const v = videoRef.current
      if (!v || !startedRef.current) return
      const pr = progressRef.current
      const now = performance.now()
      if (v.currentTime > pr.t + 0.01) {
        pr.t = v.currentTime
        pr.at = now
        // אותם ציוני דרך של הלולאה, כדי שגם בלי פריימים התאריך מגיע בזמן
        if (v.currentTime >= T_REVEAL) reveal()
        if (v.currentTime >= T_HANDOFF) handoff()
      } else if (!revealedRef.current && now - pr.at > 3500) {
        startedRef.current = false
        staticFinish(0)
      }
    }, 250)
    return () => window.clearInterval(id)
  }, [handoff, reveal, rm, staticFinish])

  /* ---------- הורדת הקליפ לזיכרון מראש ----------
   * iOS מתעלם מ-`preload` על סלולרי: הוא מתחיל להוריד רק כשקוראים ל-play,
   * ואז נגמר לו הבאפר באמצע הסיבוב והתמונה קופאת. מושכים אותו כאן כ-blob,
   * כך שכשמשחררים את הידית הקובץ כבר בזיכרון ואי אפשר לעקוף אותו.
   */
  useEffect(() => {
    const v = videoRef.current
    if (!v || typeof fetch !== 'function' || !('createObjectURL' in URL)) return
    const url = media(
      v.canPlayType('video/mp4; codecs="avc1.640028"') ? 'machine/spin.mp4' : 'machine/spin.webm',
    )
    let alive = true
    let objectUrl = ''
    fetch(url)
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status)))))
      .then((blob) => {
        // אם הסיבוב כבר רץ, החלפת src הייתה מאפסת אותו באמצע
        if (!alive || startedRef.current) return
        objectUrl = URL.createObjectURL(blob)
        v.src = objectUrl
        v.load()
      })
      .catch(() => {
        /* נשארים עם ה-<source> שבתגית — פשוט בלי היתרון של טעינה מראש */
      })
    return () => {
      alive = false
      if (!objectUrl) return
      // משחררים את האלמנט לפני שמבטלים את ה-blob, אחרת הוא מבקש אותו שוב
      // בזמן הפירוק ונרשמת בקשה שנכשלה
      const el = videoRef.current
      const url = objectUrl
      if (!el) {
        URL.revokeObjectURL(url)
        return
      }
      // דחייה במשימה אחת לא הספיקה: לפעמים קפיצה בזמן עוד רצה כשה-blob בוטל,
      // והבקשה נרשמה ככישלון. `load()` על אלמנט בלי src מרוקן אותו ומשגר
      // `emptied` — זה הרגע שבו הוא באמת שחרר את הקובץ, אז מחכים לו.
      let done = false
      const release = () => {
        if (done) return
        done = true
        el.removeEventListener('emptied', release)
        window.clearTimeout(fallback)
        URL.revokeObjectURL(url)
      }
      const fallback = window.setTimeout(release, 2000)
      el.addEventListener('emptied', release)
      el.pause()
      el.removeAttribute('src')
      el.load()
    }
  }, [])

  /* ---------- טעינה מוקדמת של פריימי המשיכה ---------- */
  useEffect(() => {
    let alive = true
    let loaded = 0
    const imgs: HTMLImageElement[] = []
    framesRef.current = imgs
    const settle = () => {
      if (!alive) return
      if (++loaded >= PULL_FRAMES) setReady(true)
    }
    for (let i = 0; i < PULL_FRAMES; i++) {
      const img = new Image()
      img.src = frameSrc(i)
      img.onload = () => {
        if (i === 0) {
          sizeCanvas()
          paint()
        }
        settle()
      }
      img.onerror = settle
      imgs.push(img)
    }
    if (confettiRef.current && !confetti.current) {
      confetti.current = new ConfettiEngine(confettiRef.current)
    }
    const onResize = () => {
      sizeCanvas()
      paint()
      confetti.current?.resize()
    }
    window.addEventListener('resize', onResize)
    const t = window.setTimeout(() => alive && setReady(true), 6000)
    return () => {
      alive = false
      window.clearTimeout(t)
      window.removeEventListener('resize', onResize)
    }
  }, [paint, sizeCanvas])

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
      setPhase('revealed')
      revealedRef.current = true
      later(handoff, 2600)
      return
    }
    // מחממים את הקליפ כדי שהשחרור יתחיל מיד, בלי המתנה לרשת
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

  /* ---------- משיכת הידית ---------- */

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (phase === 'revealed') {
      handoff() // התאריך כבר על המסך — נגיעה מקצרת את ההמתנה
      return
    }
    if (phase === 'spinning') {
      const v = videoRef.current
      if (v) v.playbackRate = 2.6 // לא מתעלמים ממגע באמצע — מזרזים
      return
    }
    // iOS מכבד play() רק בתוך מחווה אמיתית, והשחרור שלנו קורה מאוחר יותר
    // בתוך לולאת האנימציה — אז משחררים את הנעילה כאן, בתוך הנגיעה עצמה.
    const v = videoRef.current
    if (v && !unlockedRef.current) {
      unlockedRef.current = true
      void v
        .play()
        .then(() => {
          // on a slow connection this promise can settle after the real spin has
          // begun — pausing then would stop the clip dead
          if (!startedRef.current) {
            v.pause()
            v.currentTime = 0
          }
        })
        .catch(() => {})
    }

    e.currentTarget.setPointerCapture(e.pointerId)
    anim.current = null
    drag.current = { id: e.pointerId, y0: e.clientY, moved: 0, hist: [{ t: performance.now(), y: e.clientY }] }
    setPulling(true)
    setPull(Math.max(pull.current, 0.045)) // תגובה כבר על הלחיצה
  }

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current
    if (!d || e.pointerId !== d.id) return
    const dy = e.clientY - d.y0
    d.moved = Math.max(d.moved, Math.abs(dy))
    d.hist.push({ t: performance.now(), y: e.clientY })
    if (d.hist.length > 6) d.hist.shift()
    setPull(Math.max(0, 0.045 + dy / TRAVEL))
  }

  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current
    if (!d || e.pointerId !== d.id) return
    drag.current = null
    setPulling(false)

    // מהירות מתוך היסטוריית התנועה האחרונה, לא מהנקודה האחרונה בלבד
    const now = performance.now()
    const recent = d.hist.filter((p) => now - p.t < 90)
    const a = recent[0] ?? d.hist[0]
    const b = d.hist[d.hist.length - 1]
    const v = (((b.y - a.y) / Math.max(b.t - a.t, 1)) * 1000) / TRAVEL

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

  /** בלי מקלדת אי אפשר היה לעבור את מסך הפתיחה בכלל — Enter/רווח מושכים בידית */
  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Enter' && e.key !== ' ') return
    e.preventDefault()
    if (phase === 'revealed') {
      handoff()
      return
    }
    if (phase === 'spinning') {
      const v = videoRef.current
      if (v) v.playbackRate = 2.6
      return
    }
    if (drag.current || anim.current) return
    animateTo(1, 210, startSpin)
  }

  const revealed = phase === 'revealed'

  return (
    <div
      className={'slot-screen' + (leaving ? ' slot-screen--out' : '')}
      // אצבע על כל מקום במסך מפעילה את המכונה — לא צריך לפגוע בידית עצמה
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onKeyDown={onKeyDown}
      role="button"
      tabIndex={0}
      aria-label="משכו בידית לגילוי תאריך החתונה"
      style={{ cursor: revealed ? 'default' : pulling ? 'grabbing' : 'grab' }}
    >
      <div className="slot-col">
        <header className="slot-head">
          <p className="slot-eyebrow font-serif2 italic" dir="ltr">
            the wedding of
          </p>
          <h1 className="slot-names" dir="ltr">
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
              משכו בידית — או הקישו במסך
            </motion.p>
          )}
        </div>

        <div ref={stageRef} className="slot-stage" aria-hidden="true">
          <canvas ref={machineRef} className="slot-media" aria-hidden="true" />
          <video
            ref={videoRef}
            className="slot-media slot-video"
            style={{ opacity: phase === 'pull' ? 0 : 1 }}
            preload="none"
            muted
            playsInline
            onLoadedData={onLoadedData}
            aria-hidden="true"
          >
            {/* h.264 ראשון: כל טלפון מנגן אותו. ספארי מחזיר "maybe" ל-video/webm
                חשוף, בוחר בו ואז נכשל — לכן ה-webm יושב שני ועם codecs מפורש,
                כך שדפדפן שלא יודע לפענח VP9 פשוט מדלג עליו. */}
            <source src={media('machine/spin.mp4')} type='video/mp4; codecs="avc1.640028"' />
            <source src={media('machine/spin.webm')} type='video/webm; codecs="vp9"' />
          </video>
        </div>
      </div>
      <canvas ref={confettiRef} className="slot-confetti" aria-hidden="true" />
    </div>
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
