import { useEffect, useRef, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { COUPLE, DATE_LABEL } from '@/lib/wedding'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { createSlotMachine3D } from '@/lib/slotMachine3D'

type Machine = {
  grabLever: () => void
  dragLever: (dy: number) => void
  releaseLever: (velocity: number, wasTap: boolean) => void
  skip: () => void
  showResult: () => void
  dispose: () => void
  isSpinning: () => boolean
  isDone: () => boolean
}

const TAP_SLOP = 8 // פחות מזה נחשב נגיעה, לא גרירה

/**
 * שלב הפתיחה: מכונת מזל תלת־ממדית. מושכים בידית באצבע — הידית עוקבת אחת־לאחת,
 * והמהירות של המשיכה עוברת לגלגלים. אחרי שהם נעצרים על 28 · 10 · 26 ההזמנה נפתחת.
 */
export default function SlotIntro({ onDone }: { onDone: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const machineRef = useRef<Machine | null>(null)
  const audioRef = useRef<AudioContext | null>(null)
  const handoffRef = useRef<number>(0)
  const drag = useRef<{ id: number; y0: number; moved: number; hist: { t: number; y: number }[] } | null>(null)
  const [revealed, setRevealed] = useState(false)
  const [pulling, setPulling] = useState(false)
  const rm = useReducedMotion()

  /* קליק מכני קצר, נוצר ב-Web Audio */
  const tick = (freq: number, vol: number, len: number) => {
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
  }

  const finish = () => {
    setRevealed(true)
    setPulling(false)
    tick(320, 0.06, 0.25)
    if (navigator.vibrate) {
      try {
        navigator.vibrate([18, 60, 18])
      } catch {
        /* לא נתמך */
      }
    }
    handoffRef.current = window.setTimeout(onDone, 1900)
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let m: Machine | null = null
    try {
      if (!canvas.getContext('webgl2') && !canvas.getContext('webgl')) throw new Error('no webgl')
      m = createSlotMachine3D(canvas, {
        reels: ['28', '10', '26'],
        onStop: (i: number) => tick(150 + i * 26, 0.07, 0.09),
        onFinish: finish,
      }) as Machine
      machineRef.current = m
    } catch {
      // בלי WebGL אין מה להראות — עוברים ישר להזמנה
      handoffRef.current = window.setTimeout(onDone, 400)
      return
    }

    // תנועה מצומצמת: מראים את התוצאה בלי הסיבוב, בלי לוותר על הרגע
    if (rm) {
      m.showResult()
      setRevealed(true)
      handoffRef.current = window.setTimeout(onDone, 2400)
    }

    return () => {
      window.clearTimeout(handoffRef.current)
      m?.dispose()
      machineRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /* ---------- משיכת הידית: מעקב אחד־לאחד אחרי האצבע ---------- */

  const onPointerDown = (e: React.PointerEvent<HTMLButtonElement>) => {
    const m = machineRef.current
    if (!m) return

    if (m.isDone()) {
      // הזמנה כבר מוכנה מאחורה — נגיעה מקצרת את ההמתנה
      window.clearTimeout(handoffRef.current)
      onDone()
      return
    }
    if (m.isSpinning()) {
      m.skip() // לא מתעלמים ממגע באמצע הסיבוב — מזרזים
      return
    }

    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { id: e.pointerId, y0: e.clientY, moved: 0, hist: [{ t: performance.now(), y: e.clientY }] }
    setPulling(true)
    m.grabLever() // תגובה כבר על הלחיצה, לפני שזזו
  }

  const onPointerMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    const d = drag.current
    const m = machineRef.current
    if (!d || !m || e.pointerId !== d.id) return
    const dy = e.clientY - d.y0
    d.moved = Math.max(d.moved, Math.abs(dy))
    d.hist.push({ t: performance.now(), y: e.clientY })
    if (d.hist.length > 6) d.hist.shift()
    m.dragLever(dy)
  }

  const endDrag = (e: React.PointerEvent<HTMLButtonElement>) => {
    const d = drag.current
    const m = machineRef.current
    if (!d || !m || e.pointerId !== d.id) return
    drag.current = null
    setPulling(false)

    // מהירות מתוך היסטוריית התנועה האחרונה, לא מהנקודה האחרונה בלבד
    const now = performance.now()
    const recent = d.hist.filter((p) => now - p.t < 90)
    const a = recent[0] ?? d.hist[0]
    const b = d.hist[d.hist.length - 1]
    const dt = Math.max(b.t - a.t, 1)
    const velocity = ((b.y - a.y) / dt) * 1000 // פיקסלים לשנייה

    m.releaseLever(velocity, d.moved < TAP_SLOP)
  }

  return (
    <motion.div
      className="fixed inset-0 z-[130] flex flex-col items-center justify-center text-center"
      style={flavorStyle(FLAVORS.blossom, {
        background:
          'radial-gradient(900px 700px at 50% -10%, #fffdf4, transparent 60%), radial-gradient(800px 700px at 50% 115%, #f2e8d2, transparent 60%), var(--bg)',
        padding: 'max(22px, env(safe-area-inset-top)) 18px max(22px, env(safe-area-inset-bottom))',
      })}
      exit={{ y: '-100%' }}
      transition={{ type: 'spring', bounce: 0, duration: 0.55 }}
      aria-label="מכונת מזל — משכו בידית לגילוי התאריך"
    >
      <p
        className="font-serif2 italic"
        dir="ltr"
        style={{ color: 'var(--champ)', fontSize: 'clamp(1.1rem,3.8vw,1.5rem)', letterSpacing: '0.08em', margin: 0 }}
      >
        the wedding of
      </p>

      <h1
        className="font-display"
        style={{
          fontWeight: 900,
          fontSize: 'clamp(2.1rem, 8vw, 3.4rem)',
          color: 'var(--ivory)',
          lineHeight: 1.15,
          letterSpacing: '-0.015em',
          margin: '2px 0 0',
        }}
      >
        {COUPLE.one}{' '}
        <span className="font-script" style={{ color: 'var(--champ)', fontWeight: 400, letterSpacing: 'normal' }}>
          &amp;
        </span>{' '}
        {COUPLE.two}
      </h1>

      <button
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        aria-label="משכו בידית לגילוי התאריך"
        style={{
          position: 'relative',
          width: 'min(88vw, 340px)',
          aspectRatio: '340 / 430',
          margin: 'clamp(8px,2.5vh,20px) auto 0',
          border: 0,
          padding: 0,
          background: 'none',
          cursor: revealed ? 'default' : pulling ? 'grabbing' : 'grab',
          touchAction: 'none',
          WebkitUserSelect: 'none',
          userSelect: 'none',
        }}
      >
        <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', display: 'block' }} />
      </button>

      <div style={{ minHeight: 'clamp(64px, 12vh, 96px)', display: 'grid', placeItems: 'center', marginTop: 10 }}>
        {revealed ? (
          <motion.p
            initial={{ opacity: 0, y: 10, scale: 0.94 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ type: 'spring', bounce: 0.25, duration: 0.45 }}
            className="font-serif2 tabular"
            dir="ltr"
            style={{ margin: 0, fontSize: 'clamp(2rem, 9vw, 3rem)', fontWeight: 600, color: 'var(--champ)', letterSpacing: '-0.01em' }}
          >
            {DATE_LABEL}
          </motion.p>
        ) : (
          <motion.p
            animate={{ opacity: pulling ? 0.35 : 1 }}
            transition={{ type: 'spring', bounce: 0, duration: 0.25 }}
            style={{ margin: 0, color: 'var(--muted)', fontSize: 'clamp(0.85rem,3.4vw,1rem)', letterSpacing: '0.14em' }}
          >
            משכו בידית לגילוי התאריך
          </motion.p>
        )}
      </div>
    </motion.div>
  )
}
