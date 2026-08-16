import { useEffect, useRef, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { EASE } from './shared'
import { COUPLE, DATE_LABEL } from '@/lib/wedding'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { createSlotMachine3D } from '@/lib/slotMachine3D'

/**
 * שלב הפתיחה: מכונת מזל תלת־ממדית. מושכים בידית, הגלגלים נעצרים על
 * 28 · 10 · 26, ומיד אחר כך ההזמנה נפתחת מעצמה.
 */
export default function SlotIntro({ onDone }: { onDone: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const machineRef = useRef<{ spin: () => void } | null>(null)
  const audioRef = useRef<AudioContext | null>(null)
  const [revealed, setRevealed] = useState(false)
  const [pulled, setPulled] = useState(false)
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

  useEffect(() => {
    if (rm) {
      onDone()
      return
    }
    const canvas = canvasRef.current
    if (!canvas) return
    try {
      if (!canvas.getContext('webgl2') && !canvas.getContext('webgl')) throw new Error('no webgl')
      machineRef.current = createSlotMachine3D(canvas, {
        reels: ['28', '10', '26'],
        onStop: (i: number) => tick(150 + i * 26, 0.07, 0.09),
        onFinish: () => {
          setRevealed(true)
          tick(320, 0.06, 0.25)
          if (navigator.vibrate) {
            try {
              navigator.vibrate([18, 60, 18])
            } catch {
              /* לא נתמך */
            }
          }
          window.setTimeout(onDone, 1900)
        },
      })
    } catch {
      // בלי WebGL אין מה להראות — עוברים ישר להזמנה
      window.setTimeout(onDone, 400)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const pull = () => {
    if (pulled || !machineRef.current) return
    setPulled(true)
    tick(90, 0.09, 0.13)
    machineRef.current.spin()
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
      transition={{ duration: 0.85, ease: EASE }}
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
          margin: '2px 0 0',
        }}
      >
        {COUPLE.one}{' '}
        <span className="font-script" style={{ color: 'var(--champ)', fontWeight: 400 }}>
          &amp;
        </span>{' '}
        {COUPLE.two}
      </h1>

      <button
        onClick={pull}
        aria-label="משכו בידית לגילוי התאריך"
        style={{
          position: 'relative',
          width: 'min(88vw, 340px)',
          aspectRatio: '340 / 430',
          margin: 'clamp(8px,2.5vh,20px) auto 0',
          border: 0,
          padding: 0,
          background: 'none',
          cursor: pulled ? 'default' : 'pointer',
          touchAction: 'manipulation',
        }}
      >
        <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', display: 'block' }} />
      </button>

      <div style={{ minHeight: 'clamp(64px, 12vh, 96px)', display: 'grid', placeItems: 'center', marginTop: 10 }}>
        {revealed ? (
          <motion.p
            initial={{ opacity: 0, y: 10, scale: 0.92 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.55, ease: [0.34, 1.4, 0.64, 1] }}
            className="font-serif2 tabular"
            dir="ltr"
            style={{ margin: 0, fontSize: 'clamp(2rem, 9vw, 3rem)', fontWeight: 600, color: 'var(--champ)', letterSpacing: '0.04em' }}
          >
            {DATE_LABEL}
          </motion.p>
        ) : (
          <motion.p
            animate={{ opacity: pulled ? 0 : 1 }}
            transition={{ duration: 0.35 }}
            style={{ margin: 0, color: 'var(--muted)', fontSize: 'clamp(0.85rem,3.4vw,1rem)', letterSpacing: '0.14em' }}
          >
            משכו בידית לגילוי התאריך
          </motion.p>
        )}
      </div>
    </motion.div>
  )
}
