import { useEffect, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { Eyebrow, FadeUp } from './shared'
import { FlickeringGrid } from '@/components/vendor/flickering-grid'
import { TARGET_TS } from '@/lib/wedding'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { PressedBotanicals } from './Botanicals'

const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

/** a single odometer column that springs to the current digit */
function RollingDigit({ d }: { d: number }) {
  const rm = useReducedMotion()
  if (rm) {
    return <span style={{ display: 'inline-block', width: '0.62em' }}>{d}</span>
  }
  return (
    <span
      aria-hidden="true"
      style={{ display: 'inline-block', position: 'relative', width: '0.62em', height: '1.06em', overflow: 'hidden' }}
    >
      <motion.span
        style={{ position: 'absolute', insetInlineStart: 0, top: 0, display: 'flex', flexDirection: 'column', width: '100%' }}
        animate={{ y: `${-d * 1.06}em` }}
        transition={{ type: 'spring', stiffness: 170, damping: 22 }}
      >
        {DIGITS.map((n) => (
          <span key={n} style={{ height: '1.06em', lineHeight: '1.06em', textAlign: 'center' }}>
            {n}
          </span>
        ))}
      </motion.span>
    </span>
  )
}

function Unit({ value, label, minDigits = 2 }: { value: number; label: string; minDigits?: number }) {
  const str = String(value).padStart(minDigits, '0')
  return (
    <div
      className="tile text-center"
      style={{
        border: '1px solid var(--line)',
        background: 'linear-gradient(180deg, #fffdf4, #f9f2df)',
        boxShadow: '0 10px 26px rgba(140,110,50,.12)',
        padding: 'clamp(14px, 3vw, 22px) clamp(10px, 2.6vw, 18px) clamp(10px, 2vw, 16px)',
        minWidth: 'clamp(74px, 20vw, 118px)',
        transition: 'border-color .4s ease',
      }}
    >
      <span
        className="font-display tabular block"
        dir="ltr"
        aria-label={`${value} ${label}`}
        style={{ fontWeight: 900, fontSize: 'clamp(2.1rem, 8.6vw, 3.4rem)', color: 'var(--champ)', lineHeight: 1.06 }}
      >
        {str.split('').map((ch, i) => (
          <RollingDigit key={str.length - i} d={Number(ch)} />
        ))}
      </span>
      <span style={{ fontSize: '0.8rem', letterSpacing: '0.14em', color: 'var(--muted)', fontWeight: 600 }}>{label}</span>
    </div>
  )
}

export default function Countdown() {
  const [left, setLeft] = useState(() => Math.max(0, TARGET_TS - Date.now()))
  useEffect(() => {
    const t = setInterval(() => setLeft(Math.max(0, TARGET_TS - Date.now())), 1000)
    return () => clearInterval(t)
  }, [])

  const d = Math.floor(left / 864e5)
  const h = Math.floor((left % 864e5) / 36e5)
  const m = Math.floor((left % 36e5) / 6e4)
  const s = Math.floor((left % 6e4) / 1e3)

  return (
    <section className="relative text-center" style={flavorStyle(FLAVORS.blossom, { padding: 'clamp(18px, 2.4vh, 28px) 22px' })}>
      {/* flickering grid stays a soft green ELEMENT behind the rose type */}
      <FlickeringGrid
        className="absolute inset-0 pointer-events-none"
        aria-hidden="true"
        squareSize={3}
        gridGap={9}
        flickerChance={0.25}
        color="rgb(58, 154, 102)"
        maxOpacity={0.13}
        masked
      />
      <PressedBotanicals set="countdown" />
      <div className="relative mx-auto" style={{ zIndex: 1, maxWidth: '48rem' }}>
      {/* a slim band, not a full section: the date already led the hero, so this
          only has to answer "how long?" */}
      <FadeUp>
        <Eyebrow>Counting Down</Eyebrow>
      </FadeUp>

      {left <= 0 ? (
        <p className="font-display" style={{ fontWeight: 900, fontSize: 'clamp(1.5rem, 5vw, 2.1rem)', color: 'var(--champ)', marginTop: 16 }}>
          זה קורה עכשיו! 🎉
        </p>
      ) : (
        <FadeUp delay={0.1}>
          {/* a countdown reads like a clock: always left-to-right, days first */}
          <div dir="ltr" className="flex justify-center flex-wrap" style={{ gap: 'clamp(8px, 2.4vw, 16px)', marginTop: 14 }}>
            <Unit value={d} label="ימים" />
            <Unit value={h} label="שעות" />
            <Unit value={m} label="דקות" />
            <Unit value={s} label="שניות" />
          </div>
        </FadeUp>
      )}
      </div>
    </section>
  )
}
