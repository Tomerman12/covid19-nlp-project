import { useEffect, useState } from 'react'
import { motion, animate, useReducedMotion } from 'framer-motion'
import { EASE } from './shared'
import { FlickerText } from '@/components/vendor/flicker-text'
import { COUPLE } from '@/lib/wedding'
import { FLAVORS, flavorStyle } from '@/lib/flavors'

export default function Preloader({ onDone }: { onDone: () => void }) {
  const [n, setN] = useState(0)
  const rm = useReducedMotion()

  useEffect(() => {
    if (rm) {
      onDone()
      return
    }
    const controls = animate(0, 100, {
      duration: 2.0,
      ease: [0.65, 0, 0.35, 1],
      onUpdate: (v) => setN(Math.round(v)),
    })
    const done = window.setTimeout(onDone, 2450)
    return () => {
      controls.stop()
      window.clearTimeout(done)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <motion.div
      className="fixed inset-0 z-[120] flex items-center justify-center"
      style={flavorStyle(FLAVORS.blossom, { background: 'var(--bg)' })}
      exit={{ y: '-100%' }}
      transition={{ duration: 0.85, ease: EASE }}
      aria-label="ההזמנה נטענת"
    >
      <div className="text-center px-6">
        <p
          className="font-serif2 italic"
          dir="ltr"
          style={{ color: 'var(--champ)', fontSize: 'clamp(1.2rem,4vw,1.7rem)', letterSpacing: '0.08em' }}
        >
          the wedding of
        </p>
        <h1
          className="font-display"
          style={{
            fontWeight: 900,
            fontSize: 'clamp(2.6rem, 9vw, 4.6rem)',
            color: 'var(--ivory)',
            lineHeight: 1.2,
            margin: 0,
          }}
        >
          <FlickerText delay={250} duration={1300} replayOnHover={false}>
            {COUPLE.one}
          </FlickerText>{' '}
          <span className="font-script" style={{ color: 'var(--champ)', fontWeight: 400 }}>
            &amp;
          </span>{' '}
          <FlickerText delay={500} duration={1450} replayOnHover={false}>
            {COUPLE.two}
          </FlickerText>
        </h1>
      </div>

      <div
        className="absolute font-serif2 tabular"
        dir="ltr"
        style={{
          bottom: 'max(20px, 4vh)',
          insetInlineStart: 'clamp(20px, 5vw, 60px)',
          fontSize: 'clamp(3rem, 10vw, 5.5rem)',
          fontWeight: 500,
          color: 'var(--champ)',
          lineHeight: 1,
        }}
      >
        {n}
      </div>
      <motion.div
        className="absolute bottom-0 left-0 right-0"
        style={{
          height: 2,
          background: 'linear-gradient(90deg, var(--champ2), var(--champ))',
          scaleX: n / 100,
          transformOrigin: 'left center',
        }}
      />
    </motion.div>
  )
}
