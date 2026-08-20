import { useEffect } from 'react'
import { motion, useMotionValue, useSpring, useScroll, useTransform, useReducedMotion } from 'framer-motion'
import { EASE } from './shared'
import { Particles } from '@/components/vendor/particles'
import { BorderBeam } from '@/components/vendor/border-beam'
import { Meteors } from '@/components/vendor/meteors'
import { HyperText, DIGITS_SET } from '@/components/vendor/hyper-text'
import { FlickerText } from '@/components/vendor/flicker-text'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { MiniDiscoBall, SideVine } from './Flourish'
import { PressedBotanicals } from './Botanicals'
import { COUPLE, ADDR, VENUE, DATE_LABEL, DAY_LABEL, TIME_LABEL } from '@/lib/wedding'

const MARQ = 'SHACHAF & TOMER — 28 . 10 . 2026 — TEL AVIV — '

export default function Hero() {
  const rm = useReducedMotion()

  const mx = useMotionValue(0)
  const my = useMotionValue(0)
  const smx = useSpring(mx, { stiffness: 40, damping: 16 })
  const smy = useSpring(my, { stiffness: 40, damping: 16 })
  useEffect(() => {
    if (rm || !window.matchMedia('(pointer: fine)').matches) return
    const fn = (e: PointerEvent) => {
      mx.set((e.clientX / innerWidth - 0.5) * 30)
      my.set((e.clientY / innerHeight - 0.5) * 24)
    }
    addEventListener('pointermove', fn)
    return () => removeEventListener('pointermove', fn)
  }, [rm, mx, my])

  const { scrollY } = useScroll()
  const oy1 = useTransform(scrollY, [0, 700], [0, 130])
  const oy2 = useTransform(scrollY, [0, 700], [0, -90])
  const fade = useTransform(scrollY, [0, 500], [1, 0])

  return (
    <header
      className="relative flex flex-col items-center justify-center text-center overflow-hidden"
      style={flavorStyle(FLAVORS.blossom, { minHeight: 'min(100dvh, 640px)', padding: '36px 20px 44px' })}
    >
      <PressedBotanicals set="hero" />
      {!rm && <Particles className="absolute inset-0" quantity={80} color="#c4768f" ease={70} staticity={40} size={0.5} />}
      {!rm && (
        <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
          <Meteors number={10} color="rgba(196,118,143,.8)" />
        </div>
      )}
      <motion.div
        className="hero-glow absolute pointer-events-none"
        aria-hidden="true"
        style={{
          width: 460, height: 460, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(169,143,214,.32), transparent 65%)',
          top: '6%', insetInlineStart: '12%',
          x: smx, y: oy1,
          ['--glow-blur' as string]: '80px',
          filter: 'blur(80px)',
        }}
      />
      <motion.div
        className="hero-glow absolute pointer-events-none"
        aria-hidden="true"
        style={{
          width: 380, height: 380, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(255,157,190,.3), transparent 65%)',
          bottom: '10%', insetInlineEnd: '10%',
          x: smy, y: oy2,
          ['--glow-blur' as string]: '70px',
          filter: 'blur(70px)',
        }}
      />

      <motion.div
        className="absolute pointer-events-none"
        aria-hidden="true"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.9 }}
        transition={{ duration: 1.2, delay: 0.7 }}
        style={{ top: 'clamp(12px, 4vh, 44px)', insetInlineStart: 'clamp(6px, 2.5vw, 36px)' }}
      >
        <SideVine height={150} />
      </motion.div>
      <motion.div
        className="absolute pointer-events-none"
        aria-hidden="true"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.9 }}
        transition={{ duration: 1.2, delay: 0.85 }}
        style={{ top: 'clamp(12px, 4vh, 44px)', insetInlineEnd: 'clamp(6px, 2.5vw, 36px)' }}
      >
        <SideVine flip height={150} />
      </motion.div>

      <motion.div style={{ opacity: fade }} className="relative">
        <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, delay: 0.05 }}>
          <MiniDiscoBall />
        </motion.div>
        <motion.p
          className="font-serif2 italic"
          dir="ltr"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.1 }}
          style={{ color: 'var(--champ)', fontSize: 'clamp(1.15rem, 3.8vw, 1.55rem)', letterSpacing: '0.1em' }}
        >
          the wedding of
        </motion.p>

        <h1
          className="font-display flex items-center justify-center flex-wrap"
          style={{
            fontWeight: 900,
            fontSize: 'clamp(3.4rem, 16vw, 8.6rem)',
            lineHeight: 1.08,
            color: 'var(--ivory)',
            gap: '0.16em',
            margin: '10px 0 4px',
            textShadow: '0 0 60px rgba(196,118,143,.28)',
          }}
        >
          <FlickerText delay={250} duration={1400}>{COUPLE.one}</FlickerText>
          <motion.span
            className="font-script"
            dir="ltr"
            aria-hidden="true"
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, ease: EASE, delay: 0.75 }}
            style={{ color: 'var(--champ)', fontWeight: 400, fontSize: '0.6em', transform: 'translateY(.05em)' }}
          >
            &amp;
          </motion.span>
          <FlickerText delay={550} duration={1400}>{COUPLE.two}</FlickerText>
        </h1>

        <motion.p
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: EASE, delay: 1.0 }}
          style={{ color: 'var(--muted)', fontSize: 'clamp(1rem, 3.4vw, 1.22rem)', letterSpacing: '0.05em' }}
        >
          {COUPLE.full} — מתחתנים, ואתם מוזמנים!
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: EASE, delay: 1.15 }}
          className="relative inline-flex items-center flex-wrap justify-center chrome-glass"
          style={{
            gap: 14,
            border: '1px solid var(--line)',
            padding: '13px 28px',
            marginTop: 26,
            fontFamily: 'var(--font-display)',
            fontWeight: 500,
            fontSize: 'clamp(0.98rem, 3.4vw, 1.18rem)',
            letterSpacing: '0.06em',
            background: 'rgba(255,253,244,.75)',
            backdropFilter: 'blur(6px)',
            boxShadow: '0 10px 26px rgba(140,110,50,.13)',
          }}
        >
          <span>{DAY_LABEL}</span>
          <span style={{ color: 'var(--champ)' }}>·</span>
          <HyperText dir="ltr" className="tabular" characterSet={DIGITS_SET} duration={1100} delay={1300}>
            {DATE_LABEL}
          </HyperText>
          <span style={{ color: 'var(--champ)' }}>·</span>
          <HyperText dir="ltr" className="tabular" characterSet={DIGITS_SET} duration={900} delay={1600}>
            {TIME_LABEL}
          </HyperText>
          {!rm && <BorderBeam size={56} duration={7} colorFrom="var(--champ)" colorTo="var(--blush)" />}
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.3 }}
          style={{ color: 'var(--muted)', marginTop: 16, fontSize: '0.98rem' }}
        >
          {VENUE} · {ADDR}
        </motion.p>
      </motion.div>

      <motion.div
        className="absolute flex flex-col items-center"
        style={{ bottom: 86, left: '50%', translateX: '-50%', opacity: fade }}
        aria-hidden="true"
      >
        <span style={{ color: 'var(--muted)', fontSize: '0.75rem', letterSpacing: '0.2em' }}>גללו</span>
        <div style={{ width: 1, height: 44, overflow: 'hidden', marginTop: 8 }}>
          <motion.div
            style={{ width: 1, height: 44, background: 'var(--champ)' }}
            animate={rm ? undefined : { y: ['-100%', '100%'] }}
            transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
          />
        </div>
      </motion.div>

      <div className="marquee absolute bottom-0 left-0 right-0 py-4" style={{ borderTop: '1px solid var(--line)' }}>
        <div className="marquee__track">
          {[0, 1].map((k) => (
            <span
              key={k}
              className="font-serif2"
              style={{ fontWeight: 600, fontSize: '0.95rem', letterSpacing: '0.34em', color: 'var(--champ)', opacity: 0.5 }}
            >
              {MARQ.repeat(4)}
            </span>
          ))}
        </div>
      </div>
    </header>
  )
}
