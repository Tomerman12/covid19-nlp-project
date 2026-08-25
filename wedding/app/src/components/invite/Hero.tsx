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
import { COUPLE, COUPLE_EN, DATE_LABEL, DAY_LABEL, TIME_LABEL } from '@/lib/wedding'

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
    <>
      {/* סרט הפתיחה. הוא ישב בתחתית ההירו והתנגש בקישוטי הפינות, אז הוא עלה
          לראש העמוד — שם הוא קורא כמו סרט של הזמנה ולא כמו כתובית */}
      <div className="marquee py-3" style={{ borderBottom: '1px solid var(--line)', background: 'var(--bg)' }}>
        <div className="marquee__track">
          {[0, 1].map((k) => (
            <span
              key={k}
              className="font-serif2"
              style={{ fontWeight: 600, fontSize: '0.95rem', letterSpacing: '0.34em', color: 'var(--champ)', opacity: 0.55 }}
            >
              {MARQ.repeat(4)}
            </span>
          ))}
        </div>
      </div>
    <header
      className="relative flex flex-col items-center justify-center text-center overflow-hidden"
      style={flavorStyle(FLAVORS.blossom, { minHeight: 'min(78dvh, 412px)', padding: '14px 20px 56px' })}
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

      {/* hung from the top of the hero rather than floating in the middle of
          the stack, the way the party-mode ball drops from the top of the
          viewport. The wrapper carries the scroll fade so it cannot fight the
          entry animation on the inner element. */}
      <motion.div
        className="absolute pointer-events-none"
        aria-hidden="true"
        style={{ top: 0, left: '50%', translateX: '-50%', opacity: fade }}
      >
        <motion.div initial={{ opacity: 0, y: -14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, delay: 0.05 }}>
          <MiniDiscoBall cord={34} />
        </motion.div>
      </motion.div>

      <motion.div style={{ opacity: fade }} className="relative">
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
          dir="ltr"
          style={{
            fontWeight: 900,
            /* Latin sets much wider than the Hebrew this was sized for */
            fontSize: 'clamp(2.5rem, 11vw, 6.2rem)',
            lineHeight: 1.08,
            color: 'var(--ivory)',
            gap: '0.26em',   /* Latin needs more air around the ampersand than Hebrew did */
            margin: '10px 0 4px',
            textShadow: '0 0 60px rgba(196,118,143,.28)',
          }}
        >
          <FlickerText delay={250} duration={1400}>{COUPLE_EN.one}</FlickerText>
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
          <FlickerText delay={550} duration={1400}>{COUPLE_EN.two}</FlickerText>
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
      </motion.div>

      <motion.div
        className="absolute flex flex-col items-center"
        style={{ bottom: 16, left: '50%', translateX: '-50%', opacity: fade }}
        aria-hidden="true"
      >
        <span style={{ color: 'var(--muted)', fontSize: '0.75rem', letterSpacing: '0.2em' }}>גללו</span>
        <div style={{ width: 1, height: 34, overflow: 'hidden', marginTop: 6 }}>
          <motion.div
            style={{ width: 1, height: 34, background: 'var(--champ)' }}
            animate={rm ? undefined : { y: ['-100%', '100%'] }}
            transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
          />
        </div>
      </motion.div>
    </header>
    </>
  )
}
