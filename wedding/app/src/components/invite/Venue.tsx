import { motion, useReducedMotion } from 'framer-motion'
import { Eyebrow, FadeUp, MagneticLink, IconNav, IconCal, IconDown } from './shared'
import { BorderBeam } from '@/components/vendor/border-beam'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { Flourish, CornerSpray } from './Flourish'
import { PressedBotanicals, SprigRow } from './Botanicals'
import { VENUE, ADDR, DATE_LABEL, DAY_LABEL, TIME_LABEL, wazeUrl, gmapsUrl, gcalUrl, icsUrl } from '@/lib/wedding'

function Pin() {
  const rm = useReducedMotion()
  return (
    <div className="relative mx-auto" style={{ width: 54, height: 54 }} aria-hidden="true">
      {!rm &&
        [0, 1].map((k) => (
          <motion.span
            key={k}
            className="absolute inset-0 rounded-full"
            style={{ border: '1px solid var(--champ)' }}
            animate={{ scale: [1, 2.1], opacity: [0.7, 0] }}
            transition={{ duration: 2.2, repeat: Infinity, delay: k * 1.1, ease: 'easeOut' }}
          />
        ))}
      <span
        className="absolute rounded-full"
        style={{ inset: 19, background: 'radial-gradient(circle at 35% 30%, #f9d6e0, var(--champ))', boxShadow: '0 0 20px rgba(196,118,143,.45)' }}
      />
    </div>
  )
}

export default function Venue() {
  const rm = useReducedMotion()
  return (
    <section className="relative text-center" style={flavorStyle(FLAVORS.blossom, { padding: 'clamp(44px, 6.5vh, 68px) 22px' })}>
      <PressedBotanicals set="venue" />
      <div className="mx-auto" style={{ maxWidth: '46rem' }}>
      <FadeUp>
        <Flourish color="#8fa3e8" />
        <Eyebrow>Getting There</Eyebrow>
        <SprigRow color="#8fa3e8" />
      </FadeUp>
      <FadeUp delay={0.08}>
        <h2 className="font-display text-balance" style={{ fontWeight: 900, fontSize: 'clamp(1.9rem, 6.4vw, 2.7rem)', margin: '14px 0 26px', color: 'var(--champ2)' }}>
          איך מגיעים אלינו?
        </h2>
      </FadeUp>

      <FadeUp delay={0.16}>
        <div
          className="relative"
          style={{
            border: '1px solid var(--line)',
            background: 'linear-gradient(180deg, #fffdf4, #f9f2df)',
            padding: 'clamp(30px, 6vw, 48px)',
            boxShadow: '0 20px 48px rgba(140,110,50,.16)',
          }}
        >
          {!rm && <BorderBeam size={90} duration={11} colorFrom="var(--champ)" colorTo="var(--p2)" delay={2} />}
          <div className="absolute pointer-events-none" style={{ top: 6, insetInlineStart: 6, opacity: 0.9 }} aria-hidden="true">
            <CornerSpray size={84} />
          </div>
          <div className="absolute pointer-events-none" style={{ top: 6, insetInlineEnd: 6, opacity: 0.9 }} aria-hidden="true">
            <CornerSpray size={84} flip />
          </div>
          <div className="absolute pointer-events-none" style={{ bottom: 6, insetInlineStart: 6, opacity: 0.9 }} aria-hidden="true">
            <CornerSpray size={84} flipY />
          </div>
          <div className="absolute pointer-events-none" style={{ bottom: 6, insetInlineEnd: 6, opacity: 0.9 }} aria-hidden="true">
            <CornerSpray size={84} flip flipY />
          </div>
          <Pin />
          <h3 className="font-display" style={{ fontWeight: 900, fontSize: 'clamp(1.7rem, 5.8vw, 2.3rem)', color: 'var(--champ)', margin: '14px 0 0' }}>
            {VENUE}
          </h3>
          <p style={{ fontSize: '1.14rem', margin: '6px 0 0' }}>{ADDR}</p>
          <p style={{ color: 'var(--muted)', margin: '4px 0 0', fontSize: '0.96rem' }}>
            {DAY_LABEL}, <span dir="ltr" className="tabular">{DATE_LABEL}</span>, קבלת פנים ב־<span dir="ltr" className="tabular">{TIME_LABEL}</span>
          </p>

          <div className="flex flex-col sm:flex-row sm:justify-center" style={{ gap: 12, marginTop: 30 }}>
            <MagneticLink href={wazeUrl} primary>
              {IconNav}
              <span>ניווט ב־Waze</span>
            </MagneticLink>
            <MagneticLink href={gcalUrl}>
              {IconCal}
              <span>יומן Google</span>
            </MagneticLink>
            <MagneticLink href={icsUrl} download="shachaf-tomer-wedding.ics">
              {IconDown}
              <span>יומן Apple / Outlook</span>
            </MagneticLink>
          </div>

          <p style={{ marginTop: 18, fontSize: '0.88rem', color: 'var(--muted)' }}>
            מעדיפים מפה?{' '}
            <a href={gmapsUrl} target="_blank" rel="noopener" className="underline underline-offset-4" style={{ color: 'var(--muted)' }} data-cursor>
              פתחו ב־Google Maps
            </a>
          </p>
        </div>
      </FadeUp>
      </div>
    </section>
  )
}
