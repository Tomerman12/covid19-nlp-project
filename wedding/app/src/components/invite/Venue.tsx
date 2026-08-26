import { motion, useReducedMotion } from 'framer-motion'
import { Eyebrow, FadeUp, MagneticLink, IconNav, IconCal } from './shared'
import { BorderBeam } from '@/components/vendor/border-beam'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { Flourish, CornerSpray } from './Flourish'
import { PressedBotanicals, SprigRow } from './Botanicals'
import { VENUE, ADDR, wazeUrl, gmapsUrl, gcalUrl, icsUrl } from '@/lib/wedding'

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
    <section className="relative text-center" style={flavorStyle(FLAVORS.blossom, { padding: 'clamp(12px, 1.6vh, 20px) 22px' })}>
      <PressedBotanicals set="venue" />
      <div className="mx-auto" style={{ maxWidth: '46rem' }}>
      <FadeUp>
        <Flourish color="#8fa3e8" />
        <Eyebrow>Getting There</Eyebrow>
        <SprigRow color="#8fa3e8" />
      </FadeUp>
      <FadeUp delay={0.08}>
        <h2 className="font-display text-balance" style={{ fontWeight: 900, fontSize: 'clamp(1.9rem, 6.4vw, 2.7rem)', margin: '8px 0 12px', color: 'var(--champ2)' }}>
          איך מגיעים אלינו?
        </h2>
      </FadeUp>

      <FadeUp delay={0.16}>
        <div
          className="relative"
          style={{
            border: '1px solid var(--line)',
            background: 'linear-gradient(180deg, #fffdf4, #f9f2df)',
            padding: 'clamp(16px, 3.4vw, 26px)',
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

          {/* two buttons carry the work; the alternates sit under them as plain
              links, so the card stays one screen shorter */}
          <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 20 }}>
            <MagneticLink href={wazeUrl} primary>
              {IconNav}
              <span>ניווט ב־Waze</span>
            </MagneticLink>
            <MagneticLink href={gcalUrl}>
              {IconCal}
              <span>יומן</span>
            </MagneticLink>
          </div>

          <p className="flex justify-center flex-wrap" style={{ gap: '4px 18px', marginTop: 14, fontSize: '0.85rem', color: 'var(--muted)' }}>
            <a href={gmapsUrl} target="_blank" rel="noopener" className="underline underline-offset-4" style={{ color: 'inherit' }} data-cursor>
              Google Maps
            </a>
            <a href={icsUrl} download="shachaf-tomer-wedding.ics" className="underline underline-offset-4" style={{ color: 'inherit' }} data-cursor>
              יומן Apple / Outlook
            </a>
          </p>
        </div>
      </FadeUp>
      </div>
    </section>
  )
}
