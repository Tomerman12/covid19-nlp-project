import { Eyebrow, FadeUp } from './shared'
import { SCHEDULE } from '@/lib/wedding'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { Flourish, GarlandDivider } from './Flourish'
import { PressedBotanicals, SprigRow, TimeSprig } from './Botanicals'

/* the dots stay multicolored ELEMENTS; the rail they sit on is one colour */
const RAIL = '#d9769b'
const DOT_COLORS = ['#d9769b', '#7f9c5e', '#8fa3e8']
const DOT_GLOWS = ['rgba(217,118,155,.4)', 'rgba(127,156,94,.4)', 'rgba(143,163,232,.4)']

/** horizontal schedule — the whole evening in one row */
export default function Timeline() {
  return (
    <section className="relative" style={flavorStyle(FLAVORS.blossom, { padding: 'clamp(24px, 3.2vh, 38px) 18px' })}>
      <PressedBotanicals set="timeline" />
      <div className="mx-auto" style={{ maxWidth: '56rem' }}>
      <div className="text-center">
        <FadeUp>
          <Flourish color="#d9769b" />
          <Eyebrow>The Evening</Eyebrow>
        </FadeUp>
        <FadeUp delay={0.08}>
          <h2 className="font-display text-balance" style={{ fontWeight: 900, fontSize: 'clamp(1.9rem, 6.4vw, 2.7rem)', margin: '14px 0 30px', color: 'var(--champ2)' }}>
            הערב שלנו, שלב אחרי שלב
          </h2>
        </FadeUp>
      </div>

      <div className="relative">
        <span
          aria-hidden="true"
          className="absolute"
          style={{
            top: 7,
            insetInlineStart: '16.66%',
            insetInlineEnd: '16.66%',
            height: 2,
            borderRadius: 2,
            background: RAIL,
            opacity: 0.55,
          }}
        />
        <ol
          className="grid"
          style={{ gridTemplateColumns: 'repeat(3, 1fr)', gap: 'clamp(8px, 2.4vw, 24px)', margin: 0, padding: 0, listStyle: 'none' }}
        >
          {SCHEDULE.map((ev, i) => (
            <li key={ev.time} className="text-center">
              <FadeUp delay={0.12 + i * 0.14}>
                <span
                  aria-hidden="true"
                  className="block rounded-full mx-auto"
                  style={{
                    width: 15,
                    height: 15,
                    background: 'var(--bg)',
                    border: `3px solid ${DOT_COLORS[i]}`,
                    boxShadow: `0 0 16px ${DOT_GLOWS[i]}`,
                  }}
                />
                <span
                  className="font-serif2 tabular block"
                  dir="ltr"
                  style={{ fontWeight: 600, fontSize: 'clamp(1.5rem, 5.6vw, 2.2rem)', color: 'var(--champ)', lineHeight: 1.15, marginTop: 12 }}
                >
                  {ev.time}
                </span>
                {ev.range === 'עד הבוקר' ? (
                  <span className="block" style={{ color: 'var(--muted)', fontSize: 'clamp(0.72rem, 2.4vw, 0.85rem)' }}>
                    עד הבוקר
                  </span>
                ) : (
                  <span className="block tabular" dir="ltr" style={{ color: 'var(--muted)', fontSize: 'clamp(0.72rem, 2.4vw, 0.85rem)', letterSpacing: '0.05em' }}>
                    {ev.range}
                  </span>
                )}
                <h3 className="font-display" style={{ fontWeight: 900, fontSize: 'clamp(1.1rem, 3.8vw, 1.6rem)', margin: '6px 0 0', color: 'var(--champ2)' }}>
                  {ev.title}
                </h3>
                <p style={{ color: 'var(--muted)', margin: '4px 0 0', fontSize: 'clamp(0.78rem, 2.6vw, 0.95rem)', lineHeight: 1.55 }}>
                  {ev.desc}
                </p>
                <div className="flex justify-center" style={{ marginTop: 10 }}>
                  <TimeSprig color={DOT_COLORS[i]} />
                </div>
              </FadeUp>
            </li>
          ))}
        </ol>
      </div>
      <FadeUp delay={0.4}>
        <GarlandDivider width={300} />
      </FadeUp>
      </div>
    </section>
  )
}
