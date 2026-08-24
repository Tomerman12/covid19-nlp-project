import { Eyebrow, FadeUp } from './shared'
import { SCHEDULE } from '@/lib/wedding'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { Flourish } from './Flourish'
import { PressedBotanicals } from './Botanicals'

/* one colour per stage — they carry the clock, the rule and the pill */
const STAGE_COLORS = ['#d9769b', '#7f9c5e', '#8fa3e8']

/** a hairline clock showing the hour the stage actually starts */
function ClockFace({ time, color }: { time: string; color: string }) {
  const [h, m] = time.split(':').map(Number)
  const hourDeg = ((h % 12) + m / 60) * 30
  const minDeg = m * 6
  return (
    <svg
      viewBox="0 0 48 48"
      style={{ width: 'clamp(38px, 11vw, 66px)', height: 'auto', display: 'block', margin: '0 auto' }}
      aria-hidden="true"
    >
      <circle cx="24" cy="24" r="21.5" fill="var(--bg2)" stroke={color} strokeWidth="1.1" />
      {/* the quarters are longer, so the dial reads at a glance */}
      {Array.from({ length: 12 }, (_, k) => {
        const long = k % 3 === 0
        return (
          <line
            key={k}
            x1="24"
            y1={long ? 5.5 : 6.5}
            x2="24"
            y2={long ? 9.5 : 8.5}
            stroke={color}
            strokeWidth={long ? 1.4 : 0.8}
            strokeLinecap="round"
            opacity={long ? 0.9 : 0.45}
            transform={`rotate(${k * 30} 24 24)`}
          />
        )
      })}
      <line x1="24" y1="24" x2="24" y2="14.5" stroke={color} strokeWidth="2" strokeLinecap="round" transform={`rotate(${hourDeg} 24 24)`} />
      <line x1="24" y1="24" x2="24" y2="10" stroke={color} strokeWidth="1.3" strokeLinecap="round" transform={`rotate(${minDeg} 24 24)`} />
      <circle cx="24" cy="24" r="1.6" fill={color} />
    </svg>
  )
}

/** the evening as three pressed cards — a clock apiece instead of a dotted rail */
export default function Timeline() {
  return (
    <section className="relative" style={flavorStyle(FLAVORS.blossom, { padding: 'clamp(16px, 2.2vh, 26px) 18px' })}>
      <PressedBotanicals set="timeline" />
      <div className="mx-auto" style={{ maxWidth: '56rem' }}>
        <div className="text-center">
          <FadeUp>
            <Flourish color="#d9769b" />
            <Eyebrow>The Evening</Eyebrow>
          </FadeUp>
          <FadeUp delay={0.08}>
            <h2 className="font-display text-balance" style={{ fontWeight: 900, fontSize: 'clamp(1.9rem, 6.4vw, 2.7rem)', margin: '10px 0 20px', color: 'var(--champ2)' }}>
              הערב שלנו, שלב אחרי שלב
            </h2>
          </FadeUp>
        </div>

        <ol
          className="grid"
          style={{ gridTemplateColumns: 'repeat(3, 1fr)', gap: 'clamp(7px, 2vw, 18px)', margin: 0, padding: 0, listStyle: 'none' }}
        >
          {SCHEDULE.map((ev, i) => (
            <li key={ev.time} style={{ height: '100%' }}>
              <FadeUp className="h-full" delay={0.12 + i * 0.14}>
                <article
                  className="relative text-center h-full"
                  style={{
                    border: '1px solid var(--line)',
                    background: 'linear-gradient(180deg, #fffdf4, #f9f2df)',
                    boxShadow: '0 10px 26px rgba(140,110,50,.12)',
                    padding: 'clamp(12px, 3vw, 18px) clamp(7px, 2vw, 14px) clamp(12px, 2.6vw, 16px)',
                  }}
                >
                  {/* the stage's colour, drawn across the head of its own card */}
                  <span
                    aria-hidden="true"
                    className="absolute"
                    style={{ top: 0, insetInlineStart: 0, insetInlineEnd: 0, height: 3, background: STAGE_COLORS[i] }}
                  />
                  <ClockFace time={ev.time} color={STAGE_COLORS[i]} />
                  <span
                    className="font-serif2 tabular block"
                    dir="ltr"
                    style={{ fontWeight: 600, fontSize: 'clamp(1.4rem, 5.2vw, 2.1rem)', color: 'var(--champ)', lineHeight: 1.15, marginTop: 8 }}
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
                  {'tag' in ev && ev.tag && (
                    <span
                      className="inline-block"
                      style={{
                        margin: '5px 0 0',
                        padding: '2px 9px',
                        borderRadius: 999,
                        border: `1px solid ${STAGE_COLORS[i]}`,
                        color: STAGE_COLORS[i],
                        fontSize: 'clamp(0.66rem, 2.2vw, 0.8rem)',
                        fontWeight: 700,
                        letterSpacing: '0.04em',
                      }}
                    >
                      {ev.tag}
                    </span>
                  )}
                  <p style={{ color: 'var(--muted)', margin: '5px 0 0', fontSize: 'clamp(0.76rem, 2.5vw, 0.95rem)', lineHeight: 1.5 }}>
                    {ev.desc}
                  </p>
                </article>
              </FadeUp>
            </li>
          ))}
        </ol>
      </div>
    </section>
  )
}
