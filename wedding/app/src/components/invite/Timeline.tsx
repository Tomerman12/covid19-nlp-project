import { Eyebrow, FadeUp } from './shared'
import { SCHEDULE } from '@/lib/wedding'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { Flourish, ScheduleVine } from './Flourish'
import { Botanical, PressedBotanicals, type BotanicalName } from './Botanicals'

/**
 * Each stage gets a pressed specimen rather than an icon: the plant is mounted
 * in a paper seal at the head of the card, and a second, much larger print of
 * the same plant sits under the text like a flower pressed into the sheet.
 * The silhouettes are the ones the rest of the invitation already uses, so the
 * schedule stops being the one diagram on a page of botanicals.
 *
 * wheat for the feast, olive for the ceremony, dandelion for the dancing.
 */
const STAGES: { plant: BotanicalName; color: string; tilt: number; wash: number }[] = [
  { plant: 'wheat', color: '#d9769b', tilt: -12, wash: 0.1 },
  { plant: 'olive', color: '#7f9c5e', tilt: 8, wash: 0.11 },
  /* the dandelion is a dense disc, so the same opacity reads twice as loud */
  { plant: 'dandelion_flower', color: '#8fa3e8', tilt: -6, wash: 0.055 },
]

/** the plant mounted in a paper seal, the way a specimen is fixed to a card */
function PressedSeal({ plant, color, tilt }: { plant: BotanicalName; color: string; tilt: number }) {
  return (
    <div
      className="relative mx-auto"
      aria-hidden="true"
      style={{ width: 'clamp(54px, 15vw, 74px)', aspectRatio: '1', display: 'grid', placeItems: 'center' }}
    >
      {/* the outer hairline sits away from the disc, like the second rule on a card */}
      <span
        className="absolute rounded-full"
        style={{ inset: 0, border: `1px solid ${color}`, opacity: 0.28 }}
      />
      <span
        className="absolute rounded-full"
        style={{ inset: '13%', background: 'var(--bg2)', border: `1px solid ${color}`, opacity: 0.55 }}
      />
      <Botanical name={plant} size={30} color={color} opacity={0.85} rotate={tilt} style={{ position: 'relative' }} />
    </div>
  )
}

/** the evening as three pressed specimens, one per stage */
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
            <h2 className="font-display text-balance" style={{ fontWeight: 900, fontSize: 'clamp(1.9rem, 6.4vw, 2.7rem)', margin: '10px 0 10px', color: 'var(--champ2)' }}>
              הערב שלנו, שלב אחרי שלב
            </h2>
          </FadeUp>
        </div>

        {/* the connector the schedule asked for, as a stem rather than a rule */}
        <FadeUp delay={0.1}>
          <ScheduleVine colors={STAGES.map((s) => s.color)} />
        </FadeUp>

        <ol
          className="grid"
          style={{ gridTemplateColumns: 'repeat(3, 1fr)', gap: 'clamp(7px, 2vw, 18px)', margin: 0, padding: 0, listStyle: 'none' }}
        >
          {SCHEDULE.map((ev, i) => {
            const stage = STAGES[i]
            return (
              <li key={ev.time} style={{ height: '100%' }}>
                <FadeUp className="h-full" delay={0.12 + i * 0.14}>
                  <article
                    className="relative text-center h-full overflow-hidden"
                    style={{
                      border: '1px solid var(--line)',
                      background: 'linear-gradient(180deg, #fffdf4, #f9f2df)',
                      boxShadow: '0 10px 26px rgba(140,110,50,.12)',
                      padding: 'clamp(12px, 3vw, 18px) clamp(7px, 2vw, 14px) clamp(12px, 2.6vw, 16px)',
                    }}
                  >
                    {/* the same plant again, printed large and faint under the sheet */}
                    <div
                      className="absolute pointer-events-none"
                      aria-hidden="true"
                      style={{ bottom: '-14%', insetInlineStart: '-16%' }}
                    >
                      <Botanical name={stage.plant} size={150} color={stage.color} opacity={stage.wash} rotate={stage.tilt * 2} />
                    </div>

                    <div className="relative">
                      <PressedSeal plant={stage.plant} color={stage.color} tilt={stage.tilt} />
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
                            border: `1px solid ${stage.color}`,
                            color: stage.color,
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
                    </div>
                  </article>
                </FadeUp>
              </li>
            )
          })}
        </ol>
      </div>
    </section>
  )
}
