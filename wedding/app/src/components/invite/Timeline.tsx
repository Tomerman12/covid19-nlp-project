import { Eyebrow, FadeUp } from './shared'
import { SCHEDULE } from '@/lib/wedding'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { Flourish, ScheduleVine, StemLink } from './Flourish'
import { Botanical, type BotanicalName } from './Botanicals'

/**
 * The evening, printed on the paper rather than boxed into cards.
 *
 * Each stage gets a pressed specimen chosen for what happens in it, mounted in
 * a paper seal and printed again underneath, large and faint. The silhouettes
 * and the three colours are the ones the rest of the invitation already uses.
 *
 * The layout turns with the viewport. On a phone the stages stack and a stem
 * grows down the rail from one to the next, which gives the text the whole
 * column; three columns on a 390px screen leave about eleven characters to the
 * line. On a wide screen they run side by side beneath the flowering vine, the
 * way a programme is set.
 *
 * wheat for the feast, olive for the ceremony, dandelion for the dancing.
 */
const STAGES: { plant: BotanicalName; color: string; ink: string; tilt: number; wash: number }[] = [
  /* `color` draws the seal, the bloom and the wash; `ink` is the only one that
     ever sets type. The pastels measure 2.2:1 to 2.8:1 on the paper, so a label
     printed in them fails AA outright — these darker mixes clear 5:1. */
  { plant: 'wheat', color: '#d9769b', ink: '#a72d67', tilt: -12, wash: 0.1 },
  { plant: 'olive', color: '#7f9c5e', ink: '#4e6636', tilt: 8, wash: 0.11 },
  /* the dandelion is a dense disc, so the same opacity reads twice as loud */
  { plant: 'dandelion_flower', color: '#8fa3e8', ink: '#4a5fa8', tilt: -6, wash: 0.055 },
]

/** the plant mounted in a paper seal, the way a specimen is fixed to a sheet */
function PressedSeal({ plant, color, tilt }: { plant: BotanicalName; color: string; tilt: number }) {
  return (
    <div
      className="relative"
      aria-hidden="true"
      style={{ width: 'var(--seal)', height: 'var(--seal)', display: 'grid', placeItems: 'center' }}
    >
      {/* the outer hairline sits away from the disc, like the second rule on a card */}
      <span className="absolute rounded-full" style={{ inset: 0, border: `1px solid ${color}`, opacity: 0.28 }} />
      <span
        className="absolute rounded-full"
        style={{ inset: '13%', background: 'var(--bg2)', border: `1px solid ${color}`, opacity: 0.55 }}
      />
      <Botanical name={plant} size={28} color={color} opacity={0.85} rotate={tilt} style={{ position: 'relative' }} />
    </div>
  )
}

export default function Timeline() {
  return (
    <section className="relative" style={flavorStyle(FLAVORS.blossom, { padding: 'clamp(12px, 1.6vh, 20px) 18px' })}>
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

        {/* the vine spans the columns, so it only means anything once they exist */}
        <FadeUp delay={0.1} className="hidden md:block">
          <ScheduleVine colors={STAGES.map((s) => s.color)} />
        </FadeUp>

        <ol
          className="grid md:grid-cols-3"
          role="list"
          style={{
            gap: '0 clamp(16px, 3vw, 34px)',
            margin: '6px 0 0',
            padding: 0,
            listStyle: 'none',
            ['--seal' as string]: 'clamp(46px, 13vw, 66px)',
          }}
        >
          {SCHEDULE.map((ev, i) => {
            const stage = STAGES[i]
            const last = i === SCHEDULE.length - 1
            return (
              <li key={ev.time}>
                <FadeUp delay={0.12 + i * 0.14}>
                  <div className="relative overflow-hidden flex items-start md:block" style={{ gap: 16 }}>
                    {/* the same plant again, printed large and faint under the sheet */}
                    <div
                      className="absolute pointer-events-none"
                      aria-hidden="true"
                      style={{ bottom: '-34%', insetInlineEnd: '-6%' }}
                    >
                      <Botanical name={stage.plant} size={132} color={stage.color} opacity={stage.wash} rotate={stage.tilt * 2} />
                    </div>

                    <div className="relative shrink-0 md:w-fit md:mx-auto">
                      <PressedSeal plant={stage.plant} color={stage.color} tilt={stage.tilt} />
                    </div>

                    <div className="relative flex-1 md:text-center" style={{ minWidth: 0 }}>
                      <div className="flex items-baseline md:justify-center" style={{ gap: 10, flexWrap: 'wrap' }}>
                        <span
                          className="font-serif2 tabular"
                          dir="ltr"
                          style={{ fontWeight: 600, fontSize: 'clamp(1.5rem, 5.6vw, 2.1rem)', color: 'var(--champ)', lineHeight: 1.1 }}
                        >
                          {ev.time}
                        </span>
                        {'range' in ev && ev.range && (
                          <span
                            className="tabular"
                            dir="ltr"
                            style={{ color: 'var(--muted)', fontSize: 'clamp(0.74rem, 2.5vw, 0.85rem)', letterSpacing: '0.05em' }}
                          >
                            {ev.range}
                          </span>
                        )}
                      </div>

                      <div className="flex items-center md:justify-center" style={{ gap: 8, flexWrap: 'wrap', marginTop: 2 }}>
                        <h3 className="font-display" style={{ fontWeight: 900, fontSize: 'clamp(1.15rem, 3.9vw, 1.6rem)', margin: 0, color: 'var(--champ2)' }}>
                          {ev.title}
                        </h3>
                        {'tag' in ev && ev.tag && (
                          <span
                            style={{
                              padding: '2px 9px',
                              borderRadius: 999,
                              border: `1px solid ${stage.ink}`,
                              color: stage.ink,
                              fontSize: 'clamp(0.68rem, 2.3vw, 0.8rem)',
                              fontWeight: 700,
                              letterSpacing: '0.04em',
                            }}
                          >
                            {ev.tag}
                          </span>
                        )}
                      </div>

                      <p style={{ color: 'var(--muted)', margin: '4px 0 0', fontSize: 'clamp(0.82rem, 2.6vw, 0.95rem)', lineHeight: 1.55 }}>
                        {ev.desc}
                      </p>
                    </div>
                  </div>

                  {/* stacked only: the stem carries the eye down to the next stage */}
                  {!last && (
                    <div className="md:hidden" style={{ width: 'var(--seal)' }}>
                      <StemLink height={24} />
                    </div>
                  )}
                </FadeUp>
              </li>
            )
          })}
        </ol>
      </div>
    </section>
  )
}
