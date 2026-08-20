import { Eyebrow, FadeUp, MaskLine } from './shared'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { Flourish, CornerSpray, GarlandDivider } from './Flourish'
import { PressedBotanicals, SprigRow } from './Botanicals'

export default function Message() {
  return (
    <section className="relative" style={flavorStyle(FLAVORS.blossom, { color: 'var(--ink)' })}>
      <PressedBotanicals set="message" />
      <div className="absolute pointer-events-none" style={{ top: 10, insetInlineStart: 8, opacity: 0.85 }} aria-hidden="true">
        <CornerSpray size={72} />
      </div>
      <div className="absolute pointer-events-none" style={{ top: 10, insetInlineEnd: 8, opacity: 0.85 }} aria-hidden="true">
        <CornerSpray size={72} flip />
      </div>
      <div className="mx-auto text-center" style={{ maxWidth: '44rem', padding: 'clamp(26px, 3.4vh, 40px) 24px' }}>
        <FadeUp>
          <Flourish color="#8fa3e8" />
          <Eyebrow ink>You&rsquo;re Invited</Eyebrow>
        </FadeUp>

        <FadeUp delay={0.1}>
          <span
            className="font-serif2 block"
            aria-hidden="true"
            style={{ fontSize: 'clamp(4rem, 12vw, 6rem)', lineHeight: 0.6, color: 'var(--champ2)', marginTop: 34 }}
          >
            &ldquo;
          </span>
        </FadeUp>

        <MaskLine delay={0.05}>
          <h2 className="font-display text-balance" style={{ fontWeight: 900, fontSize: 'clamp(2rem, 7vw, 3rem)', margin: '10px 0 22px', color: 'var(--champ2)' }}>
            בואו לחגוג איתנו
          </h2>
        </MaskLine>

        <div style={{ fontSize: 'clamp(1.05rem, 3.6vw, 1.2rem)', lineHeight: 1.9 }}>
          <MaskLine delay={0.12}>
            <p style={{ margin: 0 }}>החלום קורם עור וגידים — אנחנו מתחתנים!</p>
          </MaskLine>
          <MaskLine delay={0.2}>
            <p style={{ margin: '10px 0 0' }}>ואין דרך יפה יותר לפתוח את הפרק הבא שלנו</p>
          </MaskLine>
          <MaskLine delay={0.28}>
            <p style={{ margin: '10px 0 0' }}>מאשר ערב אחד בלתי נשכח, עם האנשים שאנחנו הכי אוהבים.</p>
          </MaskLine>
          <MaskLine delay={0.36}>
            <p style={{ margin: '10px 0 0' }}>בואו רעבים, בואו נוצצים — ותשאירו כוח ברגליים.</p>
          </MaskLine>
        </div>

        <FadeUp delay={0.15}>
          <GarlandDivider />
        </FadeUp>
      </div>
    </section>
  )
}
