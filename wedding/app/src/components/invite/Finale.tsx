import { FadeUp } from './shared'
import { SparklesText } from '@/components/vendor/sparkles-text'
import { VENUE, ADDR, DATE_LABEL } from '@/lib/wedding'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { Flourish, CornerSpray } from './Flourish'
import { PressedBotanicals } from './Botanicals'

export default function Finale() {
  return (
    <footer className="relative text-center" style={flavorStyle(FLAVORS.blossom, { padding: '48px 20px 100px' })}>
      <PressedBotanicals set="finale" />
      <div className="absolute pointer-events-none" style={{ bottom: 8, insetInlineStart: 8, opacity: 0.85 }} aria-hidden="true">
        <CornerSpray size={76} flipY />
      </div>
      <div className="absolute pointer-events-none" style={{ bottom: 8, insetInlineEnd: 8, opacity: 0.85 }} aria-hidden="true">
        <CornerSpray size={76} flip flipY />
      </div>
      <FadeUp>
        <Flourish color="#e58c54" />
        <p
          className="font-script"
          dir="ltr"
          style={{
            fontSize: 'clamp(2.5rem, 9.5vw, 4rem)',
            color: 'var(--champ2)',
            textShadow: '0 0 30px rgba(196,118,143,.3)',
            margin: 0,
            lineHeight: 1.3,
          }}
        >
          <SparklesText>See you on the dance floor</SparklesText>
        </p>
      </FadeUp>
      <FadeUp delay={0.1}>
        <p className="font-display" style={{ fontWeight: 900, fontSize: 'clamp(1.3rem, 4.6vw, 1.7rem)', margin: '16px 0 0' }}>
          מחכים לכם <span style={{ color: 'var(--p1)' }}>♥</span> שחף &amp; תומר
        </p>
      </FadeUp>
      <FadeUp delay={0.18}>
        <div
          className="flex justify-center flex-wrap font-serif2"
          dir="ltr"
          style={{
            gap: '10px 26px',
            marginTop: 44,
            paddingTop: 22,
            borderTop: '1px solid var(--line)',
            maxWidth: '40rem',
            marginInline: 'auto',
            color: 'var(--muted)',
            fontWeight: 600,
            fontSize: '0.78rem',
            letterSpacing: '0.3em',
          }}
        >
          <span>SHACHAF &amp; TOMER</span>
          <span className="tabular">{DATE_LABEL}</span>
          <span dir="rtl" style={{ letterSpacing: '0.12em', fontFamily: 'var(--font-body)' }}>
            {VENUE} · {ADDR}
          </span>
        </div>
        {/* קרדיט לסילואטים הבוטניים — נדרש ברישיון CC BY 3.0 */}
        <p
          dir="ltr"
          style={{
            marginTop: 14,
            color: 'var(--muted)',
            opacity: 0.5,
            fontSize: '0.62rem',
            letterSpacing: '0.06em',
          }}
        >
          Botanical silhouettes:{' '}
          <a href="https://game-icons.net" target="_blank" rel="noopener" style={{ color: 'inherit' }}>
            game-icons.net
          </a>{' '}
          · CC BY 3.0
        </p>
      </FadeUp>
    </footer>
  )
}
