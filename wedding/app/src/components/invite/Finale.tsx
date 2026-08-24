import { FadeUp } from './shared'
import { DATE_LABEL } from '@/lib/wedding'
import { FLAVORS, flavorStyle } from '@/lib/flavors'
import { CornerSpray } from './Flourish'
import { PressedBotanicals } from './Botanicals'

export default function Finale() {
  return (
    <footer className="relative text-center" style={flavorStyle(FLAVORS.blossom, { padding: '22px 20px 32px' })}>
      <PressedBotanicals set="finale" />
      <div className="absolute pointer-events-none" style={{ bottom: 8, insetInlineStart: 8, opacity: 0.85 }} aria-hidden="true">
        <CornerSpray size={76} flipY />
      </div>
      <div className="absolute pointer-events-none" style={{ bottom: 8, insetInlineEnd: 8, opacity: 0.85 }} aria-hidden="true">
        <CornerSpray size={76} flip flipY />
      </div>
      <FadeUp delay={0.18} margin="0px">
        <div
          className="flex justify-center flex-wrap font-serif2"
          dir="ltr"
          style={{
            gap: '10px 26px',
            marginTop: 0,
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
          {/* the venue section right above owns the address — this is a sign-off */}
          <span>SHACHAF &amp; TOMER</span>
          <span className="tabular">{DATE_LABEL}</span>
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
