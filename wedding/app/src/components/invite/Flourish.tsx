/**
 * Hand-drawn-style botanical set — original SVGs in the spirit of a
 * painted invitation suite: sage stems, pastel buds, golden hearts.
 */

const STEM = '#7f9c5e'
const CORAL = '#e58c54'
const BLOSSOM = '#d9769b'
const PERIWINKLE = '#8fa3e8'
const GOLD = '#d9a94f'

/** symmetric floral spray above headings, buds tinted per section */
export function Flourish({ color = BLOSSOM, width = 132 }: { color?: string; width?: number }) {
  return (
    <svg
      width={width}
      height={width * 0.28}
      viewBox="0 0 132 37"
      fill="none"
      aria-hidden="true"
      style={{ display: 'block', margin: '0 auto 10px' }}
    >
      <path d="M14 30 C 34 27, 52 25, 64 15" stroke={STEM} strokeWidth="1.4" strokeLinecap="round" />
      <path d="M118 30 C 98 27, 80 25, 68 15" stroke={STEM} strokeWidth="1.4" strokeLinecap="round" />
      <ellipse cx="34" cy="26.5" rx="5" ry="1.9" transform="rotate(-16 34 26.5)" fill={STEM} opacity="0.75" />
      <ellipse cx="52" cy="22.5" rx="4.6" ry="1.7" transform="rotate(-26 52 22.5)" fill={STEM} opacity="0.75" />
      <ellipse cx="98" cy="26.5" rx="5" ry="1.9" transform="rotate(16 98 26.5)" fill={STEM} opacity="0.75" />
      <ellipse cx="80" cy="22.5" rx="4.6" ry="1.7" transform="rotate(26 80 22.5)" fill={STEM} opacity="0.75" />
      <circle cx="14" cy="29" r="3" fill={color} opacity="0.85" />
      <circle cx="118" cy="29" r="3" fill={color} opacity="0.85" />
      <circle cx="43" cy="23.5" r="2.1" fill={color} opacity="0.6" />
      <circle cx="89" cy="23.5" r="2.1" fill={color} opacity="0.6" />
      <circle cx="66" cy="9.5" r="3.1" fill={color} opacity="0.9" />
      <circle cx="61" cy="12.8" r="3.1" fill={color} opacity="0.9" />
      <circle cx="71" cy="12.8" r="3.1" fill={color} opacity="0.9" />
      <circle cx="63" cy="17" r="3.1" fill={color} opacity="0.9" />
      <circle cx="69" cy="17" r="3.1" fill={color} opacity="0.9" />
      <circle cx="66" cy="13.6" r="2.3" fill={GOLD} />
    </svg>
  )
}

/**
 * festive corner spray: arched stem with a coral trumpet flower,
 * a hanging periwinkle bell, leaves and buds.
 * flip mirrors horizontally, flipY turns it upside-down for bottom corners.
 */
export function CornerSpray({ flip = false, flipY = false, size = 96 }: { flip?: boolean; flipY?: boolean; size?: number }) {
  const t = `${flip ? 'scaleX(-1)' : ''} ${flipY ? 'scaleY(-1)' : ''}`.trim()
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 110 110"
      fill="none"
      aria-hidden="true"
      style={{ display: 'block', transform: t || undefined }}
    >
      <path d="M8 104 C 20 78, 30 52, 58 30" stroke={STEM} strokeWidth="1.7" strokeLinecap="round" />
      <path d="M26 70 C 38 62, 48 60, 60 62" stroke={STEM} strokeWidth="1.4" strokeLinecap="round" />
      <ellipse cx="20" cy="83" rx="7" ry="2.6" transform="rotate(-52 20 83)" fill={STEM} opacity="0.8" />
      <ellipse cx="30" cy="62" rx="6.4" ry="2.4" transform="rotate(-44 30 62)" fill={STEM} opacity="0.8" />
      <ellipse cx="44" cy="44" rx="6" ry="2.2" transform="rotate(-38 44 44)" fill={STEM} opacity="0.8" />
      <ellipse cx="66" cy="22" rx="9.5" ry="3.6" transform="rotate(-38 66 22)" fill={CORAL} opacity="0.92" />
      <ellipse cx="60" cy="17" rx="9.5" ry="3.6" transform="rotate(-68 60 17)" fill={CORAL} opacity="0.85" />
      <ellipse cx="70" cy="30" rx="9.5" ry="3.6" transform="rotate(-10 70 30)" fill={CORAL} opacity="0.85" />
      <circle cx="59" cy="28" r="2.6" fill={GOLD} />
      <path d="M59 28l10-4M59 28l9 2M59 28l3-9" stroke={GOLD} strokeWidth="0.9" opacity="0.8" />
      <path d="M60 62 q 4 5 2 9" stroke={STEM} strokeWidth="1.2" strokeLinecap="round" />
      <path d="M57 71 q -1.5 8 2 9 l 1.5-1.5 1.5 2 1.5-2 1.5 1.5 q 3.5-1 2-9 q -5-3.5 -10 0" fill={PERIWINKLE} opacity="0.9" />
      <circle cx="38" cy="90" r="3.2" fill={BLOSSOM} opacity="0.85" />
      <circle cx="50" cy="76" r="2.4" fill={BLOSSOM} opacity="0.6" />
      <circle cx="76" cy="44" r="2.6" fill={CORAL} opacity="0.6" />
    </svg>
  )
}

/** slim vertical vine with hanging bells — flanks the hero title */
export function SideVine({ flip = false, height = 150 }: { flip?: boolean; height?: number }) {
  return (
    <svg
      width={height * 0.33}
      height={height}
      viewBox="0 0 50 150"
      fill="none"
      aria-hidden="true"
      style={{ display: 'block', transform: flip ? 'scaleX(-1)' : undefined }}
    >
      <path d="M14 4 C 30 34, 12 62, 24 92 C 32 112, 22 132, 26 146" stroke={STEM} strokeWidth="1.6" strokeLinecap="round" />
      <ellipse cx="22" cy="28" rx="6.4" ry="2.3" transform="rotate(48 22 28)" fill={STEM} opacity="0.8" />
      <ellipse cx="17" cy="72" rx="6.4" ry="2.3" transform="rotate(-56 17 72)" fill={STEM} opacity="0.8" />
      <ellipse cx="28" cy="118" rx="6" ry="2.2" transform="rotate(50 28 118)" fill={STEM} opacity="0.8" />
      <path d="M25 48 q 8 2 9 7" stroke={STEM} strokeWidth="1.1" strokeLinecap="round" />
      <path d="M30 56 q -1.4 7 1.8 8 l 1.4-1.4 1.4 1.8 1.4-1.8 1.4 1.4 q 3.2-1 1.8-8 q -4.5-3.2 -9.2 0" fill={PERIWINKLE} opacity="0.9" />
      <path d="M20 96 q -7 2.5 -8 7" stroke={STEM} strokeWidth="1.1" strokeLinecap="round" />
      <path d="M7.5 104 q -1.4 7 1.8 8 l 1.4-1.4 1.4 1.8 1.4-1.8 1.4 1.4 q 3.2-1 1.8-8 q -4.5-3.2 -9.2 0" fill={BLOSSOM} opacity="0.85" />
      <circle cx="12" cy="12" r="2.8" fill={CORAL} opacity="0.85" />
      <circle cx="33" cy="86" r="2.2" fill={BLOSSOM} opacity="0.6" />
      <circle cx="18" cy="140" r="2.5" fill={PERIWINKLE} opacity="0.7" />
    </svg>
  )
}

/** wide garland divider: mirrored stems meeting a center flower */
export function GarlandDivider({ width = 260 }: { width?: number }) {
  return (
    <svg
      width={width}
      height={width * 0.17}
      viewBox="0 0 260 44"
      fill="none"
      aria-hidden="true"
      style={{ display: 'block', margin: '18px auto 0' }}
    >
      <path d="M6 14 C 50 34, 90 36, 122 24" stroke={STEM} strokeWidth="1.5" strokeLinecap="round" />
      <path d="M254 14 C 210 34, 170 36, 138 24" stroke={STEM} strokeWidth="1.5" strokeLinecap="round" />
      <ellipse cx="40" cy="26" rx="6" ry="2.2" transform="rotate(18 40 26)" fill={STEM} opacity="0.8" />
      <ellipse cx="78" cy="32" rx="5.6" ry="2.1" transform="rotate(8 78 32)" fill={STEM} opacity="0.8" />
      <ellipse cx="220" cy="26" rx="6" ry="2.2" transform="rotate(-18 220 26)" fill={STEM} opacity="0.8" />
      <ellipse cx="182" cy="32" rx="5.6" ry="2.1" transform="rotate(-8 182 32)" fill={STEM} opacity="0.8" />
      {/* hanging bells at the thirds */}
      <path d="M58 29 q 1 4 0 6" stroke={STEM} strokeWidth="1" strokeLinecap="round" />
      <path d="M53.5 35 q -1.2 6 1.6 7 l 1.2-1.2 1.2 1.6 1.2-1.6 1.2 1.2 q 2.8-1 1.6-7 q -4-2.8 -8 0" fill={PERIWINKLE} opacity="0.9" />
      <path d="M202 29 q -1 4 0 6" stroke={STEM} strokeWidth="1" strokeLinecap="round" />
      <path d="M197.5 35 q -1.2 6 1.6 7 l 1.2-1.2 1.2 1.6 1.2-1.6 1.2 1.2 q 2.8-1 1.6-7 q -4-2.8 -8 0" fill={CORAL} opacity="0.88" />
      {/* center five-petal flower */}
      <circle cx="130" cy="14" r="3.4" fill={BLOSSOM} opacity="0.9" />
      <circle cx="124.5" cy="17.6" r="3.4" fill={BLOSSOM} opacity="0.9" />
      <circle cx="135.5" cy="17.6" r="3.4" fill={BLOSSOM} opacity="0.9" />
      <circle cx="126.7" cy="22.2" r="3.4" fill={BLOSSOM} opacity="0.9" />
      <circle cx="133.3" cy="22.2" r="3.4" fill={BLOSSOM} opacity="0.9" />
      <circle cx="130" cy="18.5" r="2.5" fill={GOLD} />
      {/* end buds */}
      <circle cx="6" cy="13" r="2.8" fill={CORAL} opacity="0.85" />
      <circle cx="254" cy="13" r="2.8" fill={PERIWINKLE} opacity="0.85" />
      <circle cx="100" cy="29" r="2.2" fill={BLOSSOM} opacity="0.6" />
      <circle cx="160" cy="29" r="2.2" fill={BLOSSOM} opacity="0.6" />
    </svg>
  )
}

/**
 * The stem that runs the length of the schedule, in place of a rail with dots.
 * It waves between the three stages and carries a five-petal bloom over the
 * middle of each card, so the blooms sit exactly above the seals below them.
 *
 * `colors` arrives in reading order, and Hebrew reads right to left, so the
 * first stage takes the rightmost bloom. SVG coordinates do not flip with
 * `dir`, so the order is reversed here rather than in the markup.
 */
export function ScheduleVine({ colors }: { colors: readonly string[] }) {
  const CY = 26
  const at = [760, 450, 140] // bloom centres, right to left, pulled in by the grid gap
  const leaves: [number, number, number][] = [
    // x, y, rotation — one at the middle of each crest and trough of the wave
    [79, 19, -16], [223, 33, 14], [377, 19, -12], [523, 33, 16], [677, 19, -14], [821, 33, 12],
  ]
  /* drawn heavy for its box on purpose: the stem spans the full width of the
     schedule, so on a phone it renders about 18px tall and anything finer than
     this disappears */
  return (
    <svg
      viewBox="0 0 900 52"
      fill="none"
      aria-hidden="true"
      style={{ display: 'block', width: '100%', height: 'auto', margin: '0 auto' }}
    >
      {/* one stem, one colour — the blooms carry the stage colours */}
      <path
        d={`M8 ${CY} Q79 12 150 ${CY} T300 ${CY} T450 ${CY} T600 ${CY} T750 ${CY} T892 ${CY}`}
        stroke={STEM}
        strokeWidth="2.4"
        strokeLinecap="round"
        opacity="0.75"
      />
      {leaves.map(([x, y, r], i) => (
        <ellipse key={i} cx={x} cy={y} rx="15" ry="5.5" transform={`rotate(${r} ${x} ${y})`} fill={STEM} opacity="0.55" />
      ))}
      {at.map((x, i) => {
        const c = colors[i] ?? BLOSSOM
        return (
          <g key={x}>
            <circle cx={x} cy={CY - 6.5} r="7" fill={c} opacity="0.9" />
            <circle cx={x - 7} cy={CY - 1.5} r="7" fill={c} opacity="0.9" />
            <circle cx={x + 7} cy={CY - 1.5} r="7" fill={c} opacity="0.9" />
            <circle cx={x - 4.4} cy={CY + 6} r="7" fill={c} opacity="0.9" />
            <circle cx={x + 4.4} cy={CY + 6} r="7" fill={c} opacity="0.9" />
            <circle cx={x} cy={CY} r="4.2" fill={GOLD} />
          </g>
        )
      })}
      {/* the stem carries on past the outer blooms rather than stopping dead */}
      <circle cx="8" cy={CY} r="4" fill={CORAL} opacity="0.8" />
      <circle cx="892" cy={CY} r="4" fill={CORAL} opacity="0.8" />
    </svg>
  )
}

/**
 * The stem that grows from one stage to the next when the schedule is stacked.
 * Fixed height so the leaves never stretch: the rail column is a constant
 * width, and this simply fills the gap between two rows.
 */
export function StemLink({ height = 34 }: { height?: number }) {
  return (
    <svg
      viewBox="0 0 24 34"
      fill="none"
      aria-hidden="true"
      style={{ display: 'block', width: 24, height, margin: '0 auto' }}
    >
      <path d="M12 0 C 15 9, 9 20, 12 34" stroke={STEM} strokeWidth="2" strokeLinecap="round" opacity="0.7" />
      <ellipse cx="5.5" cy="11" rx="6" ry="2.6" transform="rotate(-24 5.5 11)" fill={STEM} opacity="0.5" />
      <ellipse cx="18.5" cy="23" rx="6" ry="2.6" transform="rotate(24 18.5 23)" fill={STEM} opacity="0.5" />
    </svg>
  )
}

/** tiny painted disco ball on a string */
export function MiniDiscoBall({ size = 34 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size * 1.4}
      viewBox="0 0 34 48"
      fill="none"
      aria-hidden="true"
      style={{ display: 'block', margin: '0 auto 8px' }}
    >
      <path d="M17 0v9" stroke="#b98a3c" strokeWidth="1.3" strokeLinecap="round" />
      <circle cx="17" cy="26" r="15" fill="#f3b04b" />
      <circle cx="17" cy="26" r="15" fill="url(#miniShade)" />
      <path d="M17 11v30M2.6 22h28.8M4 32h26M4.6 17.5h24.8" stroke="#c77e28" strokeWidth="1" opacity="0.75" />
      <path d="M8 13.6c5.4 4.2 12.6 4.2 18 0M8 38.4c5.4-4.2 12.6-4.2 18 0" stroke="#c77e28" strokeWidth="1" opacity="0.75" />
      <circle cx="11.5" cy="19" r="2.4" fill="#ffe6b3" opacity="0.95" />
      <path d="M28 6.5l1.1 2.4 2.4 1.1-2.4 1.1-1.1 2.4-1.1-2.4-2.4-1.1 2.4-1.1z" fill="#e8b45c" />
      <defs>
        <radialGradient id="miniShade" cx="0.35" cy="0.3" r="1">
          <stop offset="0" stopColor="#ffffff" stopOpacity="0.55" />
          <stop offset="0.55" stopColor="#f3b04b" stopOpacity="0" />
          <stop offset="1" stopColor="#8a5a14" stopOpacity="0.45" />
        </radialGradient>
      </defs>
    </svg>
  )
}
