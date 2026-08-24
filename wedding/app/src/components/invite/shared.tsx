import { useRef, type ReactNode, type CSSProperties } from 'react'
import { motion, useInView, useMotionValue, useSpring, useReducedMotion } from 'framer-motion'

export const EASE = [0.22, 1, 0.36, 1] as const

export function Eyebrow({ children, ink = false }: { children: ReactNode; ink?: boolean }) {
  return <p className={'eyebrow' + (ink ? ' eyebrow--ink' : '')}>{children}</p>
}

/** fades + rises when scrolled into view */
export function FadeUp({
  children,
  delay = 0,
  y = 30,
  className,
  style,
  /* the default asks an element to sit 12% inside the viewport before it
     reveals. The last block on the page can never do that — scrolled all the
     way down it still sits in the bottom band — so it has to opt out. */
  margin = '-12% 0px',
}: {
  children: ReactNode
  delay?: number
  y?: number
  className?: string
  style?: CSSProperties
  margin?: `${number}px` | `${number}% ${number}px` | `-${number}% ${number}px`
}) {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin })
  return (
    <motion.div
      ref={ref}
      className={className}
      style={style}
      initial={{ opacity: 0, y }}
      animate={inView ? { opacity: 1, y: 0 } : undefined}
      transition={{ duration: 0.9, ease: EASE, delay }}
    >
      {children}
    </motion.div>
  )
}

/** text line that slides up from behind a mask */
export function MaskLine({
  children,
  delay = 0,
  className,
}: {
  children: ReactNode
  delay?: number
  className?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin: '-10% 0px' })
  return (
    <div ref={ref} style={{ overflow: 'hidden' }} className={className}>
      <motion.div
        initial={{ y: '112%' }}
        animate={inView ? { y: 0 } : undefined}
        transition={{ duration: 0.95, ease: EASE, delay }}
      >
        {children}
      </motion.div>
    </div>
  )
}

/** anchor that leans toward the pointer (fine pointers only) */
export function MagneticLink({
  href,
  download,
  primary = false,
  children,
}: {
  href: string
  download?: string
  primary?: boolean
  children: ReactNode
}) {
  const x = useMotionValue(0)
  const y = useMotionValue(0)
  const sx = useSpring(x, { stiffness: 220, damping: 18 })
  const sy = useSpring(y, { stiffness: 220, damping: 18 })
  const rm = useReducedMotion()

  const onMove = (e: React.MouseEvent<HTMLAnchorElement>) => {
    if (rm || !window.matchMedia('(pointer: fine)').matches) return
    const r = e.currentTarget.getBoundingClientRect()
    x.set((e.clientX - (r.left + r.width / 2)) * 0.22)
    y.set((e.clientY - (r.top + r.height / 2)) * 0.32)
  }
  const onLeave = () => {
    x.set(0)
    y.set(0)
  }

  return (
    <motion.a
      href={href}
      download={download}
      target={download ? undefined : '_blank'}
      rel={download ? undefined : 'noopener'}
      className={'btn' + (primary ? ' btn--primary' : '')}
      style={{ x: sx, y: sy }}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      data-cursor
    >
      {children}
      <span className="btn__shine" aria-hidden="true" />
    </motion.a>
  )
}

export const IconNav = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M3 11l18-8-8 18-2.5-7.5L3 11z" />
  </svg>
)
export const IconCal = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <rect x="3" y="5" width="18" height="16" rx="3" />
    <path d="M3 10h18M8 3v4M16 3v4" />
  </svg>
)
export const IconDown = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M4 20h16" />
  </svg>
)
