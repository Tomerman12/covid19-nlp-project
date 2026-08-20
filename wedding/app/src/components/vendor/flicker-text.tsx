/**
 * FlickerText — adapted from the Framer "Flicker Text" component
 * (Originkit) supplied by the couple: neon-ignition word flicker with
 * outline flashes + random letter dips, honoring reduced motion.
 */
import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { useReducedMotion } from 'framer-motion'

type Phase = 'invisible' | 'outline' | 'filled'

const easeInOut = (t: number) => (t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t))

function generateTimings(count: number, totalMs: number): number[] {
  const intervals: number[] = []
  let prev = 0
  for (let i = 1; i <= count; i++) {
    const cur = easeInOut(i / count) * totalMs
    intervals.push(Math.max(0, cur - prev))
    prev = cur
  }
  return intervals
}

const IGNITION: Phase[] = ['invisible', 'outline', 'invisible', 'outline', 'invisible', 'filled', 'invisible', 'outline', 'invisible', 'filled']

interface FlickerTextProps {
  children: string
  delay?: number
  duration?: number
  strokeColor?: string
  strokeWidth?: number
  replayOnHover?: boolean
  className?: string
}

export function FlickerText({
  children,
  delay = 0,
  duration = 1500,
  strokeColor = 'var(--champ)',
  strokeWidth = 1.5,
  replayOnHover = true,
  className,
}: FlickerTextProps) {
  const rm = useReducedMotion()
  const [phase, setPhase] = useState<Phase>(rm ? 'filled' : 'invisible')
  const [dimmed, setDimmed] = useState<Set<number>>(new Set())
  const timers = useRef<ReturnType<typeof setTimeout>[]>([])
  const playing = useRef(false)

  const clearTimers = () => {
    timers.current.forEach(clearTimeout)
    timers.current = []
  }

  const run = (dly: number, dur: number) => {
    if (rm || playing.current) return
    playing.current = true
    clearTimers()

    const intervals = generateTimings(IGNITION.length, dur)
    let cursor = dly
    IGNITION.forEach((p, i) => {
      timers.current.push(window.setTimeout(() => setPhase(p), cursor) as unknown as ReturnType<typeof setTimeout>)
      cursor += intervals[i] ?? 0
    })

    const idx = children.split('').reduce<number[]>((acc, c, i) => {
      if (c.trim() !== '') acc.push(i)
      return acc
    }, [])
    let tick = dly + 140
    while (tick < dly + dur) {
      const t = tick
      timers.current.push(
        window.setTimeout(() => {
          const n = Math.random() < 0.5 ? 1 : 2
          const pick = [...idx].sort(() => Math.random() - 0.5).slice(0, n)
          setDimmed(new Set(pick))
        }, t) as unknown as ReturnType<typeof setTimeout>,
      )
      timers.current.push(window.setTimeout(() => setDimmed(new Set()), t + 70) as unknown as ReturnType<typeof setTimeout>)
      tick += 170
    }

    timers.current.push(
      window.setTimeout(() => {
        setPhase('filled')
        setDimmed(new Set())
        playing.current = false
      }, dly + dur) as unknown as ReturnType<typeof setTimeout>,
    )
  }

  useEffect(() => {
    if (rm) {
      setPhase('filled')
      return
    }
    run(delay, duration)
    return clearTimers
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rm])

  const phaseStyle: CSSProperties =
    phase === 'invisible'
      ? { opacity: 0, textShadow: 'none' }
      : phase === 'outline'
        ? {
            color: 'transparent',
            WebkitTextFillColor: 'transparent',
            WebkitTextStroke: `${strokeWidth}px ${strokeColor}`,
            textShadow: 'none',
          }
        : {}

  return (
    <span
      className={className}
      style={{ display: 'inline-block', ...phaseStyle }}
      onMouseEnter={replayOnHover && !rm ? () => run(0, 750) : undefined}
    >
      {phase !== 'filled' || dimmed.size === 0
        ? children
        : children.split('').map((ch, i) => (
            <span key={i} style={dimmed.has(i) ? { opacity: 0.25 } : undefined}>
              {ch}
            </span>
          ))}
    </span>
  )
}
