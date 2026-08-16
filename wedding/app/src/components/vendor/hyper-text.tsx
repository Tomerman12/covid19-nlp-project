/**
 * HyperText — adapted from Magic UI (magicui.design), MIT license.
 * Span-only scramble that inherits the invitation's type.
 */
import { useEffect, useRef, useState } from 'react'
import { useReducedMotion } from 'framer-motion'

interface HyperTextProps {
  children: string
  className?: string
  duration?: number
  delay?: number
  animateOnHover?: boolean
  characterSet?: readonly string[]
  dir?: 'ltr' | 'rtl'
}

const DEFAULT_SET = Object.freeze('ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''))
const rand = (max: number) => Math.floor(Math.random() * max)

export function HyperText({
  children,
  className,
  duration = 900,
  delay = 0,
  animateOnHover = true,
  characterSet = DEFAULT_SET,
  dir,
}: HyperTextProps) {
  const rm = useReducedMotion()
  const [displayText, setDisplayText] = useState<string[]>(() => children.split(''))
  const [isAnimating, setIsAnimating] = useState(false)
  const iterationCount = useRef(0)

  const trigger = () => {
    if (animateOnHover && !isAnimating && !rm) {
      iterationCount.current = 0
      setIsAnimating(true)
    }
  }

  useEffect(() => {
    if (rm) return
    const t = setTimeout(() => setIsAnimating(true), delay)
    return () => clearTimeout(t)
  }, [delay, rm])

  useEffect(() => {
    if (!isAnimating) return
    let raf: number | null = null
    const maxIterations = children.length
    const startTime = performance.now()

    const animate = (now: number) => {
      const progress = Math.min((now - startTime) / duration, 1)
      iterationCount.current = progress * maxIterations
      setDisplayText((current) =>
        current.map((letter, index) =>
          letter === ' ' || letter === '.' || letter === ':'
            ? children[index]
            : index <= iterationCount.current
              ? children[index]
              : characterSet[rand(characterSet.length)],
        ),
      )
      if (progress < 1) raf = requestAnimationFrame(animate)
      else setIsAnimating(false)
    }

    raf = requestAnimationFrame(animate)
    return () => {
      if (raf !== null) cancelAnimationFrame(raf)
    }
  }, [children, duration, isAnimating, characterSet])

  if (rm) {
    return (
      <span className={className} dir={dir}>
        {children}
      </span>
    )
  }

  return (
    <span className={className} dir={dir} onMouseEnter={trigger}>
      {displayText.join('')}
    </span>
  )
}

export const DIGITS_SET = Object.freeze('0123456789'.split(''))
