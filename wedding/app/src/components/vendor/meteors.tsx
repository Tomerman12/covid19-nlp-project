/**
 * Meteors — adapted from Magic UI (magicui.design), MIT license.
 * Keyframe lives in index.css; rose default color.
 */
import { useEffect, useState, type CSSProperties } from 'react'

interface MeteorsProps {
  number?: number
  minDelay?: number
  maxDelay?: number
  minDuration?: number
  maxDuration?: number
  angle?: number
  color?: string
}

export const Meteors = ({
  number = 12,
  minDelay = 0.2,
  maxDelay = 1.6,
  minDuration = 3,
  maxDuration = 9,
  angle = 215,
  color = 'rgba(196,118,143,.85)',
}: MeteorsProps) => {
  const [meteorStyles, setMeteorStyles] = useState<CSSProperties[]>([])

  useEffect(() => {
    const styles = [...new Array(number)].map(() => ({
      '--angle': -angle + 'deg',
      top: '-5%',
      left: `calc(0% + ${Math.floor(Math.random() * window.innerWidth)}px)`,
      animationDelay: Math.random() * (maxDelay - minDelay) + minDelay + 's',
      animationDuration: Math.floor(Math.random() * (maxDuration - minDuration) + minDuration) + 's',
    }))
    setMeteorStyles(styles as CSSProperties[])
  }, [number, minDelay, maxDelay, minDuration, maxDuration, angle])

  return (
    <>
      {meteorStyles.map((style, idx) => (
        <span
          key={idx}
          aria-hidden="true"
          className="meteor pointer-events-none absolute"
          style={{
            ...style,
            width: 2.5,
            height: 2.5,
            borderRadius: '50%',
            background: color,
            boxShadow: '0 0 0 1px rgba(255,255,255,.08)',
          }}
        >
          <span
            className="pointer-events-none absolute"
            style={{
              top: '50%',
              transform: 'translateY(-50%)',
              width: 56,
              height: 1,
              background: `linear-gradient(to right, ${color}, transparent)`,
            }}
          />
        </span>
      ))}
    </>
  )
}
