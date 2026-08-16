import { useEffect, useState } from 'react'
import { motion, useMotionValue, useSpring, useReducedMotion } from 'framer-motion'

/** rose dot + trailing ring, desktop only */
export default function Cursor({ onActiveChange }: { onActiveChange: (active: boolean) => void }) {
  const rm = useReducedMotion()
  const [enabled, setEnabled] = useState(false)
  const [hot, setHot] = useState(false)

  const x = useMotionValue(-100)
  const y = useMotionValue(-100)
  const rx = useSpring(x, { stiffness: 260, damping: 24 })
  const ry = useSpring(y, { stiffness: 260, damping: 24 })

  useEffect(() => {
    const fine = window.matchMedia('(pointer: fine)').matches
    if (!fine || rm) return
    setEnabled(true)
    onActiveChange(true)

    const move = (e: PointerEvent) => {
      x.set(e.clientX)
      y.set(e.clientY)
    }
    const over = (e: Event) => {
      const t = e.target as HTMLElement | null
      setHot(!!t?.closest?.('a, button, [data-cursor]'))
    }
    addEventListener('pointermove', move, { passive: true })
    addEventListener('pointerover', over, true)
    return () => {
      removeEventListener('pointermove', move)
      removeEventListener('pointerover', over, true)
      onActiveChange(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rm])

  if (!enabled) return null

  return (
    <>
      <motion.div
        aria-hidden="true"
        className="fixed rounded-full pointer-events-none"
        style={{
          width: 7,
          height: 7,
          background: 'var(--champ2)',
          x,
          y,
          top: -3.5,
          left: -3.5,
          zIndex: 130,
        }}
      />
      <motion.div
        aria-hidden="true"
        className="fixed rounded-full pointer-events-none"
        animate={{ scale: hot ? 2 : 1, opacity: hot ? 0.9 : 0.55 }}
        transition={{ duration: 0.25 }}
        style={{
          width: 30,
          height: 30,
          border: '1.5px solid var(--champ2)',
          x: rx,
          y: ry,
          top: -15,
          left: -15,
          zIndex: 130,
        }}
      />
    </>
  )
}
