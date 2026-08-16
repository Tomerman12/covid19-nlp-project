/**
 * ScrollProgress — adapted from Magic UI (magicui.design), MIT license.
 * RTL-aware origin; pastel rainbow fill.
 */
import { motion, useScroll } from 'framer-motion'

export function ScrollProgress() {
  const { scrollYProgress } = useScroll()

  return (
    <motion.div
      aria-hidden="true"
      className="fixed inset-x-0 top-0 pointer-events-none"
      style={{
        height: 2,
        zIndex: 85,
        transformOrigin: 'right center',
        background: 'linear-gradient(to left, #c73a72, #d1742f, #c09427, #3a9a66, #3e8ec4, #8a58bd)',
        scaleX: scrollYProgress,
      }}
    />
  )
}
