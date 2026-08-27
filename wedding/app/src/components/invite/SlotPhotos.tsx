import { motion, useReducedMotion } from 'framer-motion'
import pizza from '@/assets/photos/pizza.webp'
import party from '@/assets/photos/party.webp'
import times from '@/assets/photos/times.webp'
import rooftop from '@/assets/photos/rooftop.webp'

/**
 * Snapshots scattered around the machine on the opening screen.
 *
 * Messy and symmetric at once: the four sit on mirrored corners, so the
 * composition balances, but every one is tilted a different amount and hung at
 * a slightly different depth, so none of it looks aligned. The tilts are
 * mirrored in sign rather than equal, which is what keeps it from reading as a
 * grid.
 *
 * They sit behind the machine and take no pointer events, because the whole
 * screen is the lever.
 */
/* The title owns the top of the screen and the machine owns the middle, so the
   snapshots hang in the four corners below the headline. */
const SHOTS = [
  { src: pizza, pos: { top: '30%', insetInlineStart: '1.5%' }, tilt: -7, delay: 0.15 },
  { src: rooftop, pos: { top: '28%', insetInlineEnd: '1.5%' }, tilt: 6, delay: 0.3 },
  { src: party, pos: { bottom: '7%', insetInlineStart: '2.5%' }, tilt: 5, delay: 0.45 },
  { src: times, pos: { bottom: '10%', insetInlineEnd: '2.5%' }, tilt: -8, delay: 0.6 },
]

export default function SlotPhotos() {
  const rm = useReducedMotion()
  return (
    <div className="absolute inset-0 pointer-events-none" aria-hidden="true" style={{ zIndex: 0 }}>
      {SHOTS.map((s, i) => (
        <motion.div
          key={i}
          className="absolute"
          style={{
            ...s.pos,
            width: 'clamp(88px, 24vw, 200px)',
            padding: '6px 6px 15px',
            background: 'var(--bg2)',
            /* the paper edge and a shadow the colour of the page, not black */
            boxShadow: '0 8px 20px rgba(140,110,50,.22), 0 1px 0 rgba(255,255,255,.7) inset',
            rotate: s.tilt,
          }}
          initial={rm ? false : { opacity: 0, scale: 0.86, y: 10 }}
          animate={rm ? undefined : { opacity: 1, scale: 1, y: 0 }}
          transition={{ type: 'spring', bounce: 0.32, duration: 0.75, delay: s.delay }}
        >
          <img
            src={s.src}
            alt=""
            width={300}
            height={300}
            style={{ display: 'block', width: '100%', height: 'auto', aspectRatio: '1' }}
          />
        </motion.div>
      ))}
    </div>
  )
}
