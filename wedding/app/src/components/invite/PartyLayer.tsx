import { useEffect, useRef, useState } from 'react'
import { useReducedMotion } from 'framer-motion'
import { ConfettiEngine } from '@/lib/confetti'
import { SONG_URL, SONG_LABEL } from '@/lib/wedding'
import chorusUrl from '@/assets/chorus.m4a'

/* light reflections the disco ball scatters around the room — pastel set */
const SPOTS = [
  { '--x0': '6vw', '--y0': '14vh', '--x1': '30vw', '--y1': '38vh', '--s': '38px', '--c': 'rgba(255,157,190,.7)', '--d': '7.5s', '--dl': '0s' },
  { '--x0': '78vw', '--y0': '10vh', '--x1': '58vw', '--y1': '34vh', '--s': '30px', '--c': 'rgba(196,118,143,.6)', '--d': '9s', '--dl': '-2s' },
  { '--x0': '42vw', '--y0': '6vh', '--x1': '20vw', '--y1': '26vh', '--s': '24px', '--c': 'rgba(255,120,170,.6)', '--d': '6.5s', '--dl': '-4s' },
  { '--x0': '88vw', '--y0': '42vh', '--x1': '66vw', '--y1': '66vh', '--s': '34px', '--c': 'rgba(169,143,214,.65)', '--d': '10s', '--dl': '-1s' },
  { '--x0': '12vw', '--y0': '58vh', '--x1': '34vw', '--y1': '82vh', '--s': '28px', '--c': 'rgba(147,168,239,.6)', '--d': '8s', '--dl': '-5s' },
  { '--x0': '54vw', '--y0': '74vh', '--x1': '76vw', '--y1': '52vh', '--s': '40px', '--c': 'rgba(159,227,207,.6)', '--d': '11s', '--dl': '-3s' },
  { '--x0': '28vw', '--y0': '88vh', '--x1': '8vw', '--y1': '64vh', '--s': '26px', '--c': 'rgba(255,157,190,.55)', '--d': '7s', '--dl': '-6s' },
  { '--x0': '68vw', '--y0': '90vh', '--x1': '88vw', '--y1': '72vh', '--s': '32px', '--c': 'rgba(169,143,214,.6)', '--d': '9.5s', '--dl': '-2.5s' },
  { '--x0': '48vw', '--y0': '30vh', '--x1': '60vw', '--y1': '12vh', '--s': '22px', '--c': 'rgba(255,199,150,.6)', '--d': '6s', '--dl': '-1.5s' },
  { '--x0': '18vw', '--y0': '36vh', '--x1': '4vw', '--y1': '18vh', '--s': '30px', '--c': 'rgba(147,168,239,.55)', '--d': '8.5s', '--dl': '-4.5s' },
] as const

/* SVG icons instead of emoji — per the ui-ux-pro-max pre-delivery checklist */
const IconDisco = (
  <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden="true">
    <path d="M12 2v3" />
    <circle cx="12" cy="13" r="8" />
    <path d="M12 5v16M4 13h16M6.6 7.8c3 2.1 7.8 2.1 10.8 0M6.6 18.2c3-2.1 7.8-2.1 10.8 0" />
  </svg>
)
const IconMusic = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M9 18V5l12-2v13" />
    <circle cx="6" cy="18" r="3" />
    <circle cx="18" cy="16" r="3" />
  </svg>
)

export default function PartyLayer({
  party,
  onToggle,
  burstSignal,
  showButton,
}: {
  party: boolean
  onToggle: () => void
  burstSignal: number
  showButton: boolean
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<ConfettiEngine | null>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const [muted, setMuted] = useState(false)
  const rm = useReducedMotion()

  // play()/pause() must run inside the click gesture for autoplay policies
  const handleToggle = () => {
    const audio = audioRef.current
    if (audio) {
      if (!party) {
        audio.currentTime = 0
        audio.play().catch(() => {})
      } else {
        audio.pause()
        audio.currentTime = 0
      }
    }
    onToggle()
  }

  const toggleMute = () => {
    const audio = audioRef.current
    if (!audio) return
    audio.muted = !audio.muted
    setMuted(audio.muted)
  }

  useEffect(() => {
    if (!canvasRef.current) return
    engineRef.current = new ConfettiEngine(canvasRef.current)
    const onResize = () => engineRef.current?.resize()
    addEventListener('resize', onResize)
    return () => removeEventListener('resize', onResize)
  }, [])

  useEffect(() => {
    if (burstSignal > 0 && !rm) {
      engineRef.current?.burst(innerWidth / 2, innerHeight * 0.32, 130)
    }
  }, [burstSignal, rm])

  useEffect(() => {
    if (!party || rm) return
    engineRef.current?.burst(innerWidth / 2, innerHeight * 0.35, 90, true)
    const t = setInterval(() => {
      if (!document.hidden) engineRef.current?.rainTick(true)
    }, 320)
    return () => clearInterval(t)
  }, [party, rm])

  return (
    <>
      <audio ref={audioRef} src={chorusUrl} loop preload="auto" />
      <div className="beams" aria-hidden="true" />
      <div className="spots" aria-hidden="true">
        {SPOTS.map((s, i) => (
          <span key={i} style={s as React.CSSProperties} />
        ))}
      </div>
      <div className="ball" aria-hidden="true">
        <div className="ball__cord" />
        <div className="ball__wrap">
          <div className="ball__glow" />
          <div className="ball__sphere" />
        </div>
      </div>
      <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none" style={{ zIndex: 60 }} aria-hidden="true" />

      <a
        href={SONG_URL}
        target="_blank"
        rel="noopener"
        data-cursor
        className="fixed flex items-center chrome-glass press"
        style={{
          bottom: 76,
          insetInlineEnd: 18,
          zIndex: 70,
          gap: 8,
          padding: '9px 16px',
          borderRadius: 999,
          background: 'rgba(255,255,255,.85)',
          backdropFilter: 'blur(8px)',
          border: '1px solid var(--p2)',
          color: 'var(--ivory)',
          fontFamily: 'var(--font-body)',
          fontWeight: 600,
          fontSize: '0.85rem',
          textDecoration: 'none',
          boxShadow: '0 0 20px rgba(169,143,214,.45)',
          opacity: party && showButton ? 1 : 0,
          pointerEvents: party && showButton ? 'auto' : 'none',
          transform: party && showButton ? 'translateY(0)' : 'translateY(8px)',
          transition: 'opacity .5s ease, transform .5s ease',
        }}
      >
        {IconMusic}
        <span>{SONG_LABEL}</span>
      </a>

      <button
        type="button"
        onClick={toggleMute}
        aria-label={muted ? 'ביטול השתקה' : 'השתקת המוזיקה'}
        data-cursor
        className="fixed flex items-center justify-center chrome-glass press"
        style={{
          bottom: 134,
          insetInlineEnd: 18,
          zIndex: 70,
          width: 44,
          height: 44,
          borderRadius: '50%',
          background: 'rgba(255,255,255,.85)',
          backdropFilter: 'blur(8px)',
          border: '1px solid var(--line)',
          color: muted ? 'var(--muted)' : 'var(--champ)',
          cursor: 'pointer',
          opacity: party && showButton ? 1 : 0,
          pointerEvents: party && showButton ? 'auto' : 'none',
          transition: 'opacity .5s ease, color .3s ease',
        }}
      >
        {muted ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M11 5 6 9H3v6h3l5 4V5zM22 9l-6 6M16 9l6 6" />
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M11 5 6 9H3v6h3l5 4V5zM15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13" />
          </svg>
        )}
      </button>

      <button
        type="button"
        onClick={handleToggle}
        aria-pressed={party}
        data-cursor
        className="fixed flex items-center chrome-glass press"
        style={{
          bottom: 18,
          insetInlineEnd: 18,
          zIndex: 70,
          gap: 10,
          padding: '12px 20px',
          borderRadius: 999,
          background: 'rgba(255,255,255,.85)',
          backdropFilter: 'blur(8px)',
          border: party ? '1px solid var(--p1)' : '1px solid var(--line)',
          color: 'var(--ivory)',
          fontFamily: 'var(--font-body)',
          fontWeight: 700,
          fontSize: '0.96rem',
          cursor: 'pointer',
          boxShadow: party ? '0 0 26px rgba(255,157,190,.6)' : '0 10px 26px rgba(163,80,109,.16)',
          opacity: showButton ? 1 : 0,
          pointerEvents: showButton ? 'auto' : 'none',
          transition: 'opacity .5s ease, border-color .4s ease, box-shadow .4s ease',
        }}
      >
        {IconDisco}
        <span>{party ? 'המסיבה רצה!' : 'מצב מסיבה'}</span>
      </button>
    </>
  )
}
