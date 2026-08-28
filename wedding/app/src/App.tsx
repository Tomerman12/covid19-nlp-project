import { useEffect, useState } from 'react'
import SlotIntro from './components/invite/SlotIntro'
import Hero from './components/invite/Hero'
import Countdown from './components/invite/Countdown'
import Timeline from './components/invite/Timeline'
import Venue from './components/invite/Venue'
import Finale from './components/invite/Finale'
import PartyLayer from './components/invite/PartyLayer'
import Cursor from './components/invite/Cursor'

/* `leaving` is the slide-away. It is a CSS transition rather than a
   frame-driven one: a phone that starves the frame loop would otherwise leave
   the intro parked halfway up the screen, covering the invitation for good. */
type Phase = 'loading' | 'leaving' | 'site'

export default function App() {
  const [phase, setPhase] = useState<Phase>('loading')
  const [party, setParty] = useState(false)
  const [burstSignal, setBurstSignal] = useState(0)
  const [customCursor, setCustomCursor] = useState(false)

  // lock scroll until the intro finishes
  useEffect(() => {
    document.body.classList.toggle('locked', phase !== 'site')
    return () => document.body.classList.remove('locked')
  }, [phase])

  const enterSite = () => {
    setPhase('leaving')
    // plain timers, so the hand-off completes whether or not frames are drawn
    window.setTimeout(() => setPhase('site'), 620)
    window.setTimeout(() => setBurstSignal((n) => n + 1), 550)
  }

  return (
    <div dir="rtl" lang="he" className={(party ? 'party ' : '') + (customCursor ? 'cursor-none-root' : '')}>
      {phase !== 'site' && <SlotIntro leaving={phase === 'leaving'} onDone={enterSite} />}

      {phase !== 'loading' && (
        <main>
          <Hero />
          <Countdown />
          <Timeline />
          <Venue />
          <Finale />
        </main>
      )}

      <PartyLayer party={party} onToggle={() => setParty((p) => !p)} burstSignal={burstSignal} showButton={phase === 'site'} />
      <div className="grain" aria-hidden="true" />
      <Cursor onActiveChange={setCustomCursor} />
    </div>
  )
}
