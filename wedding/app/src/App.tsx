import { useEffect, useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import SlotIntro from './components/invite/SlotIntro'
import Hero from './components/invite/Hero'
import Message from './components/invite/Message'
import Countdown from './components/invite/Countdown'
import Timeline from './components/invite/Timeline'
import Venue from './components/invite/Venue'
import Finale from './components/invite/Finale'
import PartyLayer from './components/invite/PartyLayer'
import Cursor from './components/invite/Cursor'
import { ScrollProgress } from './components/vendor/scroll-progress'

type Phase = 'loading' | 'site'

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
    setPhase('site')
    window.setTimeout(() => setBurstSignal((n) => n + 1), 550)
  }

  return (
    <div dir="rtl" lang="he" className={(party ? 'party ' : '') + (customCursor ? 'cursor-none-root' : '')}>
      <AnimatePresence>{phase === 'loading' && <SlotIntro key="slot" onDone={enterSite} />}</AnimatePresence>

      {phase === 'site' && (
        <main>
          <ScrollProgress />
          <Hero />
          <Message />
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
