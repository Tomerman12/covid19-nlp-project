type Piece = {
  x: number; y: number; vx: number; vy: number
  rot: number; vr: number; w: number; h: number
  c: string; circle: boolean; ph: number; life: number
}

const CALM = ['#e79ab2', '#a98fd6', '#93a8ef', '#f6cdd9']
const PARTY = ['#ff9dbe', '#a98fd6', '#93a8ef', '#ffd6a5', '#9fe3cf', '#f9b8c8']

export class ConfettiEngine {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private pieces: Piece[] = []
  private running = false
  private dpr: number

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.dpr = Math.min(window.devicePixelRatio || 1, 2)
    this.resize()
  }

  resize() {
    this.canvas.width = innerWidth * this.dpr
    this.canvas.height = innerHeight * this.dpr
    this.canvas.style.width = innerWidth + 'px'
    this.canvas.style.height = innerHeight + 'px'
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0)
  }

  private spawn(x: number, y: number, vx: number, vy: number, party: boolean) {
    if (this.pieces.length > 340) return
    const colors = party ? PARTY : CALM
    this.pieces.push({
      x, y, vx, vy,
      rot: Math.random() * Math.PI * 2, vr: (Math.random() - 0.5) * 0.3,
      w: 5 + Math.random() * 7, h: 8 + Math.random() * 8,
      c: colors[Math.floor(Math.random() * colors.length)],
      circle: Math.random() < 0.25, ph: Math.random() * Math.PI * 2,
      life: 0,
    })
  }

  burst(x: number, y: number, n: number, party = false) {
    for (let i = 0; i < n; i++) {
      const ang = Math.random() * Math.PI * 2
      const sp = 3 + Math.random() * 8
      this.spawn(x, y, Math.cos(ang) * sp, Math.sin(ang) * sp - 4, party)
    }
    this.run()
  }

  rainTick(party = true) {
    for (let i = 0; i < 7; i++) {
      this.spawn(Math.random() * innerWidth, -16, (Math.random() - 0.5) * 1.6, 1.5 + Math.random() * 2.5, party)
    }
    this.run()
  }

  private step = () => {
    const { ctx, pieces } = this
    ctx.clearRect(0, 0, innerWidth, innerHeight)
    for (let i = pieces.length - 1; i >= 0; i--) {
      const p = pieces[i]
      p.life++
      p.vy += 0.13
      p.vx *= 0.992
      p.vy *= 0.992
      p.x += p.vx + Math.sin(p.life * 0.06 + p.ph) * 0.6
      p.y += p.vy
      p.rot += p.vr
      if (p.y > innerHeight + 30 || p.life > 900) {
        pieces.splice(i, 1)
        continue
      }
      ctx.save()
      ctx.translate(p.x, p.y)
      ctx.rotate(p.rot)
      ctx.fillStyle = p.c
      ctx.globalAlpha = p.life > 800 ? (900 - p.life) / 100 : 1
      if (p.circle) {
        ctx.beginPath()
        ctx.arc(0, 0, p.w * 0.5, 0, 7)
        ctx.fill()
      } else {
        ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h * Math.abs(Math.sin(p.life * 0.05 + p.ph)) + 1)
      }
      ctx.restore()
    }
    if (pieces.length) requestAnimationFrame(this.step)
    else {
      this.running = false
      ctx.clearRect(0, 0, innerWidth, innerHeight)
    }
  }

  private run() {
    if (!this.running) {
      this.running = true
      requestAnimationFrame(this.step)
    }
  }
}
