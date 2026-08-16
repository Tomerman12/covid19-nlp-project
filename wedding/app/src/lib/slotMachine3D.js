// @ts-nocheck
/* ============================================================
   מכונת המזל — סצנת תלת־ממד אמיתית (three.js)
   גוף פלסטיק עם clearcoat, מסגרת מתכת, גלגלים שהם גלילים
   מסתובבים עם טקסטורה, תאורת סטודיו, השתקפויות וצל רך.
   ============================================================ */
import * as THREE from 'three'

/* מכונת מזל תלת־ממדית — נטענת כשלב הפתיחה של ההזמנה */
export function createSlotMachine3D(canvas, opts) {
  const O = Object.assign({
    reels: ['28', '10', '26'],
    onStop: function () {},
    onFinish: function () {}
  }, opts || {});

  const PINK       = 0xD4809A;
  const PINK_DARK  = 0xC77E96;
  const GOLD       = 0xBF9846;
  const RED        = 0xA81628;

  const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.78;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(26, 1, 0.1, 100);
  camera.position.set(1.35, 2.5, 13.8);
  camera.lookAt(0, 2.1, 0);

  /* ---------- סביבה להשתקפויות (גרדיאנט אולפן) ---------- */
  (function environment() {
    const c = document.createElement('canvas');
    c.width = 512; c.height = 256;
    const g = c.getContext('2d');
    const grad = g.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0.00, '#FFFFFF');
    grad.addColorStop(0.22, '#F6E4EC');
    grad.addColorStop(0.48, '#C9A6BA');
    grad.addColorStop(0.72, '#6E5464');
    grad.addColorStop(1.00, '#2A1E26');
    g.fillStyle = grad; g.fillRect(0, 0, 512, 256);
    // שני "חלונות אור" — נותנים לפלסטיק והמתכת בבואות ממשיות
    g.fillStyle = 'rgba(255,255,255,.95)';
    g.fillRect(40, 20, 150, 70);
    g.fillStyle = 'rgba(255,255,255,.65)';
    g.fillRect(320, 40, 110, 46);
    const tex = new THREE.CanvasTexture(c);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromEquirectangular(tex).texture;
    tex.dispose(); pmrem.dispose();
  })();

  /* ---------- תאורה ---------- */
  scene.add(new THREE.HemisphereLight(0xFFFFFF, 0xC9A3B6, 0.30));

  const key = new THREE.DirectionalLight(0xffffff, 1.45);
  key.position.set(4.5, 7.5, 6.5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.near = 1; key.shadow.camera.far = 24;
  key.shadow.camera.left = -5; key.shadow.camera.right = 5;
  key.shadow.camera.top = 7;   key.shadow.camera.bottom = -2;
  key.shadow.radius = 7; key.shadow.bias = -0.0012;
  scene.add(key);

  const fill = new THREE.DirectionalLight(0xFFE1EC, 0.35);
  fill.position.set(-6, 2.5, 5);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0xFFF3F7, 0.65);
  rim.position.set(-3.5, 5, -6);
  scene.add(rim);

  /* ---------- עוזרים גיאומטריים ---------- */
  function roundedPath(path, w, h, r, cx, cy) {
    const x = -w / 2 + (cx || 0), y = -h / 2 + (cy || 0);
    path.moveTo(x + r, y);
    path.lineTo(x + w - r, y); path.quadraticCurveTo(x + w, y, x + w, y + r);
    path.lineTo(x + w, y + h - r); path.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    path.lineTo(x + r, y + h); path.quadraticCurveTo(x, y + h, x, y + h - r);
    path.lineTo(x, y + r); path.quadraticCurveTo(x, y, x + r, y);
    return path;
  }
  function roundedShape(w, h, r) { return roundedPath(new THREE.Shape(), w, h, r); }

  function extrude(shape, depth, bevel) {
    const g = new THREE.ExtrudeGeometry(shape, {
      depth: depth, curveSegments: 48,
      bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 5
    });
    g.center();
    return g;
  }

  const machine = new THREE.Group();
  scene.add(machine);

  /* ---------- גוף המכונה ---------- */
  const BODY_W = 2.7, BODY_H = 4.05, BODY_D = 1.25;
  const bodyShape = new THREE.Shape();
  (function () {
    const w = BODY_W / 2, straight = 1.05;   // גובה החלק הישר מלמטה
    bodyShape.moveTo(-w, -BODY_H / 2);
    bodyShape.lineTo(-w, straight);
    bodyShape.absarc(0, straight, w, Math.PI, 0, true);
    bodyShape.lineTo(w, -BODY_H / 2);
    bodyShape.closePath();
    // חור אמיתי לחלון הגלגלים
    bodyShape.holes.push(roundedPath(new THREE.Path(), 2.04, 0.56, 0.07, 0, 0.4175));
  })();
  const bodyMat = new THREE.MeshPhysicalMaterial({
    color: PINK, roughness: 0.38, metalness: 0.0,
    clearcoat: 0.7, clearcoatRoughness: 0.16,
    envMapIntensity: 0.35,
    sheen: 0.3, sheenColor: new THREE.Color(0xFFE6EF)
  });
  const body = new THREE.Mesh(extrude(bodyShape, BODY_D, 0.1), bodyMat);
  body.castShadow = true; body.receiveShadow = true;
  body.position.y = 2.05;
  machine.add(body);

  /* ---------- בסיס ---------- */
  const base = new THREE.Mesh(
    extrude(roundedShape(3.1, 0.42, 0.1), 1.5, 0.06),
    new THREE.MeshPhysicalMaterial({ color: PINK_DARK, roughness: 0.5, clearcoat: 0.6, clearcoatRoughness: 0.3 })
  );
  base.castShadow = true; base.receiveShadow = true;
  base.position.set(0, 0.19, 0);
  machine.add(base);

  const FRONT = BODY_D / 2 + 0.1;   // מישור החזית

  /* ---------- מסגרת המסך (מתכת) ---------- */
  const bezelShape = roundedShape(2.32, 0.82, 0.1);
  bezelShape.holes.push(roundedPath(new THREE.Path(), 2.0, 0.52, 0.06));
  const goldMat = new THREE.MeshStandardMaterial({ color: GOLD, roughness: 0.16, metalness: 1.0, envMapIntensity: 1.5 });
  const bezel = new THREE.Mesh(extrude(bezelShape, 0.16, 0.035), goldMat);
  bezel.castShadow = true;
  bezel.position.set(0, 2.28, FRONT - 0.02);
  machine.add(bezel);

  /* ---------- חלל הגלגלים ---------- */
  const cavity = new THREE.Mesh(
    new THREE.BoxGeometry(2.02, 0.54, 0.9),
    new THREE.MeshStandardMaterial({ color: 0x4A2C36, roughness: 0.9, side: THREE.BackSide })
  );
  cavity.position.set(0, 2.28, FRONT - 0.52);
  machine.add(cavity);

  /* ---------- טקסטורת הגלגל ---------- */
  const FACES = 8;
  function reelTexture(symbols, smear) {
    const cw = 260, ch = 480;                    // תא אחד: היקף × רוחב התוף
    const c = document.createElement('canvas');
    c.width = cw * FACES; c.height = ch;
    const g = c.getContext('2d');
    g.fillStyle = '#FBF6F2'; g.fillRect(0, 0, c.width, c.height);
    // גוון עדין לאורך התוף
    const sh = g.createLinearGradient(0, 0, 0, ch);
    sh.addColorStop(0, 'rgba(120,70,86,.10)');
    sh.addColorStop(.5, 'rgba(255,255,255,0)');
    sh.addColorStop(1, 'rgba(120,70,86,.10)');
    g.fillStyle = sh; g.fillRect(0, 0, c.width, ch);

    const passes = smear ? 9 : 1;
    for (let p = 0; p < passes; p++) {
      g.globalAlpha = smear ? 1 / passes : 1;
      const off = smear ? (p - (passes - 1) / 2) * (cw * 0.16) : 0;
      symbols.forEach((sym, i) => {
        g.save();
        g.translate(i * cw + cw / 2 + off, ch / 2);
        g.rotate(-Math.PI / 2);
        g.fillStyle = sym === '♡' ? '#D48CA2' : '#43272F';
        g.font = (sym === '♡' ? '600 200px' : '600 175px') +
                 ' Didot, "Bodoni 72", Georgia, "Times New Roman", serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText(sym, 0, 0);
        g.restore();
      });
    }
    g.globalAlpha = 1;
    const t = new THREE.CanvasTexture(c);
    t.encoding = THREE.sRGBEncoding;
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return t;
  }

  /* ---------- הגלגלים ---------- */
  const R = 0.40, DRUM_W = 0.58;
  const REELS = [];
  O.reels.forEach((target, i) => {
    const stop = 2 + i;
    const symbols = [];
    for (let f = 0; f < FACES; f++) {
      if (f === stop) symbols.push(target);
      else if (f === 0 || f === 4) symbols.push('♡');
      else symbols.push(String(Math.floor(Math.random() * 30) + 1).padStart(2, '0'));
    }
    const sharp  = reelTexture(symbols, false);
    const smear  = reelTexture(symbols, true);
    const mat    = new THREE.MeshStandardMaterial({ map: sharp, roughness: 0.5, metalness: 0.0, envMapIntensity: 0.5 });
    const drum   = new THREE.Mesh(new THREE.CylinderGeometry(R, R, DRUM_W, 96, 1, true), mat);
    drum.castShadow = true; drum.receiveShadow = true;
    const holder = new THREE.Group();             // הקבוצה מניחה את הגליל על הצד
    holder.rotation.z = -Math.PI / 2;
    holder.position.set((i - 1) * 0.65, 2.28, FRONT - 0.46);
    holder.add(drum);
    machine.add(holder);

    // דפנות מתכת לתוף
    [-1, 1].forEach(sgn => {
      const cap = new THREE.Mesh(
        new THREE.CircleGeometry(R * 0.995, 48),
        new THREE.MeshStandardMaterial({ color: 0xD9CCC6, roughness: 0.35, metalness: 0.6, envMapIntensity: 1.2 })
      );
      cap.rotation.y = sgn * Math.PI / 2;
      cap.position.set(holder.position.x + sgn * DRUM_W / 2, holder.position.y, holder.position.z);
      machine.add(cap);
    });

    REELS.push({ drum: drum, mat: mat, sharp: sharp, smear: smear, stop: stop });
  });

  /* ---------- זכוכית מול הגלגלים ---------- */
  const glass = new THREE.Mesh(
    new THREE.PlaneGeometry(2.0, 0.52),
    new THREE.MeshPhysicalMaterial({
      color: 0xffffff, transparent: true, opacity: 0.10,
      roughness: 0.02, metalness: 0, clearcoat: 1, transmission: 0
    })
  );
  glass.position.set(0, 2.28, FRONT + 0.03);
  machine.add(glass);

  /* ---------- נורות ---------- */
  const bulbs = [];
  const bulbMat = () => new THREE.MeshPhysicalMaterial({
    color: 0xFFF3F7, roughness: 0.18, clearcoat: 1,
    emissive: new THREE.Color(0xFFB570), emissiveIntensity: 0
  });
  [[-0.95, 3.22], [-0.48, 3.42], [0, 3.5], [0.48, 3.42], [0.95, 3.22]].forEach(p => {
    const b = new THREE.Mesh(new THREE.SphereGeometry(0.115, 28, 20), bulbMat());
    b.position.set(p[0], p[1], FRONT - 0.06);
    b.castShadow = true;
    machine.add(b);
    bulbs.push(b);
  });

  /* ---------- פס תאורה, כפתורים, חריץ ומגש ---------- */
  const lightbar = new THREE.Mesh(
    extrude(roundedShape(1.95, 0.2, 0.1), 0.1, 0.03),
    new THREE.MeshPhysicalMaterial({ color: 0xE7B7C7, roughness: 0.35, clearcoat: 1 })
  );
  lightbar.position.set(0, 2.95, FRONT - 0.06);
  machine.add(lightbar);
  for (let i = 0; i < 5; i++) {
    const d = new THREE.Mesh(
      new THREE.SphereGeometry(0.045, 18, 14),
      new THREE.MeshPhysicalMaterial({ color: 0xF6D6E0, roughness: 0.2, clearcoat: 1 })
    );
    d.position.set(-0.72 + i * 0.36, 2.95, FRONT + 0.01);
    machine.add(d);
  }

  [-0.42, 0, 0.42].forEach(x => {
    const btn = new THREE.Mesh(
      extrude(roundedShape(0.34, 0.13, 0.06), 0.08, 0.025),
      new THREE.MeshPhysicalMaterial({ color: 0xF0C3D2, roughness: 0.3, clearcoat: 1 })
    );
    btn.position.set(x, 1.58, FRONT - 0.02);
    btn.castShadow = true;
    machine.add(btn);
  });

  const slot = new THREE.Mesh(
    extrude(roundedShape(0.9, 0.09, 0.04), 0.06, 0.02),
    new THREE.MeshStandardMaterial({ color: 0xB07E92, roughness: 0.5 })
  );
  slot.position.set(0, 1.32, FRONT - 0.04);
  machine.add(slot);

  const tray = new THREE.Mesh(
    extrude(roundedShape(2.0, 0.78, 0.24), 0.12, 0.04),
    new THREE.MeshPhysicalMaterial({ color: 0xDFA8BA, roughness: 0.45, clearcoat: 0.7 })
  );
  tray.position.set(0, 0.82, FRONT + 0.005);
  tray.receiveShadow = true;
  machine.add(tray);

  const knob = new THREE.Mesh(
    new THREE.CylinderGeometry(0.2, 0.2, 0.09, 40),
    new THREE.MeshStandardMaterial({ color: GOLD, roughness: 0.22, metalness: 1 })
  );
  knob.rotation.x = Math.PI / 2;
  knob.position.set(0, 0.82, FRONT + 0.075);
  knob.castShadow = true;
  machine.add(knob);

  /* ---------- כותרת מוטבעת ---------- */
  (function marquee() {
    const c = document.createElement('canvas');
    c.width = 1024; c.height = 512;
    const g = c.getContext('2d');
    g.clearRect(0, 0, 1024, 512);
    g.translate(512, 470);
    g.font = '600 60px Didot, "Bodoni 72", Georgia, serif';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    const text = 'SAVE THE DATE', radius = 360, spread = 1.05;
    const chars = text.split('');
    const step = spread / (chars.length - 1);
    chars.forEach((ch, i) => {
      const a = -spread / 2 + i * step;
      g.save();
      g.rotate(a);
      g.translate(0, -radius);
      g.fillStyle = 'rgba(255,255,255,.9)';  g.fillText(ch, 0, -2.5);
      g.fillStyle = 'rgba(120,60,84,.55)';   g.fillText(ch, 0, 2.5);
      g.fillStyle = '#DCA6B9';               g.fillText(ch, 0, 0);
      g.restore();
    });
    const tex = new THREE.CanvasTexture(c);
    tex.encoding = THREE.sRGBEncoding;
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(2.75, 1.38),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false })
    );
    m.position.set(0, 3.66, FRONT + 0.005);
    machine.add(m);
  })();

  /* ---------- ידית ---------- */
  const lever = new THREE.Group();
  lever.position.set(BODY_W / 2 + 0.02, 2.3, 0.35);
  machine.add(lever);

  const mount = new THREE.Mesh(
    new THREE.CylinderGeometry(0.17, 0.17, 0.2, 32),
    new THREE.MeshStandardMaterial({ color: GOLD, roughness: 0.2, metalness: 1, envMapIntensity: 1.5 })
  );
  mount.rotation.x = Math.PI / 2;
  mount.position.set(BODY_W / 2 + 0.02, 2.3, 0.35);
  mount.castShadow = true;
  machine.add(mount);

  const rod = new THREE.Mesh(
    new THREE.CylinderGeometry(0.062, 0.07, 1.0, 24),
    new THREE.MeshStandardMaterial({ color: 0xD8B08C, roughness: 0.18, metalness: 1 })
  );
  rod.position.y = 0.5;
  rod.castShadow = true;
  lever.add(rod);

  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(0.21, 40, 28),
    new THREE.MeshPhysicalMaterial({ color: RED, roughness: 0.22, clearcoat: 0.65, clearcoatRoughness: 0.08, envMapIntensity: 0.7 })
  );
  ball.position.y = 1.08;
  ball.castShadow = true;
  lever.add(ball);

  const LEVER_REST = -0.36, LEVER_PULL = 0.95;
  lever.rotation.z = LEVER_REST;

  /* ---------- רצפה לצל ---------- */
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(24, 24),
    new THREE.ShadowMaterial({ opacity: 0.26 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.02;
  floor.receiveShadow = true;
  scene.add(floor);

  /* ---------- אנימציה ---------- */
  function easeOutCubic(t){ return 1 - Math.pow(1 - t, 3); }
  function easeInQuad(t){ return t * t; }
  function easeOutBack(t){ const c1 = 1.9, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); }

  let spinning = false, done = false;
  const SEG = (Math.PI * 2) / FACES;
  const clock = new THREE.Clock();
  let litAt = 0;

  function spin() {
    if (spinning || done) return;
    spinning = true;
    const t0 = performance.now();

    // משיכת הידית: למטה מהר, חזרה עם קפיץ
    const leverAnim = { start: t0, down: 190, up: 620 };

    const plans = REELS.map((R_, i) => {
      const dur   = 2500 + i * 620;
      const turns = 5 + i;
      // מרכז הפאה מול הצופה בסוף הסיבוב
      const end   = turns * Math.PI * 2 - (R_.stop + 0.5) * SEG;
      return { dur: dur, end: end, back: -0.16, stopped: false };
    });

    function frame(now) {
      const t = now - t0;

      // ידית
      if (t < leverAnim.down) {
        lever.rotation.z = LEVER_REST + (LEVER_PULL - LEVER_REST) * easeInQuad(t / leverAnim.down);
      } else if (t < leverAnim.down + leverAnim.up) {
        const k = (t - leverAnim.down) / leverAnim.up;
        lever.rotation.z = LEVER_PULL + (LEVER_REST - LEVER_PULL) * easeOutBack(k);
      } else {
        lever.rotation.z = LEVER_REST;
      }

      let allDone = true;
      REELS.forEach((R_, i) => {
        const p = plans[i];
        const k = Math.min(t / p.dur, 1);
        if (k < 1) allDone = false;

        let a;
        if (k < 0.06) {                       // נסיגה קטנה לאחור
          a = p.back * easeOutCubic(k / 0.06);
        } else if (k < 0.34) {                // האצה
          const u = (k - 0.06) / 0.28;
          a = p.back + (p.end * 0.26 - p.back) * easeInQuad(u);
        } else if (k < 0.72) {                // מהירות קבועה
          const u = (k - 0.34) / 0.38;
          a = p.end * 0.26 + (p.end * 0.72 - p.end * 0.26) * u;
        } else if (k < 0.94) {                // האטה + חריגה מעבר לסמל
          const u = (k - 0.72) / 0.22;
          a = p.end * 0.72 + ((p.end - SEG * 0.34) - p.end * 0.72) * easeOutCubic(u);
        } else {                              // נעילה חזרה על הסמל
          const u = (k - 0.94) / 0.06;
          a = (p.end - SEG * 0.34) + (SEG * 0.34) * easeOutBack(u);
        }
        R_.drum.rotation.y = a;

        // החלפת טקסטורה למרוחה כשהתוף מהיר
        const fast = k > 0.10 && k < 0.86;
        const want = fast ? R_.smear : R_.sharp;
        if (R_.mat.map !== want) { R_.mat.map = want; R_.mat.needsUpdate = true; }

        if (k >= 1 && !p.stopped) { p.stopped = true; O.onStop(i); }
      });

      renderer.render(scene, camera);

      if (!allDone) {
        requestAnimationFrame(frame);
      } else {
        spinning = false; done = true; litAt = performance.now();
        O.onFinish();
        requestAnimationFrame(idle);
      }
    }
    requestAnimationFrame(frame);
  }

  function idle(now) {
    // הבהוב הנורות אחרי הזכייה
    if (litAt) {
      const t = (now - litAt) / 1000;
      bulbs.forEach((b, i) => {
        b.material.emissiveIntensity = 0.55 + 0.45 * Math.sin(t * 5 - i * 0.7);
      });
    }
    renderer.render(scene, camera);
    if (litAt) requestAnimationFrame(idle);
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  }
  window.addEventListener('resize', resize);
  resize();
  renderer.render(scene, camera);

  return { spin: spin, resize: resize };
};
