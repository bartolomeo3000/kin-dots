/* The kin-sacrifice model in JavaScript, run in a Web Worker by the presentation and the playground.

A port of the NumPy model (evolife/KinSacrifice/SacrificeWorld.py) with the fertility landscape of
kin_dots/core/landscape.py. Two worlds run side by side from the same seed: gift to the neighbour met,
and the control, gift to a random dot anywhere. The page creates the worker from KinModel.workerSource
and talks to it with messages:
  {type: "init", seed, params, fert}   (re)start; params override DEFAULTS, fert = {contrast, scale, memory} or null
  {type: "run", on}  {type: "speed", v}  {type: "color", v}  {type: "reset"}  (reset = new seed, same settings)
It posts one frame of both worlds about 30 times a second.
*/
function workerMain() {
  const DEFAULTS = { W: 300, g: 75, cap: 10, growth: .05, inflow: .1, init: .5, season: 600, harsh: .9, n0: 2500, e0: 10, speed: .1, moveCost: .05,
    metab: .1, maxAge: 300, bite: .5, birth: 20, birthCost: 1, spread: 2, radius: 4, eff: .9, a0: -9, b0: 0, fs0: 1, noise: .5, mut: .15, tagMut: .02 };
  let P = Object.assign({}, DEFAULTS);
  const CAP = 24000;
  const OFF = []; for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) if (dx || dy) { const n = Math.hypot(dx, dy); OFF.push([dx, dy, dx / n, dy / n]); }
  function mulberry(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  class World {
    constructor(seed, stranger, fert) {
      this.r = mulberry(seed); this.spare = null; this.stranger = stranger;
      const F64 = n => new Float64Array(n), F32 = n => new Float32Array(n);
      this.x = F64(CAP); this.y = F64(CAP); this.e = F64(CAP); this.age = new Int32Array(CAP);
      this.a = F32(CAP); this.b = F32(CAP); this.fs = F32(CAP); this.tr = F32(CAP); this.tg = F32(CAP); this.tb = F32(CAP);
      this.n = P.n0;
      for (let i = 0; i < this.n; i++) {
        this.x[i] = this.r() * P.W; this.y[i] = this.r() * P.W; this.e[i] = P.e0; this.age[i] = 0;
        this.a[i] = P.a0 + this.nrm() * P.noise; this.b[i] = P.b0 + this.nrm() * P.noise; this.fs[i] = P.fs0 + this.nrm() * P.noise;
        this.tr[i] = this.r(); this.tg[i] = this.r(); this.tb[i] = this.r();
      }
      this.grass = new Float32Array(P.g * P.g).fill(P.cap * P.init);
      // fertility map (as kin_dots/core/landscape.py): smoothed noise on the torus, exp(contrast * z),
      // mean 1. Its own generator, seeded from the seed only, so both worlds get the same maps.
      this.fertOpt = fert || null; this.fert = null; this.fertGen = 0;
      if (fert) { this.mr = mulberry((seed * 2654435761 + 97) >>> 0); this.mspare = null; this.fz = this.smooth(); this.fertFromZ(); }
      this.nb = Math.max(3, Math.floor(P.W / P.radius)); this.bs = P.W / this.nb;
      const nb2 = this.nb * this.nb;
      this.bc = new Int32Array(nb2); this.bstart = new Int32Array(nb2 + 1); this.bfill = new Int32Array(nb2);
      this.order = new Int32Array(CAP); this.bid = new Int32Array(CAP);
      this.cell = new Int32Array(CAP); this.kc = new Float32Array(P.g * P.g); this.share = new Float32Array(P.g * P.g);
      this.dead = new Uint8Array(CAP); this.donors = new Int32Array(CAP);
      this.gifts = new Float32Array(4 * 80); this.giftN = 0;
      this.step = 0; this.ringS = new Int32Array(P.season); this.ringD = new Int32Array(P.season); this.sumS = 0; this.sumD = 0; this.yearly = [];
      this.c9 = new Int32Array(9); this.b9 = new Int32Array(9);
    }
    nrm() { if (this.spare !== null) { const s = this.spare; this.spare = null; return s; } let u = 0, v = 0; while (!u) u = this.r(); v = this.r();
      const m = Math.sqrt(-2 * Math.log(u)); this.spare = m * Math.sin(2 * Math.PI * v); return m * Math.cos(2 * Math.PI * v); }
    mn() { if (this.mspare !== null) { const s = this.mspare; this.mspare = null; return s; } let u = 0; while (!u) u = this.mr(); const v = this.mr();
      const m = Math.sqrt(-2 * Math.log(u)); this.mspare = m * Math.sin(2 * Math.PI * v); return m * Math.cos(2 * Math.PI * v); }
    smooth() {
      const g = P.g, s = this.fertOpt.scale / (P.W / g), R = Math.ceil(3 * s), ker = [];
      for (let k = -R; k <= R; k++) ker.push(Math.exp(-k * k / (2 * s * s)));
      const a = new Float32Array(g * g), b = new Float32Array(g * g);
      for (let i = 0; i < g * g; i++) a[i] = this.mn();
      for (let x = 0; x < g; x++) for (let y = 0; y < g; y++) { let t = 0; for (let k = -R; k <= R; k++) t += ker[k + R] * a[(((x + k) % g + g) % g) * g + y]; b[x * g + y] = t; }
      for (let x = 0; x < g; x++) for (let y = 0; y < g; y++) { let t = 0; for (let k = -R; k <= R; k++) t += ker[k + R] * b[x * g + (((y + k) % g + g) % g)]; a[x * g + y] = t; }
      return this.standardise(a);
    }
    standardise(z) { let m = 0, v = 0; for (const q of z) m += q; m /= z.length; for (const q of z) v += (q - m) * (q - m); const sd = Math.sqrt(v / z.length) || 1;
      for (let i = 0; i < z.length; i++) z[i] = (z[i] - m) / sd; return z; }
    fertFromZ() { const c = this.fertOpt.contrast, f = new Float32Array(this.fz.length); let m = 0;
      for (let i = 0; i < f.length; i++) { f[i] = Math.exp(c * this.fz[i]); m += f[i]; } m /= f.length; for (let i = 0; i < f.length; i++) f[i] /= m;
      this.fert = f; this.fertGen++; }
    copy(k, i) { this.x[k] = this.x[i]; this.y[k] = this.y[i]; this.e[k] = this.e[i]; this.age[k] = this.age[i]; this.a[k] = this.a[i]; this.b[k] = this.b[i];
      this.fs[k] = this.fs[i]; this.tr[k] = this.tr[i]; this.tg[k] = this.tg[i]; this.tb[k] = this.tb[i]; }
    buckets() {
      const { nb, bs, bc, bstart, bfill, order, bid } = this, n = this.n; bc.fill(0);
      for (let i = 0; i < n; i++) { let bx = (this.x[i] / bs) | 0, by = (this.y[i] / bs) | 0; if (bx >= nb) bx = nb - 1; if (by >= nb) by = nb - 1; const b = bx * nb + by; bid[i] = b; bc[b]++; }
      bstart[0] = 0; for (let b = 0; b < nb * nb; b++) { bstart[b + 1] = bstart[b] + bc[b]; bfill[b] = bstart[b]; }
      for (let i = 0; i < n; i++) order[bfill[bid[i]]++] = i;
    }
    partner(i) {
      const { nb, bc, bstart, order, c9, b9 } = this; const b = this.bid[i], bx = (b / nb) | 0, by = b % nb; let tot = 0, k = 0;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) { const bb = ((bx + dx + nb) % nb) * nb + (by + dy + nb) % nb; b9[k] = bb; tot += bc[bb]; c9[k++] = tot; }
      if (tot <= 1) return -1;
      const W = P.W, r2 = P.radius * P.radius;
      for (let t = 0; t < 8; t++) {
        const rr = (this.r() * tot) | 0; let q = 0; while (c9[q] <= rr) q++;
        const before = q ? c9[q - 1] : 0, j = order[bstart[b9[q]] + rr - before];
        if (j === i) continue;
        let dx = Math.abs(this.x[i] - this.x[j]); dx = Math.min(dx, W - dx); let dy = Math.abs(this.y[i] - this.y[j]); dy = Math.min(dy, W - dy);
        if (dx * dx + dy * dy <= r2) return j;
      }
      return -1;
    }
    tick() {
      const W = P.W, g = P.g, cs = W / g; let n = this.n, sac = 0, deaths = 0; this.giftN = 0;
      if (n > 0) {
        // sacrifice
        let nd = 0;
        for (let i = 0; i < n; i++) if (this.e[i] > 0) { const z = this.a[i] + this.b[i] * Math.min(this.e[i] / P.birth, 2); if (this.r() < 1 / (1 + Math.exp(-z))) this.donors[nd++] = i; }
        if (nd) {
          this.buckets();
          for (let k = nd - 1; k > 0; k--) { const j = (this.r() * (k + 1)) | 0; const t = this.donors[k]; this.donors[k] = this.donors[j]; this.donors[j] = t; }
          for (let k = 0; k < nd; k++) {
            const i = this.donors[k]; if (this.dead[i]) continue;
            const p = this.partner(i); if (p < 0) continue;
            const j = this.stranger ? (this.r() * n) | 0 : p; if (j === i || this.dead[j]) continue;
            this.e[j] += P.eff * this.e[i]; this.e[i] = 0; this.dead[i] = 1; sac++;
            if (this.giftN < 80) { const q = 4 * this.giftN++; this.gifts[q] = this.x[i]; this.gifts[q + 1] = this.y[i]; this.gifts[q + 2] = this.x[j]; this.gifts[q + 3] = this.y[j]; }
          }
          let w = 0; for (let i = 0; i < n; i++) { if (this.dead[i]) { this.dead[i] = 0; continue; } if (w !== i) this.copy(w, i); w++; } n = w;
        }
        // move
        const sp = P.speed, mc = P.moveCost * P.speed, grass = this.grass;
        for (let i = 0; i < n; i++) {
          const cx = ((this.x[i] / cs) | 0) % g, cy = ((this.y[i] / cs) | 0) % g, own = grass[cx * g + cy]; let gx = 0, gy = 0;
          for (let o = 0; o < 8; o++) { const d = OFF[o], v = grass[((cx + d[0] + g) % g) * g + (cy + d[1] + g) % g] - own; gx += v * d[2]; gy += v * d[3]; }
          let dx = this.fs[i] * gx + this.nrm(), dy = this.fs[i] * gy + this.nrm(); const nn = Math.hypot(dx, dy);
          if (nn > 0) { let x = this.x[i] + dx / nn * sp, y = this.y[i] + dy / nn * sp; if (x < 0) x += W; else if (x >= W) x -= W; if (y < 0) y += W; else if (y >= W) y -= W; this.x[i] = x; this.y[i] = y; }
          this.e[i] -= mc;
        }
        // eat
        const kc = this.kc, share = this.share; kc.fill(0);
        for (let i = 0; i < n; i++) { const c = (((this.x[i] / cs) | 0) % g) * g + ((this.y[i] / cs) | 0) % g; this.cell[i] = c; kc[c]++; }
        for (let c = 0; c < g * g; c++) { const k = kc[c]; if (k > 0) { const eaten = grass[c] * (1 - Math.pow(1 - P.bite, k)); share[c] = eaten / k; grass[c] -= eaten; } }
        for (let i = 0; i < n; i++) this.e[i] += share[this.cell[i]];
        // metabolism and death
        let w = 0;
        for (let i = 0; i < n; i++) { this.e[i] -= P.metab; if (this.e[i] > 0 && this.age[i] < P.maxAge) { if (w !== i) this.copy(w, i); w++; } else deaths++; }
        n = w;
        // reproduction
        const n0 = n;
        for (let i = 0; i < n0 && n < CAP; i++) if (this.e[i] >= P.birth) {
          const rest = this.e[i] - P.birthCost; this.e[i] = rest / 2; const k = n++;
          let x = this.x[i] + this.nrm() * P.spread, y = this.y[i] + this.nrm() * P.spread; x = ((x % W) + W) % W; y = ((y % W) + W) % W;
          this.x[k] = x; this.y[k] = y; this.e[k] = rest / 2; this.age[k] = 0;
          const cl = v => v < -10 ? -10 : v > 10 ? 10 : v, c01 = v => v < 0 ? 0 : v > 1 ? 1 : v;
          this.a[k] = cl(this.a[i] + this.nrm() * P.mut); this.b[k] = cl(this.b[i] + this.nrm() * P.mut); this.fs[k] = cl(this.fs[i] + this.nrm() * P.mut);
          this.tr[k] = c01(this.tr[i] + this.nrm() * P.tagMut); this.tg[k] = c01(this.tg[i] + this.nrm() * P.tagMut); this.tb[k] = c01(this.tb[i] + this.nrm() * P.tagMut);
        }
        for (let i = 0; i < n; i++) this.age[i]++;
      }
      this.n = n;
      // a new (or drifted) fertility map at the start of each year
      if (this.fert && this.fertOpt.memory < 1 && this.step > 0 && this.step % P.season === 0) {
        const m = this.fertOpt.memory, fresh = this.smooth(), q = Math.sqrt(Math.max(0, 1 - m * m));
        for (let i = 0; i < this.fz.length; i++) this.fz[i] = m * this.fz[i] + q * fresh[i];
        this.standardise(this.fz); this.fertFromZ();
      }
      // regrow
      const sf = Math.max(0, 1 + P.harsh * Math.sin(2 * Math.PI * this.step / P.season)), gr = this.grass, fe = this.fert;
      for (let c = 0; c < g * g; c++) { let f = gr[c]; f += sf * (fe ? fe[c] : 1) * (P.growth * f * (1 - f / P.cap) + P.inflow); gr[c] = f < 0 ? 0 : f > P.cap ? P.cap : f; }
      // statistics over the last year
      const pos = this.step % P.season; this.sumS += sac - this.ringS[pos]; this.sumD += sac + deaths - this.ringD[pos]; this.ringS[pos] = sac; this.ringD[pos] = sac + deaths;
      this.step++;
      if (this.step % P.season === 0) this.yearly.push(this.sumD ? this.sumS / this.sumD : 0);
    }
    frame(mode) {
      const n = this.n, pos = new Float32Array(2 * n), col = new Uint8Array(3 * n);
      for (let i = 0; i < n; i++) {
        pos[2 * i] = this.x[i]; pos[2 * i + 1] = this.y[i];
        if (mode === 1) { col[3 * i] = 40 + 215 * this.tr[i]; col[3 * i + 1] = 40 + 215 * this.tg[i]; col[3 * i + 2] = 40 + 215 * this.tb[i]; }
        else {
          const z = this.a[i] + this.b[i] * Math.min(this.e[i] / P.birth, 2), p = 1 / (1 + Math.exp(-z));
          let t = (Math.log10(Math.max(p, 1e-6)) + 4) / 2.5; t = t < 0 ? 0 : t > 1 ? 1 : t;
          // blue -> amber -> ember
          let r, gg, b; if (t < .6) { const k = t / .6; r = 91 + (243 - 91) * k; gg = 123 + (181 - 123) * k; b = 224 + (74 - 224) * k; }
          else { const k = (t - .6) / .4; r = 243 + (255 - 243) * k; gg = 181 + (107 - 181) * k; b = 74 + (61 - 74) * k; }
          col[3 * i] = r; col[3 * i + 1] = gg; col[3 * i + 2] = b;
        }
      }
      const gq = new Uint8Array(P.g * P.g); for (let c = 0; c < P.g * P.g; c++) gq[c] = Math.round(this.grass[c] / P.cap * 255);
      let fq = null;
      if (this.fert) { fq = new Uint8Array(P.g * P.g); for (let c = 0; c < P.g * P.g; c++) { const t = (Math.log(this.fert[c]) + 1.5) / 3; fq[c] = t < 0 ? 0 : t > 1 ? 15 : Math.round(t * 15); } }
      return { n, pos, col, grass: gq, fert: fq, fertGen: this.fertGen, gifts: this.gifts.slice(0, 4 * this.giftN), share: this.sumD ? this.sumS / this.sumD : 0, yearly: this.yearly.slice(), step: this.step };
    }
  }
  let worlds = [], running = true, speed = 40, mode = 0, seed = 1, fert = null, started = false;
  function reset() { worlds = [new World(seed, false, fert), new World(seed, true, fert)]; }
  function loop() {
    const t0 = performance.now();
    if (running) { let k = 0; while (k < speed && performance.now() - t0 < 26) { worlds[0].tick(); worlds[1].tick(); k++; } }
    const f = worlds.map(w => w.frame(mode)), bufs = [];
    f.forEach(x => { bufs.push(x.pos.buffer, x.col.buffer, x.grass.buffer); if (x.fert) bufs.push(x.fert.buffer); });
    postMessage({ worlds: f, season: Math.sin(2 * Math.PI * worlds[0].step / P.season) }, bufs);
    setTimeout(loop, Math.max(0, 33 - (performance.now() - t0)));
  }
  onmessage = ev => { const m = ev.data;
    // init: (re)start both worlds, with optional parameter overrides and a fertility map
    if (m.type === "init") { seed = m.seed || 1; fert = m.fert || null; P = Object.assign({}, DEFAULTS, m.params || {}); reset(); if (!started) { started = true; loop(); } }
    else if (m.type === "run") running = m.on; else if (m.type === "speed") speed = m.v; else if (m.type === "color") mode = m.v;
    else if (m.type === "reset") { seed = (seed * 7919 + 13) % 100000 || 1; reset(); } };
}

window.KinModel = { workerSource: `(${workerMain.toString()})()` };
