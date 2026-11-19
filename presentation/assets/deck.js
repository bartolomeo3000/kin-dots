/* Dots That Give Their Lives: slide engine, live model and visuals. */
(() => {
"use strict";
const D = window.DECK_DATA || {};
// folder of this script, so images load wherever the page lives (site root, presentation/, a local copy)
const BASE = (document.currentScript && document.currentScript.src) ? document.currentScript.src.replace(/[^/]*$/, "") : "presentation/assets/";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const REDUCED = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
const COL = { ink: "#0b1112", ink2: "#111a1b", rule: "#253434", paper: "#ece5d6", muted: "#8e9b95",
  kin: "#f3b54a", str: "#55c6da", ember: "#ff6b3d", moss: "#3d7a4c", grey: "#7c8681", old: "#9b7fd1" };
const ease = t => t < 0 ? 0 : t > 1 ? 1 : 1 - Math.pow(1 - t, 3);
const easeIO = t => t < 0 ? 0 : t > 1 ? 1 : t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const lerp = (a, b, t) => a + (b - a) * t;

/* ---------------- stage scaling ---------------- */
const stage = $("#stage"), viewport = $("#viewport");
let SCALE = 1;
function fit() {
  const W = viewport.clientWidth, H = viewport.clientHeight;
  SCALE = Math.min(W / 1920, H / 1080);
  stage.style.left = (W - 1920 * SCALE) / 2 + "px";
  stage.style.top = (H - 1080 * SCALE) / 2 + "px";
  stage.style.transform = `scale(${SCALE})`;
}
window.addEventListener("resize", () => { fit(); if (cur >= 0) resizeActive(); });
fit();

/* canvas sized in stage pixels, drawn at screen resolution */
function setup(c, maxRatio = 2) {
  const w = c.offsetWidth, h = c.offsetHeight;
  const r = clamp((window.devicePixelRatio || 1) * SCALE, 1, maxRatio);
  c.width = Math.max(1, Math.round(w * r)); c.height = Math.max(1, Math.round(h * r));
  const ctx = c.getContext("2d");
  ctx.setTransform(r, 0, 0, r, 0, 0);
  return { ctx, w, h, r };
}

/* glowing dot sprites, cached by colour */
const spriteCache = new Map();
function sprite(color, size = 32) {
  const key = color + size;
  if (spriteCache.has(key)) return spriteCache.get(key);
  const c = document.createElement("canvas"); c.width = c.height = size;
  const g = c.getContext("2d"), h = size / 2;
  const grad = g.createRadialGradient(h, h, 0, h, h, h);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(.18, color);
  grad.addColorStop(.42, color.replace(/hsl\(([^)]*)\)/, "hsla($1,.35)").replace(/^#(..)(..)(..)$/, (m, r, gg, b) => `rgba(${parseInt(r, 16)},${parseInt(gg, 16)},${parseInt(b, 16)},.35)`));
  grad.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grad; g.fillRect(0, 0, size, size);
  spriteCache.set(key, c); return c;
}

/* ================= DotField: decorative families of dots with gifts ================= */
class DotField {
  constructor(canvas, o = {}) {
    this.c = canvas; this.o = Object.assign({ n: 700, fams: 34, mode: "near", rate: 2.2, spread: 46, size: 15, drift: .25, alpha: 1 }, o);
    this.ready = false;
  }
  init() {
    const { ctx, w, h } = setup(this.c, 1.5); this.ctx = ctx; this.w = w; this.h = h;
    const o = this.o; this.fams = []; this.dots = []; this.gifts = []; this.ash = [];
    for (let f = 0; f < o.fams; f++) {
      const hue = Math.random() * 360;
      this.fams.push({ x: Math.random() * w, y: Math.random() * h, vx: (Math.random() - .5) * o.drift, vy: (Math.random() - .5) * o.drift,
        color: `hsl(${hue.toFixed(0)},72%,62%)`, hue });
    }
    for (let i = 0; i < o.n; i++) this.dots.push(this.spawn(this.fams[i % o.fams]));
    this.acc = 0; this.ready = true;
  }
  spawn(f, near) {
    const a = Math.random() * Math.PI * 2, r = Math.abs(gauss()) * this.o.spread;
    const bx = near ? near.x : f.x, by = near ? near.y : f.y;
    return { f, x: bx + Math.cos(a) * r * (near ? .3 : 1), y: by + Math.sin(a) * r * (near ? .3 : 1), ox: Math.random() * 6.28, life: 1, born: near ? 0 : 1 };
  }
  frame(dt) {
    if (!this.ready) return;
    const { ctx, w, h, o } = this;
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = `rgba(11,17,18,${o.trail || .35})`; ctx.fillRect(0, 0, w, h);
    for (const f of this.fams) {
      f.x += f.vx * dt * 60; f.y += f.vy * dt * 60;
      if (f.x < -40) f.x += w + 80; if (f.x > w + 40) f.x -= w + 80;
      if (f.y < -40) f.y += h + 80; if (f.y > h + 40) f.y -= h + 80;
    }
    this.acc += dt * o.rate;
    while (this.acc > 1) { this.acc -= 1; this.gift(); }
    ctx.globalCompositeOperation = "lighter";
    const t = performance.now() / 1000;
    for (const d of this.dots) {
      d.born = Math.min(1, d.born + dt * 1.5);
      const tx = d.f.x + Math.cos(d.ox + t * .3) * 6, ty = d.f.y + Math.sin(d.ox * 1.3 + t * .27) * 6;
      d.x += (tx - d.x) * .002 * dt * 60 + (Math.random() - .5) * .5;
      d.y += (ty - d.y) * .002 * dt * 60 + (Math.random() - .5) * .5;
      const s = o.size * (.55 + .45 * d.born);
      ctx.globalAlpha = o.alpha * (.55 + .45 * d.born);
      ctx.drawImage(sprite(d.f.color), d.x - s / 2, d.y - s / 2, s, s);
    }
    for (let i = this.gifts.length - 1; i >= 0; i--) {
      const g = this.gifts[i]; g.t += dt;
      const k = g.t / .9;
      if (k >= 1) { this.gifts.splice(i, 1); continue; }
      ctx.globalAlpha = (1 - k) * .9;
      ctx.strokeStyle = COL.ember; ctx.lineWidth = 2.2;
      ctx.beginPath(); ctx.moveTo(g.x0, g.y0); ctx.lineTo(g.x1, g.y1); ctx.stroke();
      const px = lerp(g.x0, g.x1, ease(k)), py = lerp(g.y0, g.y1, ease(k));
      ctx.globalAlpha = 1 - k * .5;
      ctx.drawImage(sprite(COL.ember, 48), px - 18, py - 18, 36, 36);
    }
    for (let i = this.ash.length - 1; i >= 0; i--) {
      const a = this.ash[i]; a.t += dt;
      if (a.t > 1.2) { this.ash.splice(i, 1); continue; }
      const s = 26 * (1 + a.t * 2);
      ctx.globalAlpha = (1 - a.t / 1.2) * .8;
      ctx.drawImage(sprite(COL.ember, 48), a.x - s / 2, a.y - s / 2, s, s);
    }
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over";
  }
  gift() {
    const dots = this.dots; if (dots.length < 3) return;
    const i = (Math.random() * dots.length) | 0, d = dots[i];
    let j;
    if (this.o.mode === "far") { j = (Math.random() * dots.length) | 0; }
    else { // nearest of a few random candidates from the same family
      let best = -1, bd = 1e9;
      for (let k = 0; k < 40; k++) { const c = (Math.random() * dots.length) | 0; if (c === i) continue;
        const dd = (dots[c].x - d.x) ** 2 + (dots[c].y - d.y) ** 2; if (dd < bd) { bd = dd; best = c; } }
      j = best;
    }
    if (j < 0 || j === i) return;
    const r = dots[j];
    this.gifts.push({ x0: d.x, y0: d.y, x1: r.x, y1: r.y, t: 0 });
    this.ash.push({ x: d.x, y: d.y, t: 0 });
    // the giver is gone; a relative of the recipient is born next to it (the recipient's family grows)
    dots[i] = this.spawn(r.f, r);
  }
}
function gauss() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

/* ================= background field ================= */
const bg = (() => {
  const c = $("#bg"); let ctx, pts = [], tint = COL.kin, on = true;
  function init() {
    const s = setup(c, 1); ctx = s.ctx; pts = [];
    for (let i = 0; i < 160; i++) pts.push({ x: Math.random() * 1920, y: Math.random() * 1080, z: Math.random(), vx: (Math.random() - .5) * .2, vy: (Math.random() - .5) * .2 });
  }
  function frame(dt) {
    if (!ctx) init();
    ctx.clearRect(0, 0, 1920, 1080);
    if (!on) return;
    ctx.globalCompositeOperation = "lighter";
    for (const p of pts) {
      p.x = (p.x + p.vx * dt * 60 + 1920) % 1920; p.y = (p.y + p.vy * dt * 60 + 1080) % 1080;
      const s = 6 + p.z * 16; ctx.globalAlpha = .08 + p.z * .16;
      ctx.drawImage(sprite(tint), p.x - s / 2, p.y - s / 2, s, s);
    }
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over";
  }
  return { frame, set(t, visible = true) { tint = t; on = visible; }, init };
})();

/* ================= slide modules ================= */
const M = {};

/* 1 title */
M["s-title"] = (() => { let f; return {
  enter() { if (!f) f = new DotField($("#title-canvas"), { n: 1100, fams: 46, rate: 3.2, spread: 60, size: 16, trail: .28 }); f.init(); bg.set(COL.kin, false); },
  frame(dt) { f.frame(dt); } }; })();

M["s-hook"] = { enter() { bg.set(COL.ember); } };

/* 3 torus (three.js) */
M["s-torus"] = (() => {
  let renderer, scene, camera, group, grassGeo, agentGeo, edgeLines = [], agents, t0, ready = false, grassUV, phaseEl;
  const N = 150, L = 2, NA = 520;
  function map(u, v, s1, s2, out, k) {
    // roll along v (glues top and bottom), then bend along u (glues left and right)
    let Y = v, Z = 0, R1 = 0;
    if (s1 > 1e-4) { R1 = L / (2 * Math.PI * s1); const ph = v / R1; Y = R1 * Math.sin(ph); Z = R1 * (1 - Math.cos(ph)); }
    const shrink = 1 - .45 * s2;
    if (R1 > 0) { Y *= shrink; Z = R1 + (Z - R1) * shrink; }
    let X = u, Z2 = Z;
    if (s2 > 1e-4) { const R2 = L / (2 * Math.PI * s2), ps = u / R2, d = (R1 + R2) - Z; X = d * Math.sin(ps); Z2 = (R1 + R2) - d * Math.cos(ps); }
    const cz = s1 * (L / (2 * Math.PI)) + s2 * (L / (2 * Math.PI));
    out[k] = X; out[k + 1] = Y; out[k + 2] = Z2 - cz;
  }
  function phase(t) {
    // loop of 17 s: flat, roll, bend, hold, unbend, unroll
    const T = t % 17;
    if (T < 2.5) return [0, 0, "a flat square map"];
    if (T < 5.5) return [easeIO((T - 2.5) / 3), 0, "top edge meets bottom edge"];
    if (T < 8.5) return [1, easeIO((T - 5.5) / 3), "left edge meets right edge"];
    if (T < 14) return [1, 1, "a torus: no edges anywhere"];
    if (T < 15.5) return [1, 1 - easeIO((T - 14) / 1.5), "unrolling"];
    return [1 - easeIO((T - 15.5) / 1.5), 0, "unrolling"];
  }
  function roundTex() {
    const c = document.createElement("canvas"); c.width = c.height = 64; const g = c.getContext("2d");
    const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(.35, "rgba(255,255,255,.8)"); gr.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
    const t = new THREE.CanvasTexture(c); return t;
  }
  function init() {
    if (!window.THREE) return false;
    const c = $("#torus");
    renderer = new THREE.WebGLRenderer({ canvas: c, antialias: true, alpha: true });
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(36, 1, .01, 100);
    group = new THREE.Group(); scene.add(group);
    const tex = roundTex();
    // grass points
    grassUV = new Float32Array(N * N * 2);
    const gp = new Float32Array(N * N * 3), gc = new Float32Array(N * N * 3);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const k = i * N + j, u = (i / (N - 1) - .5) * L, v = (j / (N - 1) - .5) * L;
      grassUV[2 * k] = u; grassUV[2 * k + 1] = v;
      const n = .5 + .5 * Math.sin(u * 7.1 + Math.sin(v * 5.3) * 1.7) * Math.cos(v * 6.2 - u * 2.1);
      gc[3 * k] = .1 + .14 * n; gc[3 * k + 1] = .25 + .38 * n; gc[3 * k + 2] = .16 + .16 * n;
    }
    grassGeo = new THREE.BufferGeometry();
    grassGeo.setAttribute("position", new THREE.BufferAttribute(gp, 3));
    grassGeo.setAttribute("color", new THREE.BufferAttribute(gc, 3));
    group.add(new THREE.Points(grassGeo, new THREE.PointsMaterial({ size: .022, vertexColors: true, map: tex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending })));
    // agents
    agents = new Float32Array(NA * 2);
    const ap = new Float32Array(NA * 3), ac = new Float32Array(NA * 3);
    for (let a = 0; a < NA; a++) {
      agents[2 * a] = (Math.random() - .5) * L; agents[2 * a + 1] = (Math.random() - .5) * L;
      const hot = Math.random() < .5; ac[3 * a] = 1; ac[3 * a + 1] = hot ? .72 : .78; ac[3 * a + 2] = hot ? .3 : .45;
    }
    agentGeo = new THREE.BufferGeometry();
    agentGeo.setAttribute("position", new THREE.BufferAttribute(ap, 3));
    agentGeo.setAttribute("color", new THREE.BufferAttribute(ac, 3));
    group.add(new THREE.Points(agentGeo, new THREE.PointsMaterial({ size: .06, vertexColors: true, map: tex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending })));
    // the four edges: left/right in ember, top/bottom in cyan
    const mk = (color) => { const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(120 * 3), 3));
      const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: .95, linewidth: 2 })); group.add(l); return g; };
    edgeLines = [["u", -L / 2, mk(0xff6b3d)], ["u", L / 2, mk(0xff6b3d)], ["v", -L / 2, mk(0x55c6da)], ["v", L / 2, mk(0x55c6da)]];
    phaseEl = $("#torus-phase");
    t0 = performance.now(); ready = true; resize(); return true;
  }
  function resize() {
    if (!renderer) return;
    const c = $("#torus"), w = c.offsetWidth, h = c.offsetHeight;
    renderer.setPixelRatio(clamp((window.devicePixelRatio || 1) * SCALE, 1, 2));
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  let lastPhase = "", spin = 0;
  return {
    enter() { bg.set(COL.moss); if (!ready && !init()) return; resize(); },
    resize,
    frame(dt) {
      if (!ready) return;
      const t = (performance.now() - t0) / 1000;
      const [s1, s2, label] = REDUCED ? [1, 1, "a torus: no edges anywhere"] : phase(t);
      if (label !== lastPhase) { phaseEl.textContent = label; lastPhase = label; }
      const gp = grassGeo.attributes.position.array;
      for (let k = 0; k < N * N; k++) map(grassUV[2 * k], grassUV[2 * k + 1], s1, s2, gp, 3 * k);
      grassGeo.attributes.position.needsUpdate = true;
      const ap = agentGeo.attributes.position.array;
      for (let a = 0; a < NA; a++) {
        let u = agents[2 * a] + (Math.random() - .5) * .012 + .0016, v = agents[2 * a + 1] + (Math.random() - .5) * .012 + .0011;
        if (u > L / 2) u -= L; if (u < -L / 2) u += L; if (v > L / 2) v -= L; if (v < -L / 2) v += L;
        agents[2 * a] = u; agents[2 * a + 1] = v;
        map(u, v, s1, s2, ap, 3 * a); ap[3 * a + 2] += .004;
      }
      agentGeo.attributes.position.needsUpdate = true;
      for (const [axis, val, g] of edgeLines) {
        const arr = g.attributes.position.array;
        for (let i = 0; i < 120; i++) { const w = (i / 119 - .5) * L; axis === "u" ? map(val, w, s1, s2, arr, 3 * i) : map(w, val, s1, s2, arr, 3 * i); }
        g.attributes.position.needsUpdate = true;
      }
      const torusness = s2;
      // Euler XYZ: spin about the ring's own axis (y) first, then tilt (x) so the hole faces the camera
      // spin is accumulated within each loop and reset while the map is flat, so every loop looks the same
      spin = torusness > 0 ? spin + dt * .35 * torusness : 0;
      group.rotation.y = spin * torusness;
      group.rotation.z = Math.sin(t * .2) * .18;
      group.rotation.x = lerp(-.95, -1.05, torusness) + Math.sin(t * .23) * .06;
      camera.position.set(0, 0, lerp(3.1, 2.05, torusness)); camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    } };
})();

/* 4 rule minis */
M["s-rules"] = (() => {
  const minis = []; let ready = false;
  function make(c) {
    const kind = c.dataset.mini; const s = setup(c, 1.5); const st = { kind, c, ...s, t: 0 };
    if (kind === "seasons" || kind === "eat") {
      st.G = kind === "seasons" ? 12 : 9; st.grass = new Float32Array(st.G * st.G).map(() => .3 + Math.random() * .7);
    }
    if (kind === "eat") { st.dot = { x: st.w * .3, y: st.h * .5, e: .35 }; }
    return st;
  }
  function grassColor(v) { const a = Math.sqrt(clamp(v, 0, 1)); return `rgb(${(10 + 52 * a) | 0},${(16 + 112 * a) | 0},${(14 + 66 * a) | 0})`; }
  function draw(m, dt) {
    const { ctx, w, h } = m; m.t += dt; const t = m.t;
    ctx.clearRect(0, 0, w, h); ctx.fillStyle = "#0a0f10"; ctx.fillRect(0, 0, w, h);
    if (m.kind === "seasons") {
      const G = m.G, cw = w / G, ch = (h - 56) / G, sf = Math.max(0, 1 + .9 * Math.sin(t * 2 * Math.PI / 6));
      for (let i = 0; i < G * G; i++) {
        let v = m.grass[i]; v += dt * sf * (.5 * v * (1 - v) + .04); if (Math.random() < .02) v *= .6; m.grass[i] = clamp(v, 0, 1);
        ctx.fillStyle = grassColor(m.grass[i]); ctx.fillRect((i % G) * cw + 1, ((i / G) | 0) * ch + 1, cw - 2, ch - 2);
      }
      const y = h - 34; ctx.strokeStyle = COL.rule; ctx.lineWidth = 2; ctx.beginPath();
      for (let x = 0; x <= w; x += 4) { const yy = y - 12 * Math.sin((x / w) * 2 * Math.PI); x ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy); } ctx.stroke();
      const ph = (t / 6) % 1, px = ph * w; ctx.drawImage(sprite(sf > 1 ? "#e7d27a" : "#6a8db8", 40), px - 14, y - 12 * Math.sin(ph * 2 * Math.PI) - 14, 28, 28);
      ctx.fillStyle = COL.muted; ctx.font = "500 18px JetBrains Mono, monospace"; ctx.fillText(sf > 1 ? "summer" : "winter", 10, h - 8);
    } else if (m.kind === "eat") {
      const G = m.G, cw = w / G, ch = (h - 40) / G, d = m.dot;
      for (let i = 0; i < G * G; i++) { m.grass[i] = clamp(m.grass[i] + dt * .05, 0, 1); ctx.fillStyle = grassColor(m.grass[i]); ctx.fillRect((i % G) * cw + 1, ((i / G) | 0) * ch + 1, cw - 2, ch - 2); }
      const cx = clamp((d.x / cw) | 0, 0, G - 1), cy = clamp((d.y / ch) | 0, 0, G - 1);
      let gx = 0, gy = 0;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) { if (!dx && !dy) continue; const xx = cx + dx, yy = cy + dy; if (xx < 0 || yy < 0 || xx >= G || yy >= G) continue;
        const diff = m.grass[yy * G + xx] - m.grass[cy * G + cx]; const n = Math.hypot(dx, dy); gx += diff * dx / n; gy += diff * dy / n; }
      let vx = gx * 3 + gauss() * .6, vy = gy * 3 + gauss() * .6; const n = Math.hypot(vx, vy) || 1;
      d.x = clamp(d.x + vx / n * 40 * dt, 8, w - 8); d.y = clamp(d.y + vy / n * 40 * dt, 8, h - 48);
      const k = cy * G + cx, bite = m.grass[k] * dt * 1.4; m.grass[k] -= bite; d.e = clamp(d.e + bite * .25 - dt * .05, 0, 1);
      ctx.globalCompositeOperation = "lighter"; ctx.drawImage(sprite(COL.kin, 64), d.x - 22, d.y - 22, 44, 44); ctx.globalCompositeOperation = "source-over";
      bar(ctx, 10, h - 26, w - 20, 12, d.e, COL.kin, "energy");
    } else if (m.kind === "die") {
      const T = t % 6, k = T / 5;
      // hunger: energy drains
      const e = clamp(1 - k * 1.25, 0, 1), fadeA = e > 0 ? 1 : clamp(1 - (k - .8) * 6, 0, 1);
      dotAt(ctx, w * .28, h * .4, COL.kin, 50 * (.5 + .5 * fadeA), fadeA);
      bar(ctx, w * .08, h * .72, w * .4, 12, e, COL.kin, "energy");
      // old age: ring fills
      const age = clamp(k * 1.25, 0, 1), fadeB = age < 1 ? 1 : clamp(1 - (k - .8) * 6, 0, 1);
      dotAt(ctx, w * .72, h * .4, COL.str, 50 * (.5 + .5 * fadeB), fadeB);
      ctx.strokeStyle = COL.old; ctx.lineWidth = 5; ctx.beginPath(); ctx.arc(w * .72, h * .4, 40, -Math.PI / 2, -Math.PI / 2 + age * Math.PI * 2); ctx.stroke();
      ctx.fillStyle = COL.muted; ctx.font = "500 18px JetBrains Mono, monospace"; ctx.textAlign = "center";
      ctx.fillText("hunger", w * .28, h * .9); ctx.fillText(`age ${Math.round(age * 300)}`, w * .72, h * .9); ctx.textAlign = "start";
    } else if (m.kind === "split") {
      const T = t % 5, grow = clamp(T / 2.2, 0, 1), sep = easeIO(clamp((T - 2.4) / 1.4, 0, 1));
      const cx = w / 2, cy = h * .42;
      if (sep <= 0) { dotAt(ctx, cx, cy, COL.kin, 40 + 22 * grow, 1); bar(ctx, w * .15, h * .78, w * .7, 12, grow, COL.kin, grow >= 1 ? "20 → split" : "energy"); }
      else {
        dotAt(ctx, cx - 70 * sep, cy, COL.kin, 50, 1); dotAt(ctx, cx + 70 * sep, cy + 10 * sep, "#f6c46b", 50, 1);
        ctx.fillStyle = COL.muted; ctx.font = "500 18px JetBrains Mono, monospace"; ctx.textAlign = "center";
        ctx.fillText("parent", cx - 70 * sep, h * .78); ctx.fillText("child · genes ± mutation", cx + 70 * sep, h * .86); ctx.textAlign = "start";
      }
    }
  }
  function dotAt(ctx, x, y, c, s, a) { ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = a; ctx.drawImage(sprite(c, 64), x - s / 2, y - s / 2, s, s); ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over"; }
  function bar(ctx, x, y, w, h, v, c, label) {
    ctx.fillStyle = COL.rule; ctx.fillRect(x, y, w, h); ctx.fillStyle = c; ctx.fillRect(x, y, w * v, h);
    if (label) { ctx.fillStyle = COL.muted; ctx.font = "500 16px JetBrains Mono, monospace"; ctx.fillText(label, x, y - 8); }
  }
  return {
    enter() { bg.set(COL.kin); if (!ready) { $$("[data-mini]").forEach(c => minis.push(make(c))); ready = true; } },
    resize() { minis.forEach(m => Object.assign(m, setup(m.c, 1.5))); },
    frame(dt) { minis.forEach(m => draw(m, dt)); } };
})();

/* 5 the gift */
M["s-option"] = (() => {
  let s; return {
    enter() { bg.set(COL.ember); s = setup($("#gift"), 1.5); s.t = 0; s.motes = []; },
    frame(dt) {
      const { ctx, w, h } = s; s.t += dt; const T = s.t % 7;
      ctx.clearRect(0, 0, w, h);
      const R = w / 2; const grd = ctx.createRadialGradient(R, R, 0, R, R, R);
      grd.addColorStop(0, "#152020"); grd.addColorStop(1, "#0b1112"); ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(R, R, R, 0, 7); ctx.fill();
      const approach = easeIO(clamp(T / 1.6, 0, 1));
      const gx = lerp(R - 250, R - 110, approach), gy = R + 20, rx = lerp(R + 250, R + 110, approach), ry = R - 20;
      const pour = clamp((T - 2) / 2.2, 0, 1), gone = clamp((T - 4.2) / 1, 0, 1);
      // meeting radius
      ctx.setLineDash([8, 10]); ctx.strokeStyle = `rgba(236,229,214,${.25 * (1 - gone)})`; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(gx, gy, 230, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      // motes of energy
      if (pour > 0 && pour < 1) for (let i = 0; i < 4; i++) s.motes.push({ t: 0, a: Math.random() * 6.28, o: (Math.random() - .5) * 60 });
      ctx.globalCompositeOperation = "lighter";
      for (let i = s.motes.length - 1; i >= 0; i--) {
        const m = s.motes[i]; m.t += dt / .9; if (m.t >= 1) { s.motes.splice(i, 1); continue; }
        const k = ease(m.t), x = lerp(gx, rx, k), y = lerp(gy, ry, k) + Math.sin(k * Math.PI) * m.o;
        ctx.globalAlpha = 1 - m.t * .6; ctx.drawImage(sprite(COL.ember, 32), x - 10, y - 10, 20, 20);
      }
      const gs = 120 * (1 - .85 * pour) * (1 - gone), rs = 120 * (1 + .45 * pour);
      ctx.globalAlpha = 1 - gone; ctx.drawImage(sprite(pour > .2 ? COL.ember : COL.kin, 64), gx - gs / 2, gy - gs / 2, gs, gs);
      ctx.globalAlpha = 1; ctx.drawImage(sprite(COL.kin, 64), rx - rs / 2, ry - rs / 2, rs, rs);
      ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
      // energy bars
      const eg = 9.6 * (1 - pour), er = 8 + 9.6 * .9 * pour;
      ctx.font = "500 24px JetBrains Mono, monospace"; ctx.textAlign = "center";
      ctx.fillStyle = `rgba(236,229,214,${1 - gone})`; ctx.fillText(gone > .5 ? "" : `E ${eg.toFixed(1)}`, gx, gy + 110);
      ctx.fillStyle = COL.paper; ctx.fillText(`E ${er.toFixed(1)}`, rx, ry + 120);
      ctx.fillStyle = COL.muted; ctx.font = "500 20px JetBrains Mono, monospace";
      ctx.fillText(gone > .3 ? "the giver dies" : "giver", gx, gy - 100); ctx.fillText(pour > 0 ? "neighbour · +90%" : "neighbour", rx, ry - 100);
      ctx.textAlign = "start";
      if (T > 6.6) s.motes.length = 0;
    } };
})();

M["s-missing"] = { enter() { bg.set(COL.ember); } };

/* 7 two worlds demo */
M["s-worlds"] = (() => { let a, b; return {
  enter() { bg.set(COL.kin, false);
    if (!a) { a = new DotField($('[data-worlddemo="kin"]'), { n: 420, fams: 16, rate: 3, spread: 26, size: 13, mode: "near", drift: .08, trail: .4 });
              b = new DotField($('[data-worlddemo="stranger"]'), { n: 420, fams: 16, rate: 3, spread: 26, size: 13, mode: "far", drift: .08, trail: .4 }); }
    a.init(); b.init(); },
  frame(dt) { a.frame(dt); b.frame(dt); } }; })();

M["s-question"] = { enter() { bg.set(COL.str); } };

/* 9 live model (Web Worker) */
function makeLive(id, opts) {
  let worker, frame = null, views = [], chart, ready = false, playing = true, lastGen = 0, showGrass = true, lastFrame = null;
  const BLANK = (255 << 24) | (12 << 16) | (11 << 8) | 7;  // plain dark ground when the grass layer is hidden
  const SIZE = 560, root = $("#" + id), q = s => $(s, root), qa = s => $$(s, root);
  function startWorker() {
    const src = window.KinModel.workerSource;
    worker = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
    worker.onmessage = e => { frame = e.data; };
    worker.onerror = e => console.error(`live model (${id}) stopped: ${e.message} (line ${e.lineno})`);
    worker.postMessage({ type: "init", seed: opts.seed || 7, fert: opts.fert || null });
  }
  function initView(c) {
    c.width = SIZE; c.height = SIZE; const ctx = c.getContext("2d");
    const img = ctx.createImageData(SIZE, SIZE); const lut = new Int32Array(SIZE); for (let p = 0; p < SIZE; p++) lut[p] = Math.floor(p * 75 / SIZE);
    return { c, ctx, img, buf: new Uint32Array(img.data.buffer), lut };
  }
  // colour of a grass cell: amount of grass (0..255) x fertility class (0..15; uniform maps use class 7)
  const LUT = new Uint32Array(256 * 16);
  for (let f = 0; f < 16; f++) for (let v = 0; v < 256; v++) {
    const a = Math.sqrt(v / 255), t = f / 15, k = opts.fert ? 1 : 0;
    const r = 7 + 34 * a + k * (t - .45) * 22, g = 11 + 80 * a + k * (t - .45) * 70, b = 12 + 40 * a + k * (t - .45) * 30;
    const c8 = x => x < 0 ? 0 : x > 255 ? 255 : x | 0;
    LUT[f * 256 + v] = (255 << 24) | (c8(b) << 16) | (c8(g) << 8) | c8(r);
  }
  function draw(v, w, near) {
    const { buf, lut, ctx, img } = v, grass = w.grass, fert = w.fert;
    if (!showGrass) buf.fill(BLANK);
    else for (let py = 0; py < SIZE; py++) { const cy = lut[py], row = py * SIZE; for (let px = 0; px < SIZE; px++) { const c = lut[px] * 75 + cy; buf[row + px] = LUT[(fert ? fert[c] : 7) * 256 + grass[c]]; } }
    const s = SIZE / 300, pos = w.pos, col = w.col;
    for (let i = 0; i < w.n; i++) {
      const x = (pos[2 * i] * s) | 0, y = (pos[2 * i + 1] * s) | 0; if (x >= SIZE - 1 || y >= SIZE - 1) continue;
      const c = (255 << 24) | (col[3 * i + 2] << 16) | (col[3 * i + 1] << 8) | col[3 * i], k = y * SIZE + x;
      buf[k] = c; buf[k + 1] = c; buf[k + SIZE] = c; buf[k + SIZE + 1] = c;
    }
    ctx.putImageData(img, 0, 0);
    const gf = w.gifts; ctx.lineWidth = 1.6;
    const wrap = d => d > 150 ? d - 300 : d < -150 ? d + 300 : d;
    for (let k = 0; k < gf.length; k += 4) {
      const xa = gf[k], ya = gf[k + 1], xb = gf[k + 2], yb = gf[k + 3];
      if (near) {
        // gifts to a neighbour are short on the torus: draw the shortest path, which may cross an
        // edge (then as two short pieces, one at each edge), never a line across the whole map
        const dx = wrap(xb - xa), dy = wrap(yb - ya);
        ctx.strokeStyle = "rgba(255,255,255,.95)"; ctx.beginPath();
        ctx.moveTo(xa * s, ya * s); ctx.lineTo((xa + dx) * s, (ya + dy) * s);
        if (dx !== xb - xa || dy !== yb - ya) { ctx.moveTo(xb * s, yb * s); ctx.lineTo((xb - dx) * s, (yb - dy) * s); }
        ctx.stroke();
      } else {
        ctx.strokeStyle = "rgba(255,255,255,.6)";
        ctx.beginPath(); ctx.moveTo(xa * s, ya * s); ctx.lineTo(xb * s, yb * s); ctx.stroke();
      }
    }
  }
  function drawChart(ws) {
    const { ctx, w, h } = chart; ctx.clearRect(0, 0, w, h);
    const L = 54, B = 34, T = 10, Rr = 10, PW = w - L - Rr, PH = h - T - B;
    const years = Math.max(30, ...ws.map(x => x.yearly.length));
    ctx.font = "500 18px JetBrains Mono, monospace"; ctx.fillStyle = COL.muted; ctx.strokeStyle = COL.rule; ctx.lineWidth = 1;
    for (const v of [0, .25, .5, .75, 1]) { const y = T + PH * (1 - v); ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + PW, y); ctx.stroke(); ctx.fillText(`${v * 100 | 0}%`, 0, y + 6); }
    ctx.fillText("year 0", L, h - 6); ctx.textAlign = "end"; ctx.fillText(`year ${years}`, L + PW, h - 6); ctx.textAlign = "start";
    [[ws[0].yearly, COL.kin], [ws[1].yearly, COL.str]].forEach(([ys, c]) => {
      if (!ys.length) return; ctx.strokeStyle = c; ctx.lineWidth = 4; ctx.lineJoin = "round"; ctx.beginPath();
      ys.forEach((v, i) => { const x = L + PW * (i + 1) / years, y = T + PH * (1 - v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    });
  }
  function bindControls() {
    const play = q(".lv-play");
    play.addEventListener("click", () => { playing = !playing; play.textContent = playing ? "pause" : "play"; play.setAttribute("aria-pressed", String(playing)); worker.postMessage({ type: "run", on: playing }); });
    qa("[data-speed]").forEach(b => b.addEventListener("click", () => { qa("[data-speed]").forEach(x => x.setAttribute("aria-pressed", String(x === b))); worker.postMessage({ type: "speed", v: +b.dataset.speed }); }));
    qa("[data-color]").forEach(b => b.addEventListener("click", () => { qa("[data-color]").forEach(x => x.setAttribute("aria-pressed", String(x === b))); worker.postMessage({ type: "color", v: +b.dataset.color }); }));
    q(".lv-reset").addEventListener("click", () => worker.postMessage({ type: "reset" }));
    const grassBtn = q(".lv-grass");
    grassBtn.addEventListener("click", () => {
      showGrass = !showGrass; grassBtn.setAttribute("aria-pressed", String(showGrass));
      if (lastFrame) lastFrame.worlds.forEach((w, k) => draw(views[k], w, k === 0));  // redraw at once, also when paused
    });
  }
  return {
    enter() { bg.set(COL.kin, false);
      if (!ready) { startWorker(); views = qa(".lv-view").map(initView); bindControls(); ready = true; }
      chart = setup(q(".lv-chart")); worker.postMessage({ type: "run", on: playing }); },
    leave() { if (worker) worker.postMessage({ type: "run", on: false }); },
    resize() { chart = setup(q(".lv-chart")); },
    frame() {
      if (!frame) return; const f = frame; frame = null; lastFrame = f;
      f.worlds.forEach((w, k) => { draw(views[k], w, k === 0); q(`.lv-pop[data-k="${k}"]`).textContent = `pop ${w.n.toLocaleString("en")}`; q(`.lv-share[data-k="${k}"]`).textContent = `${Math.round(w.share * 100)}%`; });
      // grass grows faster than average in the first half of the year (summer), slower in the second (winter)
      const st = f.worlds[0].step, season = (st % 600) < 300 ? "summer" : "winter";
      q(".lv-clock").textContent = `year ${Math.floor(st / 600)} · ${season} · step ${st.toLocaleString("en")}`;
      const note = q(".lv-note"), gen = f.worlds[0].fertGen;
      if (note && gen !== lastGen) { if (lastGen) { note.classList.remove("flash"); void note.offsetWidth; note.classList.add("flash"); } lastGen = gen; }
      drawChart(f.worlds);
    } };
}
M["s-live"] = makeLive("s-live", { seed: 7 });
M["s-fert"] = makeLive("s-fert", { seed: 7, fert: { contrast: 1, scale: 30, memory: 0 } });

/* ---------- chart helpers ---------- */
function axes(ctx, box, o) {
  const { L, T, W, H } = box;
  ctx.font = "500 20px JetBrains Mono, monospace"; ctx.lineWidth = 1;
  for (const v of o.yt) { const y = T + H * (1 - o.ys(v)); ctx.strokeStyle = COL.rule; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + W, y); ctx.stroke();
    ctx.fillStyle = COL.muted; ctx.textAlign = "end"; ctx.fillText(o.yf(v), L - 14, y + 7); }
  ctx.textAlign = "center";
  for (const v of o.xt) { const x = L + W * o.xs(v); ctx.fillStyle = COL.muted; ctx.fillText(o.xf(v), x, T + H + 34); }
  ctx.textAlign = "start";
  if (o.xl) { ctx.fillStyle = COL.muted; ctx.textAlign = "end"; ctx.fillText(o.xl, L + W, T + H + 64); ctx.textAlign = "start"; }
}
function poly(ctx, xs, ys, box, xsF, ysF, upto, color, width, alpha = 1) {
  const { L, T, W, H } = box; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = alpha; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) { const v = ys[i]; if (v == null || xsF(xs[i]) > upto) { if (xsF(xs[i]) > upto) break; continue; }
    const x = L + W * xsF(xs[i]), y = T + H * (1 - ysF(v)); started ? ctx.lineTo(x, y) : ctx.moveTo(x, y); started = true; }
  ctx.stroke(); ctx.globalAlpha = 1;
}

/* 10 result chart */
M["s-result"] = (() => { let s, t; return {
  enter() { bg.set(COL.kin); s = setup($("#chart-result")); t = REDUCED ? 99 : 0; countUp(); },
  resize() { s = setup($("#chart-result")); },
  frame(dt) {
    t += dt; const { ctx, w, h } = s; ctx.clearRect(0, 0, w, h);
    const box = { L: 100, T: 30, W: w - 130, H: h - 120 }, xs = v => v / 30000, ys = v => v / .8;
    axes(ctx, box, { yt: [0, .2, .4, .6, .8], ys, yf: v => `${Math.round(v * 100)}%`, xt: [0, 10000, 20000, 30000], xs, xf: v => v ? `${v / 1000}k` : "0", xl: "step" });
    const upto = ease((t - .6) / 2.8);
    for (const [key, c] of [["stranger", COL.str], ["neighbour", COL.kin]]) {
      const runs = (D.sacrifice || {})[key] || [];
      runs.forEach(r => poly(ctx, r.step, r.frac, box, xs, ys, upto, c, 2, .45));
      if (runs.length) { const n = Math.min(...runs.map(r => r.frac.length)); const mean = [];
        for (let i = 0; i < n; i++) { let s2 = 0, k = 0; runs.forEach(r => { if (r.frac[i] != null) { s2 += r.frac[i]; k++; } }); mean.push(k ? s2 / k : null); }
        poly(ctx, runs[0].step.slice(0, n), mean, box, xs, ys, upto, c, 6); }
    }
    ctx.font = "500 22px JetBrains Mono, monospace"; ctx.fillStyle = COL.muted; ctx.fillText("share of deaths that are self-sacrifice (thin: single runs, thick: mean)", box.L, 18);
  } }; })();

function countUp() {
  $$(".slide.active [data-count]").forEach(el => {
    const to = +el.dataset.count, suf = el.dataset.suffix || ""; if (REDUCED) { el.textContent = to + suf; return; }
    const t0 = performance.now() + 700; const step = () => { const k = ease((performance.now() - t0) / 2200); el.textContent = Math.round(to * k) + suf; if (k < 1) requestAnimationFrame(step); };
    el.textContent = "0" + suf; requestAnimationFrame(step);
  });
}

/* 11 p(E) over the years */
M["s-evolved"] = (() => { let s, t; return {
  enter() { bg.set(COL.ember); s = setup($("#chart-pe")); t = REDUCED ? 999 : 0; },
  resize() { s = setup($("#chart-pe")); },
  frame(dt) {
    t += dt; const { ctx, w, h } = s; ctx.clearRect(0, 0, w, h);
    const G = (D.genes || {}).neighbour; if (!G) return;
    const box = { L: 120, T: 40, W: w - 160, H: h - 130 }, lo = Math.log10(.005), hi = Math.log10(5);
    const xs = v => v / 20, ys = v => (Math.log10(Math.max(v, .005)) - lo) / (hi - lo);
    axes(ctx, box, { yt: [.01, .1, 1], ys, yf: v => `${v}%`, xt: [0, 5, 10, 15, 20], xs, xf: v => `${v}`, xl: "own energy (20 = ready to split)" });
    const E = Array.from({ length: 41 }, (_, i) => i / 2), rows = G.pe, nYears = rows.length;
    const k = Math.min(nYears - 1, Math.floor(Math.max(0, t - .8) * 4.2));
    for (let y = 0; y < k; y += 3) poly(ctx, E, rows[y], box, xs, ys, 1, `hsl(${30 - 20 * y / nYears},90%,${45 + 15 * y / nYears}%)`, 1.5, .18 + .25 * y / nYears);
    const S = (D.genes || {}).stranger; if (S && k >= nYears - 1) poly(ctx, E, S.pe[S.pe.length - 1], box, xs, ys, 1, COL.str, 3, .9);
    poly(ctx, E, rows[k], box, xs, ys, 1, COL.ember, 6);
    const yr = Math.round(G.pe_step[k] / 600);
    ctx.font = "400 64px Fraunces, Georgia, serif"; ctx.fillStyle = COL.paper; ctx.textAlign = "end"; ctx.fillText(`year ${yr}`, box.L + box.W, box.T + 60); ctx.textAlign = "start";
    ctx.font = "500 20px JetBrains Mono, monospace"; ctx.fillStyle = COL.muted; ctx.fillText("chance to sacrifice per encounter (log scale)", box.L, 22);
    if (S && k >= nYears - 1) { ctx.fillStyle = COL.str; ctx.fillText("control, final year", box.L + box.W - 250, box.T + box.H * (1 - ys(S.pe[S.pe.length - 1][40])) - 14); }
  } }; })();

M["s-why"] = { enter() { bg.set(COL.kin); } };

/* 13 population */
M["s-pop"] = (() => { let s, t; return {
  enter() { bg.set(COL.str); s = setup($("#chart-pop")); t = REDUCED ? 99 : 0; },
  resize() { s = setup($("#chart-pop")); },
  frame(dt) {
    t += dt; const { ctx, w, h } = s; ctx.clearRect(0, 0, w, h);
    const P = D.population; if (!P) return; const every = P.every, last = 6000 / every;
    const panels = [["neighbour", COL.kin], ["stranger", COL.str]], ph = (h - 40) / 2;
    const ymax = 11000;
    panels.forEach(([key, c], pi) => {
      const box = { L: 110, T: 10 + pi * ph + 16, W: w - 140, H: ph - 60 };
      const runs = P[key] || []; const n = runs.length ? runs[0].length : 0, start = Math.max(0, n - last);
      const xs = v => (v - start) / (n - 1 - start), ys = v => v / ymax;
      axes(ctx, box, { yt: [0, 5000, 10000], ys, yf: v => v ? `${v / 1000}k` : "0", xt: pi === 1 ? [start, start + 150, n - 1] : [], xs, xf: v => `year ${Math.round(v * every / 600)}` });
      const upto = ease((t - .5 - pi * .3) / 3);
      runs.forEach((r, ri) => { const idx = []; for (let i = start; i < n; i++) idx.push(i); poly(ctx, idx, idx.map(i => r[i]), box, xs, ys, upto, c, ri === 0 ? 3 : 2, ri === 0 ? .95 : .45); });
    });
  } }; })();

/* 14 family maps */
M["s-fam"] = (() => {
  let imgs = [], ready = false, frameIdx = 0, acc = 0, kymo = [];
  const HUE = []; for (let b = 0; b < 60; b++) { const hh = (b + .5) / 60; HUE.push(hsv(hh, .85, 1)); }
  function hsv(h, s, v) { const i = Math.floor(h * 6) % 6, f = h * 6 - Math.floor(h * 6), p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    const [r, g, b] = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i]; return [r * 255, g * 255, b * 255]; }
  function load() { ["neighbour", "stranger"].forEach((k, i) => { const im = new Image(); im.src = `${BASE}family_${k}.png`; imgs[i] = im; }); }
  function drawKymo(k, upto) {
    const F = (D.families || {})[k ? "stranger" : "neighbour"]; if (!F) return;
    const c = kymo[k]; const ctx = c.getContext("2d"); const n = F.hist.length;
    const img = ctx.createImageData(n, 60);
    for (let x = 0; x < n; x++) for (let b = 0; b < 60; b++) { const v = x <= upto ? Math.min(1, Math.sqrt((F.hist[x][b] || 0) / .15)) : 0; const o = 4 * ((59 - b) * n + x);
      img.data[o] = 17 + (HUE[b][0] - 17) * v; img.data[o + 1] = 26 + (HUE[b][1] - 26) * v; img.data[o + 2] = 27 + (HUE[b][2] - 27) * v; img.data[o + 3] = 255; }
    ctx.putImageData(img, 0, 0);
  }
  return {
    enter() { bg.set(COL.kin, false); if (!ready) { load(); kymo = [$("#kymo-0"), $("#kymo-1")]; kymo.forEach(c => { c.width = 150; c.height = 60; }); ready = true; } frameIdx = 0; acc = 0; },
    frame(dt) {
      acc += dt; if (acc > .14) { acc = 0; frameIdx = (frameIdx + 1) % 165; }
      const f = Math.min(frameIdx, 149);
      [0, 1].forEach(k => { const im = imgs[k]; if (!im || !im.complete || !im.naturalWidth) return; const c = $("#fam-" + k).getContext("2d");
        c.imageSmoothingEnabled = false; c.drawImage(im, (f % 15) * 75, Math.floor(f / 15) * 75, 75, 75, 0, 0, 75, 75); drawKymo(k, f); });
      $("#fam-year").textContent = `year ${f + 1} of 150`;
    } };
})();

/* 15 age at death */
M["s-age"] = (() => { let s = [], t; return {
  enter() { bg.set(COL.ember); s = [setup($("#chart-age-0")), setup($("#chart-age-1"))]; t = REDUCED ? 99 : 0; },
  resize() { s = [setup($("#chart-age-0")), setup($("#chart-age-1"))]; },
  frame(dt) {
    t += dt;
    ["neighbour", "stranger"].forEach((key, k) => {
      const { ctx, w, h } = s[k]; ctx.clearRect(0, 0, w, h); const A = (D.age || {})[key]; if (!A) return;
      const box = { L: 90, T: 34, W: w - 110, H: h - 110 }, ymax = .06, nb = A.sacrifice.length;
      axes(ctx, box, { yt: [0, .02, .04, .06], ys: v => v / ymax, yf: v => `${Math.round(v * 100)}%`, xt: [0, 100, 200, 300], xs: v => v / 310, xf: v => `${v}`, xl: "age at death (steps)" });
      const bw = box.W / 31;
      for (let i = 0; i < nb; i++) {
        const g = ease((t - .5 - i * .03 - k * .2) / .8); const sv = A.sacrifice[i] * g, hv = A.hunger[i] * g;
        const x = box.L + i * bw + 2, ys1 = box.T + box.H * (1 - sv / ymax), ys2 = box.T + box.H * (1 - (sv + hv) / ymax);
        ctx.fillStyle = COL.ember; ctx.fillRect(x, ys1, bw - 4, box.T + box.H - ys1);
        ctx.fillStyle = COL.grey; ctx.fillRect(x, ys2, bw - 4, ys1 - ys2);
      }
      const g = ease((t - 1.6 - k * .2) / .8), x = box.L + 30 * bw + 2;
      ctx.fillStyle = COL.old; ctx.fillRect(x, box.T + box.H * (1 - g), bw - 4, box.H * g);
      ctx.font = "500 20px JetBrains Mono, monospace"; ctx.fillStyle = COL.old; ctx.textAlign = "end";
      ctx.fillText(`old age: ${Math.round(A.old * 100)}% of deaths ↑`, x - 10, box.T + 22); ctx.textAlign = "start";
    });
  } }; })();


M["s-close"] = (() => { let f; return {
  enter() { bg.set(COL.ember, false); if (!f) f = new DotField($("#close-canvas"), { n: 900, fams: 40, rate: 5, spread: 55, size: 15, trail: .25, alpha: .55 }); f.init(); },
  frame(dt) { f.frame(dt); } }; })();

/* ================= navigation ================= */
const slides = $$(".slide"); let cur = -1, notesOn = false;
const counter = $("#counter"), bar = $("#progress i"), notes = $("#notes"), hint = $("#hint");
function go(i, dir = 1, origin) {
  i = clamp(i, 0, slides.length - 1); if (i === cur) return;
  const prev = slides[cur];
  if (prev) { prev.classList.remove("active", "back"); prev.classList.add("leaving"); setTimeout(() => prev.classList.remove("leaving"), 820); const pm = M[prev.id]; if (pm && pm.leave) pm.leave(); }
  cur = i; const el = slides[i];
  el.style.setProperty("--ox", origin ? origin[0] : "50%"); el.style.setProperty("--oy", origin ? origin[1] : "50%");
  el.classList.toggle("back", dir < 0); el.classList.add("active");
  $$(".frag", el).forEach(f => f.classList.toggle("on", dir < 0));
  const m = M[el.id]; if (m && m.enter) { try { m.enter(el); } catch (err) { console.error(err); } }
  if (!m || !m.enter) bg.set(COL.kin);
  counter.textContent = `${String(i + 1).padStart(2, "0")} / ${slides.length}`;
  bar.style.width = `${(i / (slides.length - 1)) * 100}%`;
  notes.innerHTML = `<b>SPEAKER NOTES · ${i + 1}</b>` + escapeHtml(el.dataset.notes || "");
  try { history.replaceState(null, "", `#s${i + 1}`); } catch (e) { location.hash = `s${i + 1}`; }
}
function escapeHtml(s) { return s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function next(origin) { if (hint) hint.style.opacity = 0; const el = slides[cur]; const f = $$(".frag:not(.on)", el)[0]; if (f) { f.classList.add("on"); return; } go(cur + 1, 1, origin); }
function prev() { go(cur - 1, -1); }
function resizeActive() { const m = M[slides[cur].id]; if (m && m.resize) m.resize(); bg.init(); }
document.addEventListener("keydown", e => {
  const tag = (e.target && e.target.tagName) || "";
  if ((e.key === " " || e.key === "Enter") && (tag === "BUTTON")) return;
  if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) { e.preventDefault(); next(); }
  else if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) { e.preventDefault(); prev(); }
  else if (e.key === "Home") go(0, -1); else if (e.key === "End") go(slides.length - 1);
  else if (e.key === "n" || e.key === "N") { notesOn = !notesOn; notes.hidden = !notesOn; }
  else if (e.key === "f" || e.key === "F") { const d = document.documentElement; if (!document.fullscreenElement) { d.requestFullscreen && d.requestFullscreen().catch(() => {}); } else { document.exitFullscreen && document.exitFullscreen().catch(() => {}); } }
});
function zoneOrigin(ev) { const r = stage.getBoundingClientRect(); return [`${((ev.clientX - r.left) / r.width * 100).toFixed(1)}%`, `${((ev.clientY - r.top) / r.height * 100).toFixed(1)}%`]; }
$("#nav-next").addEventListener("click", e => { e.stopPropagation(); next(); });
$("#nav-prev").addEventListener("click", e => { e.stopPropagation(); prev(); });
// click on the slide: left quarter goes back, elsewhere forward (not on controls or the live view)
stage.addEventListener("click", ev => {
  if (ev.target.closest("button, a, .live-grid, #notes")) return;
  const r = stage.getBoundingClientRect(); ((ev.clientX - r.left) / r.width < .25) ? prev() : next(zoneOrigin(ev));
});
let tx = null;
stage.addEventListener("touchstart", e => { tx = e.touches[0].clientX; }, { passive: true });
stage.addEventListener("touchend", e => { if (tx == null) return; const dx = e.changedTouches[0].clientX - tx; if (Math.abs(dx) > 50) dx < 0 ? next() : prev(); tx = null; });
window.addEventListener("hashchange", () => { const m = /^#s(\d+)$/.exec(location.hash); if (m && +m[1] - 1 !== cur) go(+m[1] - 1, +m[1] - 1 > cur ? 1 : -1); });

/* ================= main loop ================= */
let last = performance.now();
function loop(now) {
  const dt = Math.min(.05, (now - last) / 1000); last = now;
  bg.frame(dt);
  const el = slides[cur]; const m = el && M[el.id];
  if (m && m.frame) { try { m.frame(dt); } catch (err) { console.error(err); } }
  requestAnimationFrame(loop);
}
const start = /^#s(\d+)$/.exec(location.hash);
go(start ? +start[1] - 1 : 0);
requestAnimationFrame(loop);
})();
