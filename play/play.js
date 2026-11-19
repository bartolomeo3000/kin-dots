/* Kin-dots playground: settings panel + the two live worlds (model in ../presentation/assets/model.js). */
(() => {
"use strict";
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const COL = { kin: "#f3b54a", str: "#55c6da", muted: "#8e9b95", rule: "#253434" };

/* ---------------- settings ---------------- */
// key = the model's parameter name (model.js DEFAULTS); the label says what it means
const GROUPS = [
  ["World", [
    { k: "harsh", label: "Winter harshness", min: 0, max: 1, step: .05, def: .9, hint: "0 = no seasons; 0.9 = almost no regrowth in midwinter" },
    { k: "season", label: "Year length (steps)", min: 200, max: 1500, step: 50, def: 600 },
    { k: "growth", label: "Grass growth rate", min: .01, max: .15, step: .005, def: .05 },
    { k: "inflow", label: "Grass inflow", min: 0, max: .3, step: .01, def: .1, hint: "regrowth even on bare ground" },
  ]],
  ["Dots", [
    { k: "n0", label: "Dots at start", min: 500, max: 5000, step: 100, def: 2500 },
    { k: "speed", label: "Walking speed", min: .02, max: 1, step: .02, def: .1, hint: "faster dots mix families more" },
    { k: "spread", label: "Child born this far from parent", min: .5, max: 30, step: .5, def: 2 },
    { k: "maxAge", label: "Maximum age (steps)", min: 100, max: 1000, step: 25, def: 300 },
    { k: "birth", label: "Energy needed to split", min: 10, max: 40, step: 1, def: 20 },
  ]],
  ["Sacrifice", [
    { k: "radius", label: "Meeting radius", min: 1, max: 10, step: .5, def: 4, hint: "how close a neighbour must be to be met" },
    { k: "eff", label: "Share of the gift that arrives", min: 0, max: 1, step: .05, def: .9 },
    { k: "a0", label: "Starting urge (gene a)", min: -12, max: -3, step: .5, def: -9, hint: "−9 ≈ 1 in 8,000 per step at the start" },
    { k: "mut", label: "Mutation size", min: 0, max: .5, step: .01, def: .15 },
  ]],
];
const LAND = [
  { k: "contrast", label: "Contrast", min: 0, max: 2, step: .1, def: 1, hint: "how much richer the fertile patches are" },
  { k: "scale", label: "Patch size", min: 10, max: 100, step: 5, def: 30 },
  { k: "memory", label: "Map memory from year to year", min: 0, max: 1, step: .05, def: 1, hint: "1 = fixed map, 0 = a new map every year, between = drifting" },
];
const ALL = GROUPS.flatMap(g => g[1]).concat(LAND);
const PRESETS = [
  ["Demo", {}],
  ["Mild winter", { harsh: .6 }],
  ["No seasons", { harsh: 0 }],
  ["Fast dots", { speed: .5 }],
  ["Children far away", { spread: 20 }],
  ["Fixed landscape", { land: 1 }],
  ["New landscape every year", { land: 1, memory: 0 }],
  ["Drifting landscape", { land: 1, memory: .7 }],
];

function fieldHTML(f, prefix = "") {
  return `<div class="field" data-k="${prefix}${f.k}"><label for="f-${prefix}${f.k}">${f.label}</label><output id="o-${prefix}${f.k}" for="f-${prefix}${f.k}"></output>
    <input id="f-${prefix}${f.k}" type="range" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.def}">${f.hint ? `<span class="hint">${f.hint}</span>` : ""}</div>`;
}
$("#fields").innerHTML = GROUPS.map(([name, fs]) => `<fieldset><legend>${name}</legend>${fs.map(f => fieldHTML(f)).join("")}</fieldset>`).join("")
  + `<fieldset><legend>Landscape</legend><label class="switch"><input type="checkbox" id="f-land"> grass regrows unevenly, like a relief map</label>
     <div class="landscape-opts" id="land-opts" hidden>${LAND.map(f => fieldHTML(f, "l-")).join("")}</div></fieldset>`;
$("#presets").innerHTML = PRESETS.map(([n], i) => `<button type="button" class="chip" data-preset="${i}" aria-pressed="false">${n}</button>`).join("");

const fmt = (f, v) => f.step >= 1 ? String(Math.round(v)) : (+v).toFixed(f.step < .01 ? 3 : 2).replace(/\.?0+$/, m => m.includes(".") ? "" : m);
const input = (f, prefix = "") => $(`#f-${prefix}${f.k}`);
function refreshField(f, prefix = "") {
  const el = input(f, prefix), v = +el.value;
  $(`#o-${prefix}${f.k}`).textContent = fmt(f, v);
  el.closest(".field").classList.toggle("changed", Math.abs(v - f.def) > 1e-9);
}
function refreshAll() {
  GROUPS.forEach(([, fs]) => fs.forEach(f => refreshField(f)));
  LAND.forEach(f => refreshField(f, "l-"));
  $("#land-opts").hidden = !$("#f-land").checked;
}
function readSettings() {
  const params = {};
  GROUPS.forEach(([, fs]) => fs.forEach(f => { params[f.k] = +input(f).value; }));
  const fert = $("#f-land").checked ? Object.fromEntries(LAND.map(f => [f.k, +input(f, "l-").value])) : null;
  const seed = Math.max(1, Math.round(+$("#seed").value || 7));
  return { params, fert, seed };
}
function applyValues(vals) {
  GROUPS.forEach(([, fs]) => fs.forEach(f => { input(f).value = vals[f.k] ?? f.def; }));
  LAND.forEach(f => { input(f, "l-").value = vals[f.k] ?? f.def; });
  $("#f-land").checked = !!vals.land;
  if (vals.seed) $("#seed").value = vals.seed;
  refreshAll();
}
// settings <-> URL query (?harsh=0.6&land=1&memory=0&seed=3), so a setup can be shared as a link
function toQuery() {
  const { params, fert, seed } = readSettings(), q = new URLSearchParams();
  GROUPS.forEach(([, fs]) => fs.forEach(f => { if (Math.abs(params[f.k] - f.def) > 1e-9) q.set(f.k, params[f.k]); }));
  if (fert) { q.set("land", 1); LAND.forEach(f => { if (Math.abs(fert[f.k] - f.def) > 1e-9) q.set(f.k, fert[f.k]); }); }
  if (seed !== 7) q.set("seed", seed);
  return q.toString();
}
function fromQuery() {
  const q = new URLSearchParams(location.search), vals = {};
  ALL.forEach(f => { if (q.has(f.k)) { const v = parseFloat(q.get(f.k)); if (isFinite(v)) vals[f.k] = Math.min(f.max, Math.max(f.min, v)); } });
  if (q.get("land") === "1") vals.land = 1;
  if (q.has("seed")) vals.seed = Math.max(1, parseInt(q.get("seed"), 10) || 7);
  return vals;
}

$("#fields").addEventListener("input", e => {
  if (e.target.matches("input[type=range]")) { const k = e.target.id.replace(/^f-/, ""); const f = k.startsWith("l-") ? LAND.find(x => "l-" + x.k === k) : ALL.find(x => x.k === k); refreshField(f, k.startsWith("l-") ? "l-" : ""); }
  if (e.target.id === "f-land") $("#land-opts").hidden = !e.target.checked;
  $$(".chip").forEach(c => c.setAttribute("aria-pressed", "false"));
  status("Changed. Restart to apply.");
});
$("#presets").addEventListener("click", e => {
  const b = e.target.closest(".chip"); if (!b) return;
  const [name, vals] = PRESETS[+b.dataset.preset];
  applyValues({ ...vals, seed: +$("#seed").value });
  $$(".chip").forEach(c => c.setAttribute("aria-pressed", String(c === b)));
  restart(`Running “${name}”.`);
});
$("#settings").addEventListener("submit", e => { e.preventDefault(); restart("Restarted with your settings."); });
$("#defaults").addEventListener("click", () => { applyValues({ seed: 7 }); $$(".chip").forEach(c => c.setAttribute("aria-pressed", String(c.dataset.preset === "0"))); restart("Back to the demo settings."); });
$("#seed-random").addEventListener("click", () => { $("#seed").value = 1 + Math.floor(Math.random() * 99999); status("New seed. Restart to apply."); });
$("#copy-link").addEventListener("click", () => {
  const url = location.origin + location.pathname + (toQuery() ? "?" + toQuery() : "");
  const done = () => status("Link copied: it opens this page with these settings.");
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(url).then(done, () => status(url));
  else status(url);
});
let statusTimer;
function status(t) { $("#status").textContent = t; clearTimeout(statusTimer); statusTimer = setTimeout(() => { $("#status").textContent = ""; }, 6000); }

/* ---------------- live model ---------------- */
const SIZE = 560;
let worker = null, frame = null, last = null, playing = true, showGrass = true, popHist = [[], []], useFert = false, yearLen = 600;
function restart(msg) {
  const { params, fert, seed } = readSettings();
  useFert = !!fert; buildLUT(); yearLen = params.season;
  if (!worker) {
    worker = new Worker(URL.createObjectURL(new Blob([window.KinModel.workerSource], { type: "text/javascript" })));
    worker.onmessage = e => { frame = e.data; };
    worker.onerror = e => status(`The simulation stopped: ${e.message}`);
  }
  worker.postMessage({ type: "init", seed, params, fert });
  worker.postMessage({ type: "run", on: playing });
  popHist = [[], []];
  try { history.replaceState(null, "", location.pathname + (toQuery() ? "?" + toQuery() : "")); } catch (e) { /* ignore */ }
  if (msg) status(msg);
}
const views = ["#view-0", "#view-1"].map(id => {
  const c = $(id); c.width = SIZE; c.height = SIZE; const ctx = c.getContext("2d"); const img = ctx.createImageData(SIZE, SIZE);
  const lut = new Int32Array(SIZE); for (let p = 0; p < SIZE; p++) lut[p] = Math.floor(p * 75 / SIZE);
  return { ctx, img, buf: new Uint32Array(img.data.buffer), lut };
});
const LUT = new Uint32Array(256 * 16), BLANK = (255 << 24) | (12 << 16) | (11 << 8) | 7;
function buildLUT() {
  for (let f = 0; f < 16; f++) for (let v = 0; v < 256; v++) {
    const a = Math.sqrt(v / 255), t = f / 15, k = useFert ? 1 : 0, c8 = x => x < 0 ? 0 : x > 255 ? 255 : x | 0;
    LUT[f * 256 + v] = (255 << 24) | (c8(12 + 40 * a + k * (t - .45) * 30) << 16) | (c8(11 + 80 * a + k * (t - .45) * 70) << 8) | c8(7 + 34 * a + k * (t - .45) * 22);
  }
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
  const gf = w.gifts, wrap = d => d > 150 ? d - 300 : d < -150 ? d + 300 : d; ctx.lineWidth = 1.6;
  for (let q = 0; q < gf.length; q += 4) {
    const xa = gf[q], ya = gf[q + 1], xb = gf[q + 2], yb = gf[q + 3];
    if (near) {  // short on the torus: shortest path, in two pieces when it crosses an edge
      const dx = wrap(xb - xa), dy = wrap(yb - ya);
      ctx.strokeStyle = "rgba(255,255,255,.95)"; ctx.beginPath(); ctx.moveTo(xa * s, ya * s); ctx.lineTo((xa + dx) * s, (ya + dy) * s);
      if (dx !== xb - xa || dy !== yb - ya) { ctx.moveTo(xb * s, yb * s); ctx.lineTo((xb - dx) * s, (yb - dy) * s); }
      ctx.stroke();
    } else { ctx.strokeStyle = "rgba(255,255,255,.6)"; ctx.beginPath(); ctx.moveTo(xa * s, ya * s); ctx.lineTo(xb * s, yb * s); ctx.stroke(); }
  }
}
function chartCtx(c) {
  const r = Math.min(2, window.devicePixelRatio || 1), w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * r) || c.height !== Math.round(h * r)) { c.width = Math.round(w * r); c.height = Math.round(h * r); }
  const ctx = c.getContext("2d"); ctx.setTransform(r, 0, 0, r, 0, 0); ctx.clearRect(0, 0, w, h); return { ctx, w, h };
}
function lineChart(c, series, ymax, yfmt, xlabel) {
  const { ctx, w, h } = chartCtx(c), L = 46, B = 22, T = 6, R = 8, PW = w - L - R, PH = h - T - B;
  const n = Math.max(2, ...series.map(s => s.ys.length));
  ctx.font = "500 11px JetBrains Mono, monospace"; ctx.fillStyle = COL.muted; ctx.strokeStyle = COL.rule; ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const v = ymax * k / 4, y = T + PH * (1 - k / 4); ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + PW, y); ctx.stroke(); ctx.fillText(yfmt(v), 0, y + 4); }
  ctx.fillText(xlabel(0), L, h - 4); ctx.textAlign = "end"; ctx.fillText(xlabel(n), L + PW, h - 4); ctx.textAlign = "start";
  series.forEach(({ ys, color }) => { if (ys.length < 1) return; ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.lineJoin = "round"; ctx.beginPath();
    ys.forEach((v, i) => { const x = L + PW * (i + 1) / n, y = T + PH * (1 - Math.min(v, ymax) / ymax); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); });
}
function render() {
  if (frame) {
    const f = frame; frame = null; last = f;
    f.worlds.forEach((w, k) => {
      draw(views[k], w, k === 0);
      $(`#pop-${k}`).textContent = `pop ${w.n.toLocaleString("en")}`;
      $(`#share-${k}`).textContent = `${Math.round(w.share * 100)}%`;
      const ph = popHist[k]; if (!ph.length || w.step - ph[ph.length - 1][0] >= 50) ph.push([w.step, w.n]);
    });
    // grass grows faster than average in the first half of the year (summer), slower in the second (winter)
    const st = f.worlds[0].step, season = (st % yearLen) < yearLen / 2 ? "summer" : "winter";
    $("#clock").textContent = `year ${f.worlds[0].yearly.length} · ${season} · step ${st.toLocaleString("en")}`;
    lineChart($("#chart-share"), [{ ys: f.worlds[0].yearly, color: COL.kin }, { ys: f.worlds[1].yearly, color: COL.str }], 1, v => `${Math.round(v * 100)}%`, n => `year ${n === 0 ? 0 : n}`);
    const pmax = Math.max(1000, ...popHist.flat().map(p => p[1]));
    lineChart($("#chart-pop"), popHist.map((ph, k) => ({ ys: ph.map(p => p[1]), color: k ? COL.str : COL.kin })), Math.ceil(pmax / 2000) * 2000,
      v => v >= 1000 ? `${Math.round(v / 1000)}k` : `${v}`, n => n === 0 ? "step 0" : `step ${(st).toLocaleString("en")}`);
  }
  requestAnimationFrame(render);
}

/* ---------------- controls ---------------- */
$("#play").addEventListener("click", () => { playing = !playing; $("#play").textContent = playing ? "pause" : "play"; $("#play").setAttribute("aria-pressed", String(playing)); worker.postMessage({ type: "run", on: playing }); });
$$("[data-speed]").forEach(b => b.addEventListener("click", () => { $$("[data-speed]").forEach(x => x.setAttribute("aria-pressed", String(x === b))); worker.postMessage({ type: "speed", v: +b.dataset.speed }); }));
$$("[data-color]").forEach(b => b.addEventListener("click", () => { $$("[data-color]").forEach(x => x.setAttribute("aria-pressed", String(x === b))); worker.postMessage({ type: "color", v: +b.dataset.color }); }));
$("#grass").addEventListener("click", () => { showGrass = !showGrass; $("#grass").setAttribute("aria-pressed", String(showGrass)); if (last) last.worlds.forEach((w, k) => draw(views[k], w, k === 0)); });

/* ---------------- start ---------------- */
const initial = fromQuery();
applyValues(initial);
if (!Object.keys(initial).length) $('.chip[data-preset="0"]').setAttribute("aria-pressed", "true");
restart();
requestAnimationFrame(render);
})();
