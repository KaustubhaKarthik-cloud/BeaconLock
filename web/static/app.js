/* BeaconLock dashboard client -- vanilla JS, no external libraries (offline). */
"use strict";

const $ = (id) => document.getElementById(id);
/* Semantic state palette (matches style.css): green locked, amber acquiring, red lost. */
const STATE_COLORS = { IDLE: "#6b7786", SEARCHING: "#aeb7c2", ACQUIRING: "#e8a33d", LOCKED: "#4fb477", LOST: "#e0574c" };
const ACCENT = "#e8a33d";
const MONO = '"JetBrains Mono", "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
const DIST = ["turbulence", "vibration", "noise", "blur"];

async function api(path, body) {
  const opt = body === undefined ? { method: "POST" } :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function fillSettings(s) {
  $("trajectory").value = s.trajectory;
  for (const k of ["speed", "acquire_n", "acquire_m", "lose_n", "lose_m", "seed", ...DIST]) $(k).value = s[k];
  $("show_truth").checked = !!s.show_truth;
  refreshOutputs();
}

function refreshOutputs() {
  $("speed-out").textContent = $("speed").value;
  for (const k of DIST) $(k + "-out").textContent = String($(k).value).padStart(3, "\u2007");
}

function readScenario() {
  const num = (k) => Number($(k).value);
  return {
    trajectory: $("trajectory").value, speed: num("speed"),
    acquire_n: num("acquire_n"), acquire_m: num("acquire_m"), lose_n: num("lose_n"), lose_m: num("lose_m"),
    seed: num("seed"), show_truth: $("show_truth").checked,
    ...Object.fromEntries(DIST.map((k) => [k, num(k)])),
  };
}

function msg(text, ok) {
  const m = $("config-msg");
  m.textContent = text;
  m.style.color = ok ? "var(--ok)" : "var(--warn)";
  if (text) setTimeout(() => { if (m.textContent === text) m.textContent = ""; }, 3000);
}

/* ---------------- live disturbance sliders (debounced) ---------------- */
let distTimer = null;
function onDistChange() {
  refreshOutputs();
  clearTimeout(distTimer);
  distTimer = setTimeout(() => {
    const body = Object.fromEntries(DIST.map((k) => [k, Number($(k).value)]));
    body.show_truth = $("show_truth").checked;
    api("/api/disturbances", body).catch((e) => msg(e.message));
  }, 80);
}

/* --------------- eased number tickers with dim unit suffix --------------- */
const unitSpan = (u) => (u ? `<span class="u">${u}</span>` : "");
function setNum(id, v, digits, unit) {
  const el = $(id);
  if (v === null || v === undefined || Number.isNaN(Number(v))) {
    el._target = null; el._cur = null;
    el.innerHTML = `<span class="nil">\u2013</span>${unitSpan(unit)}`;
    return;
  }
  v = Number(v);
  el._from = (el._cur === null || el._cur === undefined) ? v : el._cur;
  el._target = v; el._t0 = performance.now(); el._digits = digits; el._unit = unit;
  if (el._raf) return;
  const frame = (now) => {
    if (el._target === null) { el._raf = null; return; }
    const k = Math.min(1, (now - el._t0) / 200);
    const e = 1 - Math.pow(1 - k, 3);                 // ease-out cubic, no overshoot
    el._cur = el._from + (el._target - el._from) * e;
    el.innerHTML = el._cur.toFixed(el._digits) + unitSpan(el._unit);
    el._raf = k < 1 ? requestAnimationFrame(frame) : null;
  };
  el._raf = requestAnimationFrame(frame);
}

/* ------------------------------ chart ------------------------------ */
function niceMax(v) {
  const steps = [20, 40, 60, 80, 100, 120, 160, 200, 240, 320, 400];  // divisible by 4 -> integer ticks
  return steps.find((s) => s >= v) || 400;
}

function drawChart(h, lockRadius) {
  const c = $("chart"), ctx = c.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const L = 44, R = 12, T = 12, B = 26, pw = W - L - R, ph = H - T - B;
  const n = h.err.length, N = 300;
  // y-scale: focus on the tracking regime, clip large search errors.
  const recentMax = Math.max(10, ...h.err.filter((_, i) => h.state[i] === "LOCKED"));
  const ymax = niceMax(Math.min(Math.max(lockRadius * 1.25, recentMax * 1.15), 400));
  const y = (v) => T + ph - Math.min(v, ymax) / ymax * ph;
  const x = (i) => L + (i + (N - n)) / (N - 1) * pw;

  ctx.font = `10px ${MONO}`;
  ctx.textBaseline = "middle";
  // state bands (very faint) + 3px state strip along the time axis
  for (let i = 0; i < n; i++) {
    const col = STATE_COLORS[h.state[i]] || "#6b7786";
    const w = pw / (N - 1) + 0.6;
    ctx.fillStyle = col + "12"; ctx.fillRect(x(i), T, w, ph);
    ctx.fillStyle = col + "b0"; ctx.fillRect(x(i), T + ph - 3, w, 3);
  }
  // horizontal graticule + y ticks
  ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = ymax * k / 4, yy = Math.round(y(v)) + 0.5;
    ctx.strokeStyle = "#1c2530";
    ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(L + pw, yy); ctx.stroke();
    ctx.strokeStyle = "#46515e";
    ctx.beginPath(); ctx.moveTo(L - 4, yy); ctx.lineTo(L, yy); ctx.stroke();
    ctx.fillStyle = "#6b7786"; ctx.textAlign = "right";
    ctx.fillText(v.toFixed(0), L - 7, yy);
  }
  // vertical graticule + time ticks every 2 s
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 1; i < n; i++) {
    if (Math.floor(h.t[i] / 2) !== Math.floor(h.t[i - 1] / 2)) {
      const xx = Math.round(x(i)) + 0.5, label = (Math.floor(h.t[i] / 2) * 2).toFixed(0);
      ctx.strokeStyle = "#1c2530";
      ctx.beginPath(); ctx.moveTo(xx, T); ctx.lineTo(xx, T + ph); ctx.stroke();
      ctx.strokeStyle = "#46515e";
      ctx.beginPath(); ctx.moveTo(xx, T + ph); ctx.lineTo(xx, T + ph + 4); ctx.stroke();
      if (xx > L + 12 && xx < L + pw - 24) { ctx.fillStyle = "#6b7786"; ctx.fillText(label, xx, T + ph + 7); }
    }
  }
  // axes
  ctx.strokeStyle = "#2a3542";
  ctx.beginPath(); ctx.moveTo(L + 0.5, T); ctx.lineTo(L + 0.5, T + ph + 0.5); ctx.lineTo(L + pw, T + ph + 0.5); ctx.stroke();
  // units
  ctx.fillStyle = "#46515e"; ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillText("px", 8, T + ph + 7);
  ctx.textAlign = "right";
  ctx.fillText("t [s]", L + pw, T + ph + 7);
  // lock-radius reference
  const ly = Math.round(y(lockRadius)) + 0.5;
  ctx.setLineDash([2, 3]); ctx.strokeStyle = "#6b7786";
  ctx.beginPath(); ctx.moveTo(L, ly); ctx.lineTo(L + pw, ly); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#6b7786"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.fillText(`LOCK RADIUS ${lockRadius.toFixed(0)} px`, L + 6, ly - 2);
  // error trace (the key datum -> accent)
  ctx.save();
  ctx.beginPath(); ctx.rect(L, T, pw, ph); ctx.clip();
  ctx.strokeStyle = ACCENT; ctx.lineWidth = 1.25; ctx.lineJoin = "round"; ctx.beginPath();
  for (let i = 0; i < n; i++) { const xx = x(i), yy = y(h.err[i]); i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); }
  ctx.stroke();
  ctx.restore();
  // latest value tag
  if (n) {
    ctx.fillStyle = ACCENT; ctx.textAlign = "right"; ctx.textBaseline = "top";
    ctx.fillText(`${h.err[n - 1].toFixed(1)} px  @ ${h.t[n - 1].toFixed(1)} s`, L + pw - 4, T + 4);
  }
}

/* ---------------------------- polling ---------------------------- */
async function pollState() {
  try {
    const s = await (await fetch("/api/state", { cache: "no-store" })).json();
    document.body.classList.toggle("is-running", !!s.running);
    const b = $("badge");
    b.textContent = s.state + (s.running ? "" : (s.state === "IDLE" ? "" : " (paused)"));
    b.className = "badge " + s.state;
    $("s-state").textContent = s.state;
    $("s-state").className = s.state;
    setNum("s-fps", s.running ? s.fps : null, 1, "fps");
    $("s-cap").textContent = s.fps_processing_capacity ? `capacity ${s.fps_processing_capacity.toFixed(0)} fps` : "";
    setNum("s-err", s.error_px, 1, "px");
    setNum("s-avg", s.err_locked_avg_px, 2, "px");
    $("s-max").textContent = s.err_locked_max_px != null ? `max ${s.err_locked_max_px.toFixed(1)} px` : "";
    setNum("s-ret", s.lock_retention_pct, 1, "%");
    $("s-lockpct").textContent = s.lock_pct != null ? `overall lock ${s.lock_pct.toFixed(1)} %` : "";
    setNum("s-acq", s.acquisition_time_s, 2, "s");
    $("s-loss").textContent = `${s.lock_losses ?? 0} / ${s.reacquisitions ?? 0}`;
    setNum("s-t", s.t, 1, "s");
    $("s-proc").textContent = s.proc_ms != null ? `frame time ${s.proc_ms.toFixed(1)} ms` : "";
    drawChart(s.history, s.lock_radius_px);
  } catch (e) { /* server restarting */ }
  setTimeout(pollState, 200);
}

function pollMinimap() {
  const img = new Image();
  img.onload = () => { $("minimap").src = img.src; setTimeout(pollMinimap, 150); };
  img.onerror = () => setTimeout(pollMinimap, 1000);
  img.src = "/api/minimap.jpg?ts=" + Date.now();
}

function startCamera() {
  // MJPEG stream; fall back to polling single frames if the browser can't do multipart.
  const cam = $("cam");
  cam.onerror = () => {
    cam.onerror = null;
    const poll = () => {
      const im = new Image();
      im.onload = () => { cam.src = im.src; setTimeout(poll, 33); };
      im.onerror = () => setTimeout(poll, 500);
      im.src = "/api/frame.jpg?ts=" + Date.now();
    };
    poll();
  };
  cam.src = "/video.mjpg";
}

/* --------------------------- UTC mission clock --------------------------- */
function tickClock() {
  const iso = new Date().toISOString();            // always UTC
  $("utc-clock").textContent = `${iso.slice(0, 10)} ${iso.slice(11, 19)}`;
}

/* ------------------------------ init ------------------------------ */
async function init() {
  tickClock();
  setInterval(tickClock, 250);
  const cfg = await (await fetch("/api/config")).json();
  $("trajectory").innerHTML = cfg.trajectories.map((t) => `<option value="${t}">${t.replace("_", " ")}</option>`).join("");
  fillSettings(cfg.settings);
  $("speed").addEventListener("input", refreshOutputs);
  for (const k of DIST) $(k).addEventListener("input", onDistChange);
  $("show_truth").addEventListener("change", onDistChange);
  $("btn-start").onclick = () => api("/api/start").catch((e) => msg(e.message));
  $("btn-stop").onclick = () => api("/api/stop").catch((e) => msg(e.message));
  $("btn-reset").onclick = () => api("/api/reset").then(() => msg("Reset", true)).catch((e) => msg(e.message));
  $("btn-apply").onclick = () => api("/api/config", readScenario())
    .then((r) => { fillSettings(r.settings); msg("Applied - press Start", true); })
    .catch((e) => msg(e.message));
  startCamera();
  pollMinimap();
  pollState();
}
init();
