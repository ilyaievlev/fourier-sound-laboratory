/* «Как ИИ слышит мир» — клиентская часть. Все расчёты на сервере, здесь только графики и формулы. */
"use strict";

// ---------------------------------------------------------------- утилиты
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const CSRF = $("input[name=csrfmiddlewaretoken]").value;

function f32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

/** Число для KaTeX: d значащих цифр, десятичная запятая, степень 10 для очень малых/больших. */
function num(v, d = 4) {
  if (v === null || v === undefined || !isFinite(v)) return "\\text{—}";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e6) {
    const dec = Math.max(0, d - 1 - Math.floor(Math.log10(a)));
    let s = v.toFixed(dec);
    if (dec > 0) s = s.replace(/0+$/, "").replace(/\.$/, "");
    let [ip, fp] = s.split(".");
    if (Math.abs(+ip) >= 10000) ip = ip.replace(/\B(?=(\d{3})+(?!\d))/g, "\\,");
    return fp ? `${ip}{,}${fp}` : ip;
  }
  const e = Math.floor(Math.log10(a));
  const m = (v / 10 ** e).toFixed(d - 1).replace(/0+$/, "").replace(/\.$/, "").replace(".", "{,}");
  return `${m}\\cdot 10^{${e}}`;
}
/** То же для обычного текста. */
function txt(v, d = 4) {
  return num(v, d).replace(/\{,\}/g, ",").replace(/\\,/g, " ").replace(/\\cdot 10\^\{(-?\d+)\}/, "·10^$1").replace("\\text{—}", "—");
}
const pct = (v, d = 3) => txt(100 * v, d) + "%";
const sign = (v) => (v < 0 ? "-" : "+");

function tex(el, s, display = true) {
  katex.render(s, el, { throwOnError: false, displayMode: display, strict: "ignore" });
}
/** Заполняет блок «Где здесь математика»: список [формула | {html}] */
function mathBlock(id, lines) {
  const box = $("#" + id);
  box.innerHTML = "";
  for (const ln of lines) {
    const div = document.createElement("div");
    div.className = "line";
    if (typeof ln === "string") tex(div, ln);
    else div.innerHTML = ln.html;
    box.appendChild(div);
  }
  $$(".tex", box).forEach((e) => tex(e, e.dataset.t, false));
}
const T = (s) => `<span class="tex" data-t="${s.replace(/"/g, "&quot;")}"></span>`;

function debounce(fn, ms = 180) {
  let id;
  return (...a) => { clearTimeout(id); id = setTimeout(() => fn(...a), ms); };
}

async function api(path, params = {}, opts = {}) {
  const q = new URLSearchParams(params).toString();
  const r = await fetch(path + (q ? "?" + q : ""), opts);
  const j = await r.json().catch(() => ({ error: `Ошибка сервера (${r.status})` }));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const post = (path, body, isJson = false) => api(path, {}, {
  method: "POST",
  headers: { "X-CSRFToken": CSRF, ...(isJson ? { "Content-Type": "application/json" } : {}) },
  body: isJson ? JSON.stringify(body) : body,
});

function showError(e) {
  const box = $("#input-error");
  box.textContent = e.message || String(e);
  box.hidden = false;
  console.error(e);
}

// ---------------------------------------------------------------- графики
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const C = { accent: "#2458a6", gray: "#9aa0a6", pos: "#2e8b57", neg: "#c0392b", orange: "#d9822b" };
function baseLayout(extra = {}) {
  const fs = parseFloat(getComputedStyle(document.body).fontSize);
  return Object.assign({
    margin: { l: 60, r: 20, t: 30, b: 45 },
    font: { size: Math.round(fs * 0.8) },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    legend: { orientation: "h", y: 1.12, x: 0 },
    hovermode: "closest",
    separators: ", ",
  }, extra);
}
const PCONF = { responsive: true, displaylogo: false };
function plot(id, data, layout) { return Plotly.react(id, data, baseLayout(layout), PCONF); }
const range = (n, f) => Array.from({ length: n }, (_, i) => f(i));

// ---------------------------------------------------------------- состояние
const S = {
  rid: null, sr: 0, n: 0, duration: 0, wave: null, quality: null, meta: null,
  t0: 0, T: 30, N: 10, src: "rec", n3: 1, kind: "cos", m: 2,
  fs: 8000, aa: true, win: "hann", scale: "db", fmax: 5000, bw: 15, ops: [],
  step: 1, spec: null, edited: null, period: null,
  mode5: "custom", fscale: "lin", win5: 1024, overlap: 0.75, wtype: "hann", mels: 64, fmax5: 8000, t5: null,
};

// ---------------------------------------------------------------- вход
async function loadRecording(j) {
  Object.assign(S, {
    rid: j.rid, sr: j.sr, n: j.n, duration: j.duration, wave: f32(j.wave),
    quality: j.quality, meta: j.meta, t0: j.suggested_t0, ops: [], spec: null, edited: null, period: null, t5: null,
  });
  S.fs = S.sr >= 8000 ? 8000 : 2000;
  $("#input-error").hidden = true;
  $("#audio").src = `/api/rec/${S.rid}/audio.wav`;
  $("#edit-audio").removeAttribute("src");
  $("#player").hidden = $("#stepper").hidden = $("#steps").hidden = false;
  $("#rec-name").textContent = j.meta.name + (j.meta.synthetic ? " (синтетический пример)" : "");
  $("#rec-info").textContent = `${txt(S.duration, 3)} с · ${S.sr} Гц · ${S.n.toLocaleString("ru")} отсчётов` +
    (j.meta.original_channels > 1 ? ` · из ${j.meta.original_channels} каналов` : "") +
    (j.meta.original_sr !== S.sr ? ` · исходно ${j.meta.original_sr} Гц` : "") +
    (j.meta.trimmed ? " · обрезано до 10 с" : "") + (j.meta.padded ? " · дополнено тишиной до 1 с" : "");
  renderQuality();
  renderMini();
  $("#player").scrollIntoView({ behavior: "smooth" });
  goStep(1, true);
}

function renderQuality() {
  const q = S.quality, out = [];
  out.push(`<span>пик ${txt(q.peak_dbfs, 3)} дБFS</span>`, `<span>RMS ${txt(q.rms_dbfs, 3)} дБFS</span>`);
  if (q.is_silent) out.push(`<span class="warn">почти тишина: RMS ниже ${q.thresholds.silence_rms_dbfs} дБFS</span>`);
  if (q.is_clipped) out.push(`<span class="warn">клиппинг: ${pct(q.clip_fraction)} отсчётов на пределе</span>`);
  if (q.frame_contrast_db !== null) {
    out.push(`<span class="${q.maybe_noisy ? "warn" : ""}" title="Отношение энергии 10% самых громких кадров к 10% самых тихих. Это SNR, только если в записи были паузы.">контраст кадров ${txt(q.frame_contrast_db, 3)} дБ${q.maybe_noisy ? " — если были паузы, фон громкий" : ""}</span>`);
  }
  if (Math.abs(q.dc_offset) > 0.01) out.push(`<span class="warn">смещение нуля ${txt(q.dc_offset, 2)}</span>`);
  $("#quality").innerHTML = out.join("");
}

function envelope(x, buckets) {
  const step = Math.max(1, Math.floor(x.length / buckets));
  const t = [], y = [];
  for (let i = 0; i < x.length; i += step) {
    let lo = Infinity, hi = -Infinity;
    for (let j = i; j < Math.min(i + step, x.length); j++) { if (x[j] < lo) lo = x[j]; if (x[j] > hi) hi = x[j]; }
    t.push(i / S.sr, i / S.sr); y.push(lo, hi);
  }
  return { t, y };
}

function segShape() {
  return [{ type: "rect", xref: "x", yref: "paper", x0: S.t0, x1: S.t0 + S.T / 1000, y0: 0, y1: 1,
    fillcolor: "rgba(217,130,43,.35)", line: { color: C.orange, width: 1 } }];
}

function renderMini() {
  const e = envelope(S.wave, 1500);
  plot("mini-wave", [{ x: e.t, y: e.y, mode: "lines", line: { color: C.accent, width: 1 }, hoverinfo: "none" }],
    { margin: { l: 5, r: 5, t: 5, b: 18 }, height: 70, showlegend: false, shapes: segShape(),
      yaxis: { visible: false, fixedrange: true }, xaxis: { fixedrange: true, ticksuffix: " с" } });
  $("#mini-wave").removeAllListeners?.("plotly_click");
  $("#mini-wave").on("plotly_click", (ev) => setT0(ev.points[0].x));
  updateSegInfo();
}

function updateSegInfo() {
  $("#seg-info").textContent = `участок: t₀ = ${txt(S.t0, 4)} с, T = ${txt(S.T, 3)} мс`;
}

function setT0(t) {
  const maxT0 = Math.max(0, S.duration - S.T / 1000 - 1 / S.sr);
  S.t0 = Math.min(Math.max(0, t), maxT0);
  Plotly.relayout("mini-wave", { shapes: segShape() });
  updateSegInfo();
  if (S.step === 1) renderStep1Wave();
  if (S.step === 2 || S.step === 3) refresh();
}

// запись с микрофона
async function record(sec, btn) {
  if (!navigator.mediaDevices?.getUserMedia) return showError(new Error("Браузер не даёт доступ к микрофону. Откройте страницу по адресу http://localhost:8000."));
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
  } catch (e) { return showError(new Error("Нет доступа к микрофону: " + e.message)); }
  const mime = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4", ""].find((m) => m === "" || MediaRecorder.isTypeSupported(m));
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  $$(".rec").forEach((b) => (b.disabled = true));
  btn.classList.add("recording");
  let left = sec;
  const status = $("#rec-status");
  status.textContent = `идёт запись… ${left} с`;
  const timer = setInterval(() => { left--; status.textContent = left > 0 ? `идёт запись… ${left} с` : "обработка…"; }, 1000);
  rec.onstop = async () => {
    clearInterval(timer);
    stream.getTracks().forEach((t) => t.stop());
    btn.classList.remove("recording");
    $$(".rec").forEach((b) => (b.disabled = false));
    const type = rec.mimeType || "audio/webm";
    const ext = type.includes("ogg") ? ".ogg" : type.includes("mp4") ? ".m4a" : ".webm";
    const fd = new FormData();
    fd.append("file", new Blob(chunks, { type }), "mic" + ext);
    fd.append("source", "mic");
    try { await loadRecording(await post("/api/upload", fd)); status.textContent = ""; }
    catch (e) { status.textContent = ""; showError(e); }
  };
  rec.start();
  setTimeout(() => rec.state !== "inactive" && rec.stop(), sec * 1000);
}

// ---------------------------------------------------------------- степпер
function goStep(k, force = false) {
  if (S.step === k && !force) return;
  S.step = k;
  $$("#stepper button").forEach((b) => b.classList.toggle("active", +b.dataset.step === k));
  $$(".step").forEach((s) => (s.hidden = +s.dataset.step !== k));
  refresh();
  // Plotly не знает размеров скрытых графиков
  setTimeout(() => $$(`.step[data-step="${k}"] .plot`).forEach((p) => p.data && Plotly.Plots.resize(p)), 50);
}

let seq = 0;
async function refresh() {
  if (!S.rid) return;
  const my = ++seq;
  try {
    if (S.step === 1) await step1(my);
    else if (S.step === 2) await step2(my);
    else if (S.step === 3) await step3(my);
    else if (S.step === 4) await step4(my);
    else if (S.step === 5) await step5(my);
  } catch (e) { showError(e); }
}
const refreshSoon = debounce(refresh, 160);

// ---------------------------------------------------------------- шаг 1
let s1 = null;
async function step1(my) {
  const r = await api(`/api/rec/${S.rid}/step1`, { fs: S.fs, aa: S.aa ? 1 : 0 });
  if (my !== seq) return;
  s1 = r;
  const box = $("#fs-options");
  box.innerHTML = "";
  for (const o of r.options) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = o.fs >= 1000 ? `${txt(o.fs / 1000, 3)} кГц` : `${o.fs} Гц`;
    b.disabled = !o.available;
    if (!o.available) b.title = `Исходная частота записи ${S.sr} Гц: повысить её нельзя, новых данных не появится`;
    b.classList.toggle("active", o.fs === r.fs);
    b.onclick = () => { S.fs = o.fs; refresh(); };
    box.appendChild(b);
  }
  renderStep1Wave();

  const ex = r.example;
  const lines = [
    `x[n] = f(n\\,\\Delta t),\\qquad \\Delta t = \\frac{1}{f_s} = \\frac{1}{${num(r.fs, 5)}\\ \\text{Гц}} = ${num(r.dt * 1000, 4)}\\ \\text{мс}`,
    `x[${ex.n}] = f(${ex.n}\\cdot ${num(r.dt * 1000, 4)}\\ \\text{мс}) = f(${num(ex.t, 5)}\\ \\text{с}) = ${num(ex.value, 4)}`,
    `f_{\\text{Найквиста}} = \\frac{f_s}{2} = ${num(r.nyquist, 5)}\\ \\text{Гц}`,
    `f_{99\\%} = ${num(r.f99, 4)}\\ \\text{Гц}:\\quad f_s = ${num(r.fs, 5)} ${r.kotelnikov_ok ? ">" : "\\le"} 2 f_{99\\%} = ${num(2 * r.f99, 4)}\\ \\text{Гц}`,
    { html: `Здесь <i>f</i><sub>99%</sub> — частота, ниже которой лежит 99% энергии записи (из ДПФ на шаге 4). Условие Котельникова для неё ${r.kotelnikov_ok ? '<span class="ok">выполнено</span>' : '<span class="bad">не выполнено</span>'}. Выше ${txt(r.nyquist, 4)} Гц лежит ${pct(r.energy_above_nyquist)} энергии.` },
  ];
  if (r.f_peak_alias !== null) {
    lines.push(`f_{\\text{пик}} = ${num(r.f_peak, 4)}\\ \\text{Гц} > \\frac{f_s}{2}\\ \\Rightarrow\\ \\left|f - f_s\\cdot\\operatorname{round}\\tfrac{f}{f_s}\\right| = ${num(r.f_peak_alias, 4)}\\ \\text{Гц}`);
    lines.push({ html: r.antialias ? "С фильтром эта компонента подавлена до дискретизации." : `Без фильтра самый сильный пик записи прозвучит как ${txt(r.f_peak_alias, 4)} Гц.` });
  }
  mathBlock("m1", lines);
  const up = r.fs >= r.sr;
  $("#s1-summary").textContent = up
    ? `Это исходная частота записи: ${r.sr} значений в секунду. Микрофон уже «нарезал» непрерывный звук на отсчёты через каждые ${txt(r.dt * 1000, 3)} мс.`
    : `При fₛ = ${txt(r.fs, 5)} Гц берётся ${txt(r.fs, 5)} значений в секунду, шаг ${txt(r.dt * 1000, 3)} мс. Частоты выше ${txt(r.nyquist, 4)} Гц так не передать: в этой записи там ${pct(r.energy_above_nyquist)} энергии. ` +
      (r.energy_above_nyquist < 0.001 ? "Это ничтожная доля, поэтому на слух разницы почти нет." :
       r.antialias ? "Фильтр их убирает, звук становится глуше." : "Без фильтра они превращаются в ложные низкие частоты: это и есть алиасинг.");
}

function renderStep1Wave() {
  if (!s1) return;
  const e = envelope(S.wave, 4000);
  plot("p1-wave", [{ x: e.t, y: e.y, mode: "lines", line: { color: C.accent, width: 1 }, name: "x(t)", hoverinfo: "x" }],
    { title: { text: "Вся запись (клик выбирает участок)" }, xaxis: { title: { text: "t, с" } },
      yaxis: { title: { text: "x(t)" } }, showlegend: false, shapes: segShape() });
  const p = $("#p1-wave");
  p.removeAllListeners?.("plotly_click");
  p.on("plotly_click", (ev) => setT0(ev.points[0].x));

  // увеличенный фрагмент 10 мс: исходный сигнал и отсчёты при fs
  const W = 0.01, i0 = Math.floor(S.t0 * S.sr), i1 = Math.min(S.n, i0 + Math.ceil(W * S.sr));
  const tx = range(i1 - i0, (i) => (i0 + i) / S.sr * 1000), xx = Array.from(S.wave.subarray(i0, i1));
  const ys = f32(s1.samples), j0 = Math.ceil(S.t0 * s1.fs), j1 = Math.min(ys.length, Math.floor((S.t0 + W) * s1.fs) + 1);
  const st = [], sy = [], mt = [], my = [];
  for (let j = j0; j < j1; j++) {
    const t = (j / s1.fs) * 1000;
    st.push(t, t, null); sy.push(0, ys[j], null); mt.push(t); my.push(ys[j]);
  }
  plot("p1-zoom", [
    { x: tx, y: xx, mode: "lines", line: { color: C.gray, width: 1.5 }, name: `запись (${S.sr} Гц)` },
    { x: st, y: sy, mode: "lines", line: { color: C.orange, width: 1 }, hoverinfo: "skip", showlegend: false },
    { x: mt, y: my, mode: "markers", marker: { color: C.orange, size: 7 }, name: `x[n] при fₛ = ${s1.fs} Гц` },
  ], { title: { text: `Увеличено: 10 мс от t₀, Δt = ${txt(1000 / s1.fs, 3)} мс` }, xaxis: { title: { text: "t, мс" } } });
}

// ---------------------------------------------------------------- шаг 2
async function step2(my) {
  const r = await api(`/api/rec/${S.rid}/series`, { t0: S.t0, T: S.T, N: S.N, src: S.src });
  if (my !== seq) return;
  S.period = r.period || null;
  $("#fit-period").disabled = !(S.src === "rec" && r.period);
  $("#fit-period").title = r.period ? `Основной тон ≈ ${txt(r.period.f0, 4)} Гц, период ${txt(r.period.P * 1000, 4)} мс` : "Явной периодичности на участке не найдено";
  const Ns = $("#N");
  Ns.max = r.n_cap;
  if (S.N > r.n_cap) { S.N = r.n_cap; Ns.value = S.N; }
  $("#N-val").textContent = r.N;

  const t = Array.from(f32(r.t), (v) => v * 1000), f = f32(r.f), Sn = f32(r.S);
  const data = [
    { x: t, y: f, mode: "lines", line: { color: C.gray, width: 2 }, name: S.src === "rec" ? "f(t) — участок записи" : "f(t)" },
    { x: t, y: Sn, mode: "lines", line: { color: C.accent, width: 2 }, name: `S_${r.N}(t)` },
  ];
  if ($("#show-harm").checked) {
    r.harmonics.forEach((h, i) => data.push({ x: t, y: f32(h), mode: "lines", line: { width: 1, dash: "dot" }, name: `n = ${i + 1}` }));
  }
  plot("p2-sum", data, { title: { text: "Сигнал и частичная сумма ряда" }, xaxis: { title: { text: "t, мс" } } });
  plot("p2-err", [{ x: t, y: f32(r.err), mode: "lines", line: { color: C.neg, width: 1 }, name: "f − S_N" }],
    { title: { text: `Ошибка f(t) − S_${r.N}(t)` }, xaxis: { title: { text: "t, мс" } }, showlegend: false });
  const rm = f32(r.rmse);
  plot("p2-rmse", [
    { x: range(rm.length, (i) => i), y: Array.from(rm, (v) => Math.max(v, 1e-12)), mode: "lines", line: { color: C.accent }, name: "RMSE(N)" },
    { x: [r.N], y: [Math.max(r.rmse_N, 1e-12)], mode: "markers", marker: { color: C.orange, size: 10 }, name: "текущее N" },
  ], { title: { text: "Среднеквадратичная ошибка от N" }, xaxis: { title: { text: "N" } }, yaxis: { type: "log" }, showlegend: false });

  const te = Array.from(f32(r.ext.t), (v) => v * 1000);
  const ext = [
    { x: te, y: f32(r.ext.f), mode: "lines", line: { color: C.gray, width: 1.5 }, name: "периодическое продолжение участка" },
    { x: te, y: f32(r.ext.S), mode: "lines", line: { color: C.accent, width: 1.5 }, name: `S_${r.N}` },
  ];
  if (r.ext.real) ext.push({ x: te, y: f32(r.ext.real), mode: "lines", line: { color: C.pos, width: 1, dash: "dot" }, name: "что на самом деле в записи" });
  const Tm = r.T * 1000;
  plot("p2-ext", ext, { title: { text: "Три периода: где стыкуются копии участка" }, xaxis: { title: { text: "t − t₀, мс" } },
    shapes: [0, Tm].map((x) => ({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: C.orange, dash: "dash" } })) });

  // формулы с числами
  const c = r.coeffs, lines = [];
  lines.push(`T = ${num(Tm, 4)}\\ \\text{мс},\\quad \\omega = \\frac{2\\pi}{T} = ${num(r.omega, 5)}\\ \\text{рад/с},\\quad M = ${r.M}\\ \\text{отсчётов на периоде}`);
  const terms = [];
  for (let n = 1; n <= Math.min(3, r.N); n++) {
    terms.push(`${sign(c[n].a)} ${num(Math.abs(c[n].a), 3)}\\cos ${n > 1 ? n : ""}\\omega t ${sign(c[n].b)} ${num(Math.abs(c[n].b), 3)}\\sin ${n > 1 ? n : ""}\\omega t`);
  }
  lines.push(`S_{${r.N}}(t) = ${num(c[0].a / 2, 3)} ${terms.join(" ")}${r.N > 3 ? " + \\dots" : ""}`);
  lines.push(`\\text{RMSE} = \\sqrt{\\tfrac1M\\textstyle\\sum_j (f(t_j) - S_{${r.N}}(t_j))^2} = ${num(r.rmse_N, 4)}`);
  lines.push(`\\frac{a_0^2}{2} + \\sum_{n=1}^{${r.N}}(a_n^2+b_n^2) = ${num(r.bessel_N, 5)} \\;\\le\\; \\frac{2}{T}\\int_0^T f^2\\,dt = ${num(r.energy, 5)}\\quad(${num(100 * r.bessel_N / r.energy, 4)}\\%)`);
  if (r.N < r.n_cap) lines.push({ html: `При N = ${r.n_cap} (наибольшее различимое по M = ${r.M} отсчётам) сумма равна ${txt(r.bessel_cap, 5)}: дискретный аналог равенства Парсеваля.` });
  let table = `<table class="coef"><tr><th>n</th><th>aₙ</th><th>bₙ</th>${r.exact ? "<th>точно aₙ</th><th>точно bₙ</th>" : ""}</tr>`;
  c.forEach((row, i) => {
    table += `<tr><td>${row.n}</td><td>${txt(row.a, 4)}</td><td>${txt(row.b, 4)}</td>` +
      (r.exact ? `<td>${txt(r.exact[i].a, 4)}</td><td>${txt(r.exact[i].b, 4)}</td>` : "") + "</tr>";
  });
  table += "</table>";
  lines.push({ html: table });
  if (r.exact) {
    const formula = { square: "b_n = \\frac{4}{\\pi n}\\ (n\\ \\text{нечётное}),\\ a_n = 0", saw: "b_n = \\frac{2(-1)^{n+1}}{\\pi n},\\ a_n = 0", abs: "a_0 = 1,\\ a_n = -\\frac{4}{\\pi^2 n^2}\\ (n\\ \\text{нечётное}),\\ b_n = 0" }[S.src];
    lines.push(`${formula};\\quad \\max_n |\\text{численно} - \\text{точно}| = ${num(r.coef_max_err, 2)}`);
  }
  if (r.gibbs) {
    lines.push(`\\max S_{${r.N}} = ${num(r.gibbs.max_S, 5)},\\quad \\frac{\\max S_N - 1}{\\text{скачок } 2} = ${num(100 * r.gibbs.overshoot_fraction, 4)}\\%\\quad(\\text{предел } ${num(100 * r.gibbs.theory, 4)}\\%)`);
    if (S.src === "saw") lines.push({ html: `У пилы f около разрыва наклонная, поэтому max S<sub>N</sub> − 1 приближается к пределу медленнее, с поправкой порядка 1/(2N) = ${txt(50 / r.N, 3)}%.` });
  }
  if (r.boundary && S.src === "rec") {
    const b = r.boundary, ratio = b.typical_step > 0 ? Math.abs(b.jump) / b.typical_step : 0;
    lines.push(`f(0+) - f(T-0) = ${num(b.f0, 3)} - (${num(b.fT, 3)}) = ${num(b.jump, 3)}\\quad(\\text{типичный шаг между отсчётами } ${num(b.typical_step, 2)})`);
    lines.push({ html: ratio > 3 ? `Скачок на границе в ${txt(ratio, 2)} раз больше обычного шага между отсчётами: у периодического продолжения здесь разрыв, и частичные суммы «звенят» у краёв (эффект Гиббса).` : "Скачка на границе почти нет: периодическое продолжение близко к непрерывному." });
  }
  mathBlock("m2", lines);

  let sm;
  if (S.src === "rec") {
    sm = `Участок ${txt(Tm, 3)} мс собран из ${r.N} гармоник с частотами, кратными 1/T = ${txt(1000 / Tm, 4)} Гц. Ошибка RMSE = ${txt(r.rmse_N, 3)}, сумма уже содержит ${txt(100 * r.bessel_N / r.energy, 3)}% энергии участка.`;
    if (r.period) sm += ` Основной тон ≈ ${txt(r.period.f0, 4)} Гц: если взять T = ${r.period.k} периодам (${txt(r.period.suggest_T * 1000, 4)} мс), стык станет плавнее.`;
  } else if (r.gibbs) {
    sm = `У ${S.src === "square" ? "меандра" : "пилы"} есть разрывы. Как бы много гармоник ни взять, у скачка остаётся «рожок» высотой около 9% скачка — он только становится уже.`;
  } else {
    sm = "|x| непрерывна, поэтому ряд сходится равномерно: коэффициенты убывают как 1/n², и уже несколько гармоник дают почти точную форму.";
  }
  $("#s2-summary").textContent = sm;
}

// ---------------------------------------------------------------- шаг 3
async function step3(my) {
  const r = await api(`/api/rec/${S.rid}/coef`, { t0: S.t0, T: S.T, n: S.n3, kind: S.kind, m: S.m, src: S.src });
  if (my !== seq) return;
  $("#n").max = $("#m").max = r.n_cap;
  $("#n-val").textContent = r.n; $("#m-val").textContent = r.m;
  const t = Array.from(f32(r.t), (v) => v * 1000), f = f32(r.f), bs = f32(r.basis), pr = f32(r.prod), bm = f32(r.basis_m);
  const fn = S.kind === "cos" ? "cos" : "sin";
  plot("p3-basis", [
    { x: t, y: f, mode: "lines", line: { color: C.gray, width: 2 }, name: "f(t)" },
    { x: t, y: bs, mode: "lines", line: { color: C.accent, width: 1.5 }, name: `${fn}(${r.n}ωt)`, yaxis: "y2" },
  ], { title: { text: "Сигнал и базисная функция" }, xaxis: { title: { text: "t, мс" } },
    yaxis: { title: { text: "f(t)" } }, yaxis2: { overlaying: "y", side: "right", range: [-1.1, 1.1], title: { text: fn } } });
  const posY = Array.from(pr, (v) => Math.max(v, 0)), negY = Array.from(pr, (v) => Math.min(v, 0));
  plot("p3-prod", [
    { x: t, y: posY, fill: "tozeroy", mode: "lines", line: { color: C.pos, width: 0.5 }, fillcolor: "rgba(46,139,87,.4)", name: "положительная площадь" },
    { x: t, y: negY, fill: "tozeroy", mode: "lines", line: { color: C.neg, width: 0.5 }, fillcolor: "rgba(192,57,43,.4)", name: "отрицательная площадь" },
    { x: t, y: pr, mode: "lines", line: { color: "#333", width: 1 }, name: `f(t)·${fn}(${r.n}ωt)` },
  ], { title: { text: "Произведение: коэффициент — это его средняя площадь" }, xaxis: { title: { text: "t, мс" } } });
  const ob = Array.from(bs, (v, i) => v * bm[i]);
  plot("p3-ortho", [
    { x: t, y: ob.map((v) => Math.max(v, 0)), fill: "tozeroy", mode: "none", fillcolor: "rgba(46,139,87,.4)", showlegend: false },
    { x: t, y: ob.map((v) => Math.min(v, 0)), fill: "tozeroy", mode: "none", fillcolor: "rgba(192,57,43,.4)", showlegend: false },
    { x: t, y: ob, mode: "lines", line: { color: "#333", width: 1 }, name: `${fn}(${r.n}ωt)·${fn}(${r.m}ωt)` },
  ], { title: { text: `Ортогональность: ${fn}(${r.n}ωt)·${fn}(${r.m}ωt)` }, xaxis: { title: { text: "t, мс" } } });

  const L = S.kind === "cos" ? "a" : "b", TT = num(r.T, 4);
  const lines = [
    `${L}_{${r.n}} = \\frac{2}{T}\\int_0^T f(t)\\${fn}(${r.n}\\omega t)\\,dt = \\frac{2}{${TT}\\ \\text{с}}\\left(${num(r.area_pos, 4)} ${sign(r.area_neg)} ${num(Math.abs(r.area_neg), 4)}\\right)\\text{с} = ${num(r.coef_trap, 5)}`,
    { html: `Зелёная площадь ${T(num(r.area_pos, 4))}, красная ${T(num(r.area_neg, 4))} (метод трапеций по ${r.M + 1} точкам, T = ${txt(r.T * 1000, 4)} мс, ω = ${txt(r.omega, 5)} рад/с).` },
    `\\text{через БПФ: } ${L}_{${r.n}} \\approx ${num(r.coef_fft, 5)},\\qquad \\text{разница } ${num(r.diff, 2)}`,
  ];
  if (r.exact !== undefined) lines.push(`\\text{точное значение: } ${L}_{${r.n}} = ${num(r.exact, 5)}`);
  if (r.boundary) lines.push({ html: `Трапеции используют и правый конец f(t₀+T) = ${txt(r.boundary.fT, 3)}, а БПФ считает сигнал периодическим и вместо него берёт f(t₀) = ${txt(r.boundary.f0, 3)}. Отсюда и разница.` });
  lines.push(`c_{${r.n}} = \\frac{a_{${r.n}} - i\\,b_{${r.n}}}{2} = ${num(r.c_trap.re, 4)} ${sign(r.c_trap.im)} ${num(Math.abs(r.c_trap.im), 4)}\\,i,\\quad |c_{${r.n}}| = ${num(r.c_trap.abs, 4)}`);
  lines.push(`\\int_0^T \\${fn}(${r.n}\\omega t)\\${fn}(${r.m}\\omega t)\\,dt = ${num(r.ortho.same, 3)}\\ \\text{с}\\quad(\\text{теория: } ${num(r.ortho.expected_same, 4)})`);
  const other = S.kind === "cos" ? "sin" : "cos";
  lines.push(`\\int_0^T \\${fn}(${r.n}\\omega t)\\${other}(${r.m}\\omega t)\\,dt = ${num(r.ortho.cross, 3)}\\ \\text{с}\\quad(\\text{теория: } 0)`);
  mathBlock("m3", lines);
  $("#s3-summary").textContent = `Коэффициент показывает, насколько сигнал «похож» на ${fn}(${r.n}ωt): перемножаем, считаем площадь и делим на T/2. ` +
    `Здесь зелёная площадь ${Math.abs(r.area_pos) > Math.abs(r.area_neg) ? "больше" : "меньше"} красной, поэтому ${L}${r.n} ${r.coef_trap >= 0 ? "положителен" : "отрицателен"}. ` +
    `Разные гармоники ортогональны: площадь их произведения равна нулю, поэтому каждая «ловит» только свой коэффициент.`;
}

// ---------------------------------------------------------------- шаг 4
async function step4(my) {
  if (!S.spec || S.spec.window !== S.win) {
    const r = await api(`/api/rec/${S.rid}/spectrum`, { window: S.win });
    if (my !== seq) return;
    S.spec = r; S.spec.ampArr = f32(r.amp);
    if (S.ops.length) await applyOps(false);
  }
  renderSpectrum();
}

async function applyOps(render = true) {
  if (!S.ops.length) { S.edited = null; $("#edit-audio").removeAttribute("src"); if (render) renderSpectrum(); return; }
  const r = await post(`/api/rec/${S.rid}/edit`, { ops: S.ops, window: S.win }, true);
  S.edited = r; S.edited.ampArr = f32(r.amp);
  $("#edit-audio").src = r.audio_url;
  if (render) renderSpectrum();
}

function opText(o) {
  return { keep_band: `оставить ${txt(o.f_lo, 4)}–${txt(o.f_hi, 4)} Гц`, remove_band: `вырезать ${txt(o.f_lo, 4)}–${txt(o.f_hi, 4)} Гц`,
    lowpass: `НЧ до ${txt(o.f_c, 4)} Гц`, highpass: `ВЧ от ${txt(o.f_c, 4)} Гц` }[o.type];
}

function band(f) {
  const ny = S.sr / 2;
  return { f_lo: Math.max(0, f - S.bw), f_hi: Math.min(ny, f + S.bw) };
}

function listenBand(f) {
  const b = band(f);
  const a = $("#edit-audio");
  a.src = `/api/rec/${S.rid}/band.wav?` + new URLSearchParams(b);
  a.play();
  $("#ops-list").textContent = `Слушаем только ${txt(b.f_lo, 4)}–${txt(b.f_hi, 4)} Гц. ` + (S.ops.length ? "Изменения: " + S.ops.map(opText).join("; ") : "Изменений нет.");
}

function renderSpectrum() {
  const r = S.spec;
  if (!r) return;
  const fmax = S.fmax > 0 ? Math.min(S.fmax, r.fs / 2) : r.fs / 2;
  const kmax = Math.min(r.ampArr.length, Math.floor(fmax / r.df) + 1);
  const fx = range(kmax, (k) => k * r.df);
  let ref = 1e-12;
  for (let k = 1; k < r.ampArr.length; k++) if (r.ampArr[k] > ref) ref = r.ampArr[k];
  const conv = (a) => (S.scale === "db" ? Array.from(a.subarray(0, kmax), (v) => 20 * Math.log10(Math.max(v, 1e-12) / ref)) : Array.from(a.subarray(0, kmax)));
  const data = [{ x: fx, y: conv(r.ampArr), mode: "lines", line: { color: S.edited ? C.gray : C.accent, width: 1 }, name: "исходный" }];
  if (S.edited) data.push({ x: fx, y: conv(S.edited.ampArr), mode: "lines", line: { color: C.accent, width: 1 }, name: "после изменений" });
  const peaks = r.peaks.filter((p) => p.f <= fmax);
  const yp = (p) => (S.scale === "db" ? 20 * Math.log10(p.amp / ref) : p.amp);
  data.push({ x: peaks.map((p) => p.f), y: peaks.map(yp), mode: "markers+text", text: peaks.map((p) => `${txt(p.f, 4)} Гц`),
    textposition: "top center", marker: { color: C.orange, size: 8 }, name: "пики", hovertemplate: "%{x:.1f} Гц<extra></extra>" });
  plot("p4-spec", data, {
    title: { text: `Амплитудный спектр, окно ${S.win === "hann" ? "Ханна" : "прямоугольное"}` },
    xaxis: { title: { text: "f, Гц" }, range: [0, fmax] },
    yaxis: S.scale === "db" ? { title: { text: "дБ от максимума" }, range: [-100, 8] } : { title: { text: "амплитуда A" } },
  });
  const p = $("#p4-spec");
  p.removeAllListeners?.("plotly_click");
  p.on("plotly_click", (ev) => {
    let f = ev.points[0].x;
    const near = r.peaks.reduce((b, q) => (Math.abs(q.f - f) < Math.abs(b.f - f) ? q : b), r.peaks[0] || { f });
    if (near && Math.abs(near.f - f) < Math.max(3 * S.bw, 0.02 * fmax)) f = near.f;
    listenBand(f);
  });

  const box = $("#peaks");
  box.innerHTML = r.peaks.length ? "" : '<span class="muted">Выраженных пиков нет: спектр сплошной, похоже на шум.</span>';
  for (const q of r.peaks) {
    const el = document.createElement("span");
    el.className = "peak";
    el.innerHTML = `${txt(q.f, 4)} Гц <span class="muted">(${txt(q.db, 3)} дБ)</span>`;
    const play = Object.assign(document.createElement("button"), { type: "button", textContent: "▶", title: "Послушать только эту компоненту" });
    play.onclick = () => listenBand(q.f);
    const del = Object.assign(document.createElement("button"), { type: "button", textContent: "✕", title: "Удалить компоненту" });
    del.onclick = () => { S.ops.push({ type: "remove_band", ...band(q.f) }); applyOps().catch(showError); };
    el.append(play, del);
    box.appendChild(el);
  }
  $("#ops-list").textContent = S.ops.length ? "Изменения: " + S.ops.map(opText).join("; ") + (S.edited ? `. Осталось ${pct(S.edited.kept_energy)} энергии.` : "") : "Изменений нет.";

  const top = r.top, pv = r.parseval, nv = r.naive;
  const lines = [
    `X[k] = \\sum_{n=0}^{N-1} x[n]\\,e^{-2\\pi i k n/N},\\qquad N = ${num(r.N, 7)},\\ f_s = ${num(r.fs, 5)}\\ \\text{Гц}`,
    `\\Delta f = \\frac{f_s}{N} = \\frac{${num(r.fs, 5)}}{${num(r.N, 7)}} = ${num(r.df, 4)}\\ \\text{Гц}\\quad(= 1/T,\\ T = ${num(r.N / r.fs, 4)}\\ \\text{с})`,
    `\\text{самый сильный пик: } k = ${num(top.k, 7)},\\ f_k = k\\cdot\\frac{f_s}{N} = ${num(top.f, 5)}\\ \\text{Гц},\\ |X[k]| = ${num(top.absX, 4)}`,
    `A \\approx \\frac{2|X[k]|}{\\sum_n w[n]} = \\frac{2\\cdot ${num(top.absX, 4)}}{${num(top.sum_w, 5)}} = ${num(top.amp, 4)}`,
    `\\sum_n x[n]^2 = ${num(pv.time, 7)},\\quad \\frac1N\\sum_k |X[k]|^2 = ${num(pv.freq, 7)},\\quad \\text{отн. разница } ${num(pv.rel_error, 2)}`,
    `\\text{проверка на } N=${nv.N}:\\ \\max_k |X_{\\text{наивн}}[k] - X_{\\text{numpy}}[k]| = ${num(nv.max_abs_err, 2)}\\quad(\\max|X| = ${num(nv.max_abs_X, 3)})`,
    { html: `Время на N = ${nv.N}: прямая сумма ${txt(nv.t_naive * 1000, 3)} мс, БПФ ${txt(nv.t_fft * 1000, 3)} мс. Для всей записи: N² ≈ ${txt(r.ops_naive, 3)} операций против N·log₂N ≈ ${txt(r.ops_fft, 3)}.` },
  ];
  if (S.edited) lines.push(`\\text{после изменений: } \\frac{\\sum y[n]^2}{\\sum x[n]^2} = ${num(S.edited.kept_energy, 4)}`);
  mathBlock("m4", lines);
  const pk = r.peaks.slice().sort((a, b) => b.amp - a.amp).slice(0, 3).map((q) => `${txt(q.f, 4)} Гц`);
  $("#s4-summary").textContent = (pk.length ? `Самые сильные частоты: ${pk.join(", ")}. ` : "Явных пиков нет — энергия распределена по многим частотам. ") +
    `Сетка частот идёт с шагом ${txt(r.df, 3)} Гц, потому что запись длится ${txt(r.N / r.fs, 3)} с: чем дольше слушаем, тем точнее различаем частоты. ` +
    `Окно ${S.win === "hann" ? "Ханна уменьшает «растекание» пиков" : "прямоугольное даёт узкие, но «растекающиеся» пики"}.`;
}

// ---------------------------------------------------------------- шаг 5
let s5 = null;
async function step5(my) {
  if (S.t5 === null) S.t5 = S.t0;
  const r = await api(`/api/rec/${S.rid}/spectrogram`, {
    mode: S.mode5, win: S.win5, overlap: S.overlap, window: S.wtype, mels: S.mels, fmax: S.fmax5, t: S.t5,
  });
  if (my !== seq) return;
  s5 = r;
  r.timesArr = f32(r.times); r.freqArr = f32(r.freqs); r.melC = f32(r.mel_centers);
  r.specArr = f32(r.spec_db); r.melArr = f32(r.mel_db);
  $("#custom-controls").hidden = r.mode === "ast";
  $("#win-ms").textContent = `(${txt(1000 * S.win5 / S.sr, 3)} мс)`;
  $("#mels-val").textContent = S.mels;
  renderSpectrogram();
}

function renderSpectrogram() {
  const r = s5;
  if (!r) return;
  const mel = S.fscale === "mel", M = r.M, K = mel ? r.n_mels : r.K, src = mel ? r.melArr : r.specArr;
  const stride = mel ? r.n_mels : r.K;
  let zmax = -Infinity;
  for (let i = 0; i < src.length; i++) if (src[i] > zmax) zmax = src[i];
  // z[k][m]: строки — частоты, столбцы — кадры; дБ относительно максимума
  const z = range(K, (k) => {
    const row = new Float32Array(M);
    for (let m = 0; m < M; m++) row[m] = src[m * stride + k] - zmax;
    return row;
  });
  const y = mel ? range(K, (j) => j) : Array.from(r.freqArr);
  const shapes = [{ type: "line", x0: r.timesArr[r.example.m], x1: r.timesArr[r.example.m], yref: "paper", y0: 0, y1: 1, line: { color: "#fff", width: 2, dash: "dot" } }];
  let xrange;
  if (r.ast && mel) {
    // сетка патчей AST: 16×16, шаг 10 (по мел-полосам и кадрам); кадры идут через 10 мс
    const dt = r.hop / r.sr;
    const t0 = r.timesArr[0];
    for (let i = 0; i < r.ast.n_patches_t; i += 1) {
      const x = t0 + i * r.ast.stride * dt;
      shapes.push({ type: "line", x0: x, x1: x, y0: -0.5, y1: K - 0.5, line: { color: "rgba(255,255,255,.25)", width: 1 } });
    }
    for (let i = 0; i < r.ast.n_patches_f; i++) {
      const yy = i * r.ast.stride - 0.5;
      shapes.push({ type: "line", x0: t0, x1: t0 + r.ast.max_frames * dt, y0: yy, y1: yy, line: { color: "rgba(255,255,255,.25)", width: 1 } });
    }
    const end = t0 + r.ast.max_frames * dt;
    if (r.ast.padded > 0) shapes.push({ type: "rect", x0: r.timesArr[M - 1], x1: end, y0: -0.5, y1: K - 0.5, fillcolor: "rgba(120,120,120,.35)", line: { width: 0 } });
    xrange = [0, end];
  }
  plot("p5-spec", [{
    type: "heatmap", x: Array.from(r.timesArr), y, z, colorscale: "Viridis", zmin: -90, zmax: 0,
    colorbar: { title: { text: "дБ" } }, hovertemplate: mel ? "t=%{x:.3f} с<br>полоса %{y}<br>%{z:.1f} дБ<extra></extra>" : "t=%{x:.3f} с<br>f=%{y:.0f} Гц<br>%{z:.1f} дБ<extra></extra>",
  }], {
    title: { text: mel ? `Мел-спектрограмма, ${r.n_mels} полос${r.ast ? " · сетка патчей AST 16×16, шаг 10" : ""}` : "Спектрограмма" },
    xaxis: { title: { text: "t, с" }, range: xrange }, yaxis: { title: { text: mel ? "номер мел-полосы" : "f, Гц" } }, shapes,
  });
  const p = $("#p5-spec");
  p.removeAllListeners?.("plotly_click");
  p.on("plotly_click", (ev) => { S.t5 = ev.points[0].x; refresh(); });

  // срез в момент t_m
  const m = r.example.m;
  const col = range(K, (k) => src[m * stride + k] - zmax);
  plot("p5-slice", [{ x: mel ? Array.from(r.melC) : Array.from(r.freqArr), y: col, mode: mel ? "lines+markers" : "lines", line: { color: C.accent, width: 1 }, marker: { size: 4 } }], {
    title: { text: `Срез при t = ${txt(r.example.t, 4)} с (кадр m = ${m})` }, showlegend: false,
    xaxis: { title: { text: mel ? "центр мел-полосы, Гц" : "f, Гц" } }, yaxis: { title: { text: "дБ от максимума" }, range: [-100, 5] },
  });

  const e = r.example, g = r.gabor;
  const lines = [
    `X(m,k) = \\sum_{n=0}^{N-1} x[n + mH]\\,w[n]\\,e^{-2\\pi i k n/N},\\quad N = ${r.n_win}\\ (${num(r.win_ms, 4)}\\ \\text{мс}),\\ H = ${r.hop}\\ (${num(r.hop_ms, 4)}\\ \\text{мс}),\\ f_s = ${num(r.sr, 5)}\\ \\text{Гц}`,
    `\\Delta f = \\frac{f_s}{N_{\\text{ДПФ}}} = \\frac{${num(r.sr, 5)}}{${r.n_fft}} = ${num(r.df, 4)}\\ \\text{Гц},\\qquad \\text{кадров } M = ${r.M},\\ \\text{перекрытие } ${num(100 * r.overlap, 3)}\\%`,
    `m = ${e.m}:\\ \\text{кадр с отсчёта } mH = ${e.start},\\ \\text{пик } k = ${e.k},\\ f_k = ${num(e.f, 5)}\\ \\text{Гц}`,
    `S(m,k) = |X(m,k)|^2 = ${num(e.S, 4)},\\quad 10\\log_{10}(S + \\varepsilon) = ${num(e.S_db, 4)}\\ \\text{дБ}\\quad(\\varepsilon = ${num(r.eps, 1)})`,
    `m(f_k) = 2595\\log_{10}\\!\\left(1 + \\frac{${num(e.f, 5)}}{700}\\right) = ${num(e.mel_of_f, 5)}\\ \\text{мел} \\Rightarrow \\text{полоса } j = ${e.band}\\ (\\text{центр } ${num(e.band_center, 4)}\\ \\text{Гц, вес } ${num(e.band_weight, 3)})`,
    `\\sigma_t = ${num(1000 * g.sigma_t, 4)}\\ \\text{мс},\\ \\sigma_f = ${num(g.sigma_f, 4)}\\ \\text{Гц}:\\quad \\sigma_t\\sigma_f = ${num(g.product, 4)} \\;\\ge\\; \\frac{1}{4\\pi} = ${num(g.bound, 4)}`,
  ];
  if (r.window === "rect") lines.push({ html: "У прямоугольного окна в непрерывном случае σ<sub>f</sub> = ∞ (|W(f)|² убывает как 1/f²). Здесь число конечно только потому, что частоты ограничены ±fₛ/2." });
  if (r.ast) lines.push({ html: `Вход AST: 1024 кадра по 10 мс → патчи 16×16 с шагом 10: ${r.ast.n_patches_f} × ${r.ast.n_patches_t} = ${r.ast.n_patches} патчей. ` +
    (r.ast.truncated ? "Запись длиннее 10,24 с: лишние кадры отрезаются." : `Здесь ${r.ast.frames_used} кадров записи, остальные ${r.ast.padded} — нули (серая зона).`) });
  mathBlock("m5", lines);

  const tonal = r.flatness < 0.1;
  $("#s5-summary").textContent = `Спектрограмма — это ДПФ шага 4, посчитанное в коротких окнах по ${txt(r.win_ms, 3)} мс через каждые ${txt(r.hop_ms, 3)} мс. ` +
    `По вертикали частота, по горизонтали время, цвет — громкость. В выбранный момент самая сильная частота ${txt(e.f, 4)} Гц, спектральная плоскость ${txt(r.flatness, 2)} (1 — белый шум, около 0 — отдельные тоны)${tonal ? ": спектр с выраженными пиками" : ""}. ` +
    `Длинное окно точнее по частоте, короткое — по времени: σₜ·σ_f = ${txt(g.product, 3)}, а меньше 1/(4π) ≈ 0,0796 не бывает. Именно такая картинка (мел, лог) идёт на вход нейросети.`;
}

// ---------------------------------------------------------------- обработчики
function segButtons(id, attr, key, after) {
  $$(`#${id} button`).forEach((b) => (b.onclick = () => {
    S[key] = b.dataset[attr];
    $$(`#${id} button`).forEach((x) => x.classList.toggle("active", x === b));
    after();
  }));
}

window.addEventListener("DOMContentLoaded", () => {
  $$(".tex").forEach((e) => tex(e, e.textContent, false));
  $$(".tex-d").forEach((e) => tex(e, e.textContent, true));

  $$(".rec").forEach((b) => (b.onclick = () => record(+b.dataset.sec, b)));
  $("#file").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f); fd.append("source", "upload");
    try { await loadRecording(await post("/api/upload", fd)); } catch (err) { showError(err); }
    e.target.value = "";
  };
  $$(".sample").forEach((b) => (b.onclick = async () => {
    $$(".sample").forEach((x) => x.classList.toggle("active", x === b));
    const fd = new FormData();
    fd.append("id", b.dataset.id);
    try { await loadRecording(await post("/api/sample", fd)); } catch (err) { showError(err); }
  }));
  $$("#stepper button").forEach((b) => (b.onclick = () => goStep(+b.dataset.step)));

  // шаг 1
  $("#aa").onchange = (e) => { S.aa = e.target.checked; refresh(); };
  $("#play-orig").onclick = () => { $("#audio").currentTime = 0; $("#audio").play(); };
  $("#play-fs").onclick = () => {
    const a = $("#edit-audio");
    a.src = `/api/rec/${S.rid}/step1.wav?fs=${S.fs}&aa=${S.aa ? 1 : 0}`;
    a.play();
  };

  // шаг 2–3
  const Tval = () => ($("#T-val").textContent = txt(S.T, 3));
  $("#T").oninput = (e) => { S.T = +e.target.value; Tval(); setT0(S.t0); refreshSoon(); };
  Tval();
  $("#N").oninput = (e) => { S.N = +e.target.value; $("#N-val").textContent = S.N; refreshSoon(); };
  $("#N-val").textContent = S.N;
  $("#show-harm").onchange = () => refresh();
  segButtons("src-options", "src", "src", refresh);
  $("#fit-period").onclick = () => {
    if (!S.period) return;
    S.T = Math.round(S.period.suggest_T * 1e5) / 100;
    $("#T").value = S.T; Tval(); setT0(S.t0); refresh();
  };
  $("#n").oninput = (e) => { S.n3 = +e.target.value; $("#n-val").textContent = S.n3; refreshSoon(); };
  $("#m").oninput = (e) => { S.m = +e.target.value; $("#m-val").textContent = S.m; refreshSoon(); };
  segButtons("kind-options", "kind", "kind", refresh);

  // шаг 4
  segButtons("win-options", "win", "win", refresh);
  segButtons("scale-options", "scale", "scale", renderSpectrum);
  $("#fmax").onchange = (e) => { S.fmax = +e.target.value; renderSpectrum(); };
  $("#bw").oninput = (e) => { S.bw = +e.target.value; $("#bw-val").textContent = S.bw; };
  $("#bw-val").textContent = S.bw;
  const addOp = (op) => { S.ops.push(op); applyOps().catch((e) => { S.ops.pop(); showError(e); }); };
  $("#apply-lp").onclick = () => addOp({ type: "lowpass", f_c: +$("#lp").value });
  $("#apply-hp").onclick = () => addOp({ type: "highpass", f_c: +$("#hp").value });
  $("#apply-cut").onclick = () => addOp({ type: "remove_band", f_lo: +$("#cut-lo").value, f_hi: +$("#cut-hi").value });
  $("#reset-ops").onclick = () => { S.ops = []; applyOps().catch(showError); };

  // шаг 5
  segButtons("mode-options", "mode", "mode5", () => { if (S.mode5 === "ast") { S.fscale = "mel"; $$("#fscale-options button").forEach((b) => b.classList.toggle("active", b.dataset.fscale === "mel")); } refresh(); });
  segButtons("fscale-options", "fscale", "fscale", renderSpectrogram);
  segButtons("wtype-options", "wtype", "wtype", refresh);
  $("#win").onchange = (e) => { S.win5 = +e.target.value; refresh(); };
  $("#overlap").onchange = (e) => { S.overlap = +e.target.value; refresh(); };
  $("#fmax5").onchange = (e) => { S.fmax5 = +e.target.value; refresh(); };
  $("#mels").oninput = (e) => { S.mels = +e.target.value; $("#mels-val").textContent = S.mels; refreshSoon(); };
  $("#mels-val").textContent = S.mels;

  // презентационный режим
  $("#btn-present").onclick = (e) => {
    const on = document.body.classList.toggle("present");
    e.target.textContent = on ? "Обычный режим" : "Показать презентационный режим";
    setTimeout(() => { $$(".plot, .mini").forEach((p) => p.data && Plotly.Plots.resize(p)); refresh(); }, 50);
  };
});
