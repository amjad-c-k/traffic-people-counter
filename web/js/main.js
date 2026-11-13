// Page controller: wires the controls to the pipeline  detect -> track -> count -> draw.
import { Detector, CLASSES } from "./detector.js";
import { ByteTracker } from "./tracker.js";
import { LineCounter, center } from "./lineCounter.js";
import { COLORS, drawFrame, drawPreview, handleRadius } from "./draw.js";

// Replaced at build time (scripts/build_web.py) with a short code that changes on every
// deploy; adding it to URLs stops browsers from mixing old cached files with new ones.
const VERSION = "__VERSION__";
// The model gets its own code (hash of the model file only), so a code-only update doesn't
// force visitors to download the 10 MB model again.
const MODEL_URL = "model/yolo26n.onnx?v=__MODEL_VERSION__";
const SAMPLE_URL = "samples/highway.mp4";
// Browsers don't tell us a video's frame rate, so we step through every video at a fixed
// 25 frames per second of video time (the sample video's own rate).
const STEP_FPS = 25;
const MAX_WIDTH = 1280; // bigger videos are drawn/processed at this width (YOLO resizes to 640 anyway)

const BACKEND_NAMES = { webnn: "WebNN", webgpu: "WebGPU", wasm: "CPU" };

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("view"), ctx = canvas.getContext("2d");
const params = new URLSearchParams(location.search);
// rect: dynamic-shape model, 16:9 video -> 640x384 input. ?rect=0 for fixed 640x640 models.
// ?imgsz= is for speed tests only: 480 was NOT faster on WebGPU (measured) and misses small objects.
const detector = new Detector({ rect: params.get("rect") !== "0", imgsz: Number(params.get("imgsz")) || 640 });
let busy = false, stopRequested = false, lastEvents = [], videoName = "highway";

// --- small UI helpers ------------------------------------------------------------------------
function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
  console.log(`[status] ${text}`); // also in the browser console, handy for debugging
}

const selectedClasses = () => [...document.querySelectorAll("#classChecks input:checked")].map((el) => Number(el.value));

// The counting line, stored as fractions of the frame (0..1) so it fits any video size.
// Default: horizontal across the middle, the same as the Python app.
const DEFAULT_LINE = { a: [0, 0.5], b: [1, 0.5] };
let line = structuredClone(DEFAULT_LINE);

function linePoints() { // line in canvas pixels
  const w = canvas.width, h = canvas.height;
  return [[Math.trunc(line.a[0] * w), Math.trunc(line.a[1] * h)], [Math.trunc(line.b[0] * w), Math.trunc(line.b[1] * h)]];
}

function updateButtons(ready = !!video.src) {
  $("startBtn").disabled = busy || !ready || !detector.session;
  $("stopBtn").disabled = !busy;
  $("sampleBtn").disabled = busy;
  $("fileInput").disabled = busy;
  $("resetLineBtn").disabled = busy;
  $("csvBtn").disabled = busy || lastEvents.length === 0;
  // The line can only be dragged before/after a run, not while counting.
  $("canvasHint").hidden = busy;
  if (busy) canvas.className = "";
  else if (video.src) canvas.className = "editable";
}

function renderCounts(counter) {
  const rows = selectedClasses().map((id) => {
    const name = CLASSES[id];
    const n = (dir) => (counter ? counter.counts.get(`${name}|${dir}`) || 0 : 0);
    return `<tr><td><span class="swatch" style="background:${COLORS[name]}"></span>${name}</td><td>${n("IN")}</td><td>${n("OUT")}</td></tr>`;
  });
  const { IN, OUT } = counter ? counter.totals() : { IN: 0, OUT: 0 };
  rows.push(`<tr class="total"><td>Total</td><td>${IN}</td><td>${OUT}</td></tr>`);
  $("countsBody").innerHTML = rows.join("");
}

function seek(t) { // wait until the video shows the frame at time t (seeking is asynchronous)
  return new Promise((resolve) => {
    video.addEventListener("seeked", resolve, { once: true });
    video.currentTime = t;
  });
}

async function showPreview(active = null) {
  if (!video.src || busy) return;
  if (video.currentTime === 0) await seek(0.5 / STEP_FPS);
  drawPreview(ctx, video, ...linePoints(), active);
}

// --- dragging the counting line on the video -------------------------------------------------
// Pointer events work for mouse, pen and touch. The canvas is shown scaled down, so pointer
// positions are converted from screen pixels to canvas pixels first.
function toCanvas(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
}

function handleAt([x, y]) { // which handle ("a", "b", "mid") is under the pointer, if any
  const [a, b] = linePoints();
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const grab = handleRadius(canvas) * 2.2; // generous hit area, easier with a finger
  for (const [name, [hx, hy]] of [["a", a], ["b", b], ["mid", mid]]) {
    if (Math.hypot(x - hx, y - hy) <= grab) return name;
  }
  return null;
}

function wireLineDragging() {
  let dragging = null, last = null;
  const clamp = (v) => Math.min(1, Math.max(0, v));
  canvas.addEventListener("pointerdown", (e) => {
    if (busy || !video.src) return;
    dragging = handleAt(toCanvas(e));
    if (!dragging) return;
    last = toCanvas(e);
    try { canvas.setPointerCapture(e.pointerId); } catch { /* keeps moves coming outside the canvas; optional */ }
    canvas.className = "grabbing";
  });
  canvas.addEventListener("pointermove", (e) => {
    if (busy || !video.src) return;
    const p = toCanvas(e);
    if (!dragging) { // just hovering: show which handle would be grabbed
      const h = handleAt(p);
      canvas.className = h ? "grab" : "editable";
      showPreview(h);
      return;
    }
    const dx = (p[0] - last[0]) / canvas.width, dy = (p[1] - last[1]) / canvas.height;
    last = p;
    for (const end of dragging === "mid" ? ["a", "b"] : [dragging]) {
      line[end] = [clamp(line[end][0] + dx), clamp(line[end][1] + dy)];
    }
    showPreview(dragging);
  });
  const stop = () => { if (dragging) { dragging = null; canvas.className = "editable"; showPreview(); } };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
}

// --- loading ---------------------------------------------------------------------------------
// Download the model ourselves (instead of letting ONNX Runtime do it) to show a progress bar.
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed (${res.status}) for ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  return bytes;
}

async function loadModel() {
  const mb = (n) => (n / 1e6).toFixed(1);
  const bytes = await fetchWithProgress(MODEL_URL, (loaded, total) => {
    $("progress").value = total ? loaded / total : 0;
    setStatus(`Downloading model: ${mb(loaded)}${total ? ` / ${mb(total)}` : ""} MB (only on the first visit)…`);
  });
  // The engine's own ~15 MB download can't report progress, so show a running timer instead:
  // on a slow connection this step can take a minute, and a frozen message looks broken.
  const t0 = performance.now();
  const msg = () => `Starting the AI engine (first visit downloads ~15 MB)… ${Math.round((performance.now() - t0) / 1000)} s`;
  setStatus(msg());
  const timer = setInterval(() => { $("status").textContent = msg(); }, 1000);
  const backend = await detector.load(bytes, params.get("ep") || "auto").finally(() => clearInterval(timer));
  $("progress").value = 0;
  return backend;
}

// Load a video (sample URL or uploaded File) into memory and show its first frame.
// Blob videos can always be seeked frame by frame, whatever the web server supports.
async function loadVideo(source, name) {
  const blob = source instanceof Blob ? source : await (await fetch(source)).blob();
  if (video.src) URL.revokeObjectURL(video.src);
  video.src = URL.createObjectURL(blob);
  await new Promise((resolve, reject) => {
    video.addEventListener("loadeddata", resolve, { once: true });
    video.addEventListener("error", () => reject(new Error("This browser can't play that video format. Try an MP4 (H.264) file.")), { once: true });
  });
  const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  videoName = name.replace(/\.[^.]+$/, "");
  lastEvents = [];
  renderCounts(null);
  const secs = video.duration;
  $("videoInfo").textContent = `${name} · ${video.videoWidth}×${video.videoHeight} · ${secs.toFixed(1)} s`;
  await seek(0.5 / STEP_FPS);
  await showPreview();
  canvas.className = "editable";
  updateButtons(true);
}

// --- processing ------------------------------------------------------------------------------
// Step through the video (seek, detect, track, count, draw). Seeking instead of normal playback
// means no step is skipped however slow the device is, so tracking always sees every frame.
async function processVideo({ classIds = Object.keys(CLASSES).map(Number), line = linePoints(),
                              step = 1, maxFrames = Infinity, onFrame = null } = {}) {
  const w = canvas.width, h = canvas.height;
  const total = Math.min(Math.floor((video.duration * STEP_FPS) / step), maxFrames);
  const tracker = new ByteTracker();
  const counter = new LineCounter(...line);
  const trails = new Map(); // trackId -> last 20 centre points
  const t0 = performance.now();
  const recent = []; // timestamps + model times of the last frames, for the live speed display
  let n = 0;
  for (; n < total && !stopRequested; n++) {
    const frame = n * step; // frame number in 25-fps units (step > 1 only in experiments)
    await seek((frame + 0.5) / STEP_FPS); // +0.5: land in the middle of the frame, not on its edge
    const dets = await detector.detect(video, w, h, { classIds });
    const tracks = tracker.update(dets, [w, h]).map((t) => ({ ...t, className: CLASSES[t.classId] }));
    counter.update(tracks, frame, STEP_FPS);
    for (const t of tracks) {
      const pts = trails.get(t.trackId) || [];
      pts.push(center(t.box));
      trails.set(t.trackId, pts.slice(-20));
    }
    drawFrame(ctx, video, tracks, counter, trails);
    if (onFrame) onFrame(n, tracks);
    renderCounts(counter);

    // Speed over the last 10 frames (not since the start), so it reflects the current speed.
    const now = performance.now();
    recent.push({ now, model: detector.timings.model });
    if (recent.length > 11) recent.shift();
    const fps = recent.length > 1 ? (recent.length - 1) / ((now - recent[0].now) / 1000) : 0;
    const modelMs = recent.reduce((s, r) => s + r.model, 0) / recent.length;
    $("progress").value = (n + 1) / total;
    const eta = n >= 5 ? ` · about ${Math.ceil((total - n - 1) / fps)} s left` : "";
    setStatus(`Frame ${n + 1}/${total} · ${fps.toFixed(1)} FPS (${BACKEND_NAMES[detector.ep]}, model ${modelMs.toFixed(0)} ms/frame)${eta}`);
  }
  return { counter, seconds: (performance.now() - t0) / 1000, frames: n };
}

async function start() {
  if (selectedClasses().length === 0) return setStatus("Select at least one class to count.", true);
  busy = true;
  stopRequested = false;
  updateButtons();
  try {
    // Every frame is processed: skipping frames (tested with step = 2) broke tracking, because
    // ByteTrack needs a car's box to overlap itself between consecutive frames (9 -> 0 counts).
    const { counter, seconds, frames } = await processVideo({ classIds: selectedClasses() });
    lastEvents = counter.events;
    const { IN, OUT } = counter.totals();
    setStatus(`${stopRequested ? "Stopped" : "Done"}: ${frames} frames in ${seconds.toFixed(0)} s · IN ${IN} · OUT ${OUT}`);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, true);
  } finally {
    busy = false;
    updateButtons();
  }
}

// Same columns as the Python app's CSV: one row per crossing event.
function downloadCsv() {
  const rows = [["track_id", "class", "direction", "frame", "time_s"],
                ...lastEvents.map((e) => [e.trackId, e.className, e.direction, e.frame, e.timeS])];
  const blob = new Blob([rows.map((r) => r.join(",")).join("\n") + "\n"], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `${videoName}_crossings.csv` });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --- developer checks (used with the local test server, see scripts/check_web.py) -------------
const report = (data) => fetch("/report", { method: "POST", body: JSON.stringify(data) });

async function checkDetections(frameList) { // ?check=0,60,120: single-frame detections vs Python
  const results = {};
  let seekMs = 0, detectMs = 0;
  const parts = { pre: 0, model: 0, post: 0 };
  await seek(0.5 / STEP_FPS); // warm-up (not timed): e.g. WebGPU compiles its GPU programs now
  await detector.detect(video, canvas.width, canvas.height);
  for (const n of frameList) {
    const t0 = performance.now();
    await seek((n + 0.5) / STEP_FPS);
    const t1 = performance.now();
    results[n] = await detector.detect(video, canvas.width, canvas.height);
    seekMs += t1 - t0;
    detectMs += performance.now() - t1;
    for (const p in parts) parts[p] += detector.timings[p];
  }
  const k = frameList.length;
  for (const p in parts) parts[p] = Math.round((parts[p] / k) * 10) / 10;
  await report({
    results, msPerFrame: (seekMs + detectMs) / k, seekMs: seekMs / k, detectMs: detectMs / k, parts,
    crossOriginIsolated: self.crossOriginIsolated, wasmThreads: ort.env.wasm.numThreads, ep: detector.ep,
    userAgent: navigator.userAgent,
  });
}

async function runDevMode() {
  if (params.get("check")) return checkDetections(params.get("check").split(",").map(Number));
  if (params.get("track")) { // ?track=1: whole pipeline with default settings, events vs Python
    const ids = [];
    const { counter, seconds, frames } = await processVideo({ onFrame: (n, tracks) => ids.push(tracks.map((t) => t.trackId)) });
    return report({ events: counter.events, totals: counter.totals(), ids, seconds, frames, ep: detector.ep });
  }
  if (params.get("uitest")) { // ?uitest=1: drive the real controls like a visitor would
    // Drag the left end down to 40% height and the right end to 60%, with pointer events.
    const r = canvas.getBoundingClientRect();
    const screen = ([x, y]) => [r.left + (x / canvas.width) * r.width, r.top + (y / canvas.height) * r.height];
    const drag = (from, to) => {
      const [fx, fy] = screen(from), [tx, ty] = screen(to);
      const ev = (type, x, y) => canvas.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: y, pointerId: 1, bubbles: true }));
      ev("pointerdown", fx, fy);
      for (let i = 1; i <= 5; i++) ev("pointermove", fx + ((tx - fx) * i) / 5, fy + ((ty - fy) * i) / 5);
      ev("pointerup", tx, ty);
    };
    const [a, b] = linePoints();
    drag(a, [a[0], 0.4 * canvas.height]);
    drag(b, [b[0], 0.6 * canvas.height]);
    await new Promise((res) => setTimeout(res, 1500)); // let the preview redraw
    const preview = canvas.toDataURL("image/png");
    const lineAfterDrag = structuredClone(line);
    $("startBtn").click();
    await new Promise((r) => setTimeout(r, 500));
    while (busy) await new Promise((r) => setTimeout(r, 500));
    return report({
      preview, final: canvas.toDataURL("image/png"), status: $("status").textContent,
      table: [...document.querySelectorAll("#countsBody tr")].map((tr) => tr.innerText.replace(/\s+/g, " ").trim()),
      events: lastEvents.length, csvDisabled: $("csvBtn").disabled, lineAfterDrag,
    });
  }
  if (params.get("layout")) { // ?layout=1: does the page fit the window? (no sideways scrolling)
    return report({ innerWidth, scrollWidth: document.documentElement.scrollWidth, status: $("status").textContent });
  }
  if (params.get("snapshot")) { // ?snapshot=N: process N frames, send back a PNG of the page canvas
    await processVideo({ maxFrames: Number(params.get("snapshot")) });
    return report({ snapshot: canvas.toDataURL("image/png"), status: $("status").textContent });
  }
  return false;
}

// --- start-up --------------------------------------------------------------------------------
function buildClassChecks() {
  $("classChecks").innerHTML = Object.entries(CLASSES).map(([id, name]) =>
    `<label><input type="checkbox" value="${id}" checked><span class="swatch" style="background:${COLORS[name]}"></span>${name}</label>`).join("");
  $("classChecks").addEventListener("change", () => renderCounts(null));
}

function wireControls() {
  wireLineDragging();
  $("resetLineBtn").addEventListener("click", () => { line = structuredClone(DEFAULT_LINE); showPreview(); });
  $("startBtn").addEventListener("click", start);
  $("stopBtn").addEventListener("click", () => { stopRequested = true; setStatus("Stopping after this frame…"); });
  $("csvBtn").addEventListener("click", downloadCsv);
  $("sampleBtn").addEventListener("click", () => loadVideo(SAMPLE_URL, "highway.mp4").catch((e) => setStatus(e.message, true)));
  $("fileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) loadVideo(file, file.name).then(() => setStatus("Video loaded. Place the line, then press Start counting."))
      .catch((err) => setStatus(err.message, true));
  });
}

async function main() {
  buildClassChecks();
  wireControls();
  renderCounts(null);
  updateButtons(false);
  const backend = await loadModel();
  setStatus("Loading the sample video…");
  await loadVideo(SAMPLE_URL, "highway.mp4");
  if (await runDevMode() !== false) return;
  // Warm-up: the very first run is slow (the GPU compiles its programs), so do it now,
  // while the visitor is still looking at the page, not during counting.
  setStatus("Warming up the model…");
  await detector.detect(video, canvas.width, canvas.height);
  const where = { webnn: "WebNN: Windows ML engine", webgpu: `WebGPU on ${detector.gpuName || "your GPU"}`, wasm: "CPU" }[backend];
  if (backend === "wasm" && detector.softwareGpu) {
    // We skipped a software-emulated GPU; tell the visitor how to get the real one.
    return setStatus(`Ready (CPU). Your browser's GPU access is software-only (${detector.softwareGpu}). ` +
      "For 2x speed turn on 'Use graphics acceleration when available' in browser settings and restart.");
  }
  setStatus(`Ready (${where}). Press Start counting.`);
}

main().catch((err) => {
  console.error(err);
  setStatus(`Error: ${err.message}`, true);
  report({ error: String(err.stack || err) }).catch(() => {});
});
