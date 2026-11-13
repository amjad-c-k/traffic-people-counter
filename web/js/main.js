// Page controller: wires the controls to the pipeline  detect -> track -> count -> draw.
import { Detector, CLASSES } from "./detector.js";
import { ByteTracker } from "./tracker.js";
import { LineCounter, center } from "./lineCounter.js";
import { COLORS, drawFrame, drawPreview } from "./draw.js";

// Replaced at build time (scripts/build_web.py) with a short code that changes on every
// deploy; adding it to URLs stops browsers from mixing old cached files with new ones.
const VERSION = "__VERSION__";
const MODEL_URL = `model/yolo26n.onnx?v=${VERSION}`;
const SAMPLE_URL = "samples/highway.mp4";
// Browsers don't tell us a video's frame rate, so we step through every video at a fixed
// 25 frames per second of video time (the sample video's own rate).
const STEP_FPS = 25;
const MAX_WIDTH = 1280; // bigger videos are drawn/processed at this width (YOLO resizes to 640 anyway)

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

function linePoints() {
  const w = canvas.width, h = canvas.height;
  return [[0, Math.trunc(Number($("leftY").value) * h)], [w, Math.trunc(Number($("rightY").value) * h)]];
}

function updateButtons(ready = !!video.src) {
  $("startBtn").disabled = busy || !ready || !detector.session;
  $("stopBtn").disabled = !busy;
  $("sampleBtn").disabled = busy;
  $("fileInput").disabled = busy;
  $("csvBtn").disabled = busy || lastEvents.length === 0;
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

async function showPreview() {
  if (!video.src || busy) return;
  if (video.currentTime === 0) await seek(0.5 / STEP_FPS);
  drawPreview(ctx, video, ...linePoints());
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
  setStatus("Starting the AI engine (first visit also downloads ~15 MB)…");
  const backend = await detector.load(bytes, params.get("ep") || "auto");
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
  drawPreview(ctx, video, ...linePoints());
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

    const elapsed = (performance.now() - t0) / 1000, speed = (n + 1) / elapsed;
    const left = Math.max(0, (total - n - 1) / speed);
    $("progress").value = (n + 1) / total;
    const where = detector.ep === "webgpu" ? "WebGPU" : "CPU";
    // The first frames include warm-up, so the time estimate is only shown after a few frames.
    const eta = n >= 5 ? ` · about ${Math.ceil(left)} s left` : "";
    setStatus(`Frame ${n + 1}/${total} · ${speed.toFixed(1)} FPS in your browser (${where})${eta}`);
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
    for (const [id, v] of [["leftY", "0.4"], ["rightY", "0.6"]]) {
      $(id).value = v;
      $(id).dispatchEvent(new Event("input"));
    }
    await new Promise((r) => setTimeout(r, 1500)); // let the preview redraw
    const preview = canvas.toDataURL("image/png");
    $("startBtn").click();
    await new Promise((r) => setTimeout(r, 500));
    while (busy) await new Promise((r) => setTimeout(r, 500));
    return report({
      preview, final: canvas.toDataURL("image/png"), status: $("status").textContent,
      table: [...document.querySelectorAll("#countsBody tr")].map((tr) => tr.innerText.replace(/\s+/g, " ").trim()),
      events: lastEvents.length, csvDisabled: $("csvBtn").disabled,
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
  for (const id of ["leftY", "rightY"]) {
    $(id).addEventListener("input", () => {
      $("leftVal").textContent = Number($("leftY").value).toFixed(2);
      $("rightVal").textContent = Number($("rightY").value).toFixed(2);
      showPreview();
    });
  }
  $("leftVal").textContent = Number($("leftY").value).toFixed(2);
  $("rightVal").textContent = Number($("rightY").value).toFixed(2);
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
  setStatus(`Ready (${backend === "webgpu" ? "WebGPU: using your graphics chip" : "CPU mode"}). Press Start counting.`);
}

main().catch((err) => {
  console.error(err);
  setStatus(`Error: ${err.message}`, true);
  report({ error: String(err.stack || err) }).catch(() => {});
});
