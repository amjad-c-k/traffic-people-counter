// Page logic: load the model, then for every video frame: detect -> track -> count -> draw.
import { Detector, CLASSES } from "./detector.js";
import { ByteTracker } from "./tracker.js";
import { LineCounter, center } from "./lineCounter.js";

const MODEL_URL = "model/yolo26n.onnx";
const SAMPLE_URL = "samples/highway.mp4";
const FPS = 25; // the sample video's frame rate
const LINE_Y = 0.5; // counting line at half the frame height (same default as the Python app)
const COLORS = { car: "#2d9cff", truck: "#ff7a28", bus: "#c83cc8", motorcycle: "#ffc83c", person: "#50dc50" };
const COUNTED_COLOR = "#ff3c3c";

const video = document.getElementById("video");
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const detector = new Detector();

function setStatus(text) {
  statusEl.textContent = text;
  console.log(`[status] ${text}`); // also in the browser console, handy for debugging
}

// Wait until the video shows the frame at time t (seeking is asynchronous).
function seek(t) {
  return new Promise((resolve) => {
    video.addEventListener("seeked", resolve, { once: true });
    video.currentTime = t;
  });
}

// --- drawing -------------------------------------------------------------------------------
function label(text, x, y, color) {
  ctx.font = "13px system-ui, sans-serif";
  const w = ctx.measureText(text).width + 6;
  ctx.fillStyle = color;
  ctx.fillRect(x, Math.max(0, y - 17), w, 17);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x + 3, Math.max(13, y - 4));
}

function drawFrame(tracks, counter, trails) {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  // Counting line
  ctx.strokeStyle = "#ffe600";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(...counter.a);
  ctx.lineTo(...counter.b);
  ctx.stroke();

  for (const t of tracks) {
    const color = counter.counted.has(t.trackId) ? COUNTED_COLOR : COLORS[t.className] || "#ccc";
    // Trail: the last few centre points, so the direction of motion is visible.
    const pts = trails.get(t.trackId) || [];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    const [x1, y1, x2, y2] = t.box;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    label(`#${t.trackId} ${t.className}`, x1, y1, color);
  }
  const { IN, OUT } = counter.totals();
  ctx.fillStyle = "rgba(20,20,20,0.65)";
  ctx.fillRect(10, 10, 170, 40);
  ctx.fillStyle = "#ffe600";
  ctx.font = "bold 20px system-ui, sans-serif";
  ctx.fillText(`IN ${IN}   OUT ${OUT}`, 22, 38);
}

// --- processing ----------------------------------------------------------------------------
// Process the video frame by frame (seek, detect, track, count, draw). Seeking instead of
// normal playback means no frame is skipped, however slow the device is, so tracking sees
// the same frames as the Python version.
async function processVideo({ maxFrames = Infinity, onFrame = null } = {}) {
  const w = video.videoWidth, h = video.videoHeight;
  const total = Math.min(Math.floor(video.duration * FPS), maxFrames);
  const tracker = new ByteTracker();
  const counter = new LineCounter([0, Math.trunc(LINE_Y * h)], [w, Math.trunc(LINE_Y * h)]);
  const trails = new Map(); // trackId -> last 20 centre points
  const t0 = performance.now();

  for (let n = 0; n < total; n++) {
    await seek((n + 0.5) / FPS); // +0.5: land in the middle of frame n, not on its edge
    const dets = await detector.detect(video, w, h);
    const tracks = tracker.update(dets, [w, h]).map((t) => ({ ...t, className: CLASSES[t.classId] }));
    counter.update(tracks, n, FPS);
    for (const t of tracks) {
      const pts = trails.get(t.trackId) || [];
      pts.push(center(t.box));
      trails.set(t.trackId, pts.slice(-20));
    }
    drawFrame(tracks, counter, trails);
    if (onFrame) onFrame(n, tracks);

    const speed = (n + 1) / ((performance.now() - t0) / 1000);
    const { IN, OUT } = counter.totals();
    const where = detector.ep === "webgpu" ? "WebGPU" : "CPU";
    setStatus(`Frame ${n + 1}/${total} · IN ${IN} · OUT ${OUT} · ${speed.toFixed(1)} FPS in your browser (${where})`);
  }
  return { counter, seconds: (performance.now() - t0) / 1000, frames: total };
}

// --- developer checks (only used with the local test server, see scripts/check_web.py) ------
const report = (data) => fetch("/report", { method: "POST", body: JSON.stringify(data) });

// ?check=0,60,120: detections on single frames, compared with Python.
async function checkDetections(frameList) {
  const results = {};
  let seekMs = 0, detectMs = 0;
  // Warm-up (not timed): the first run is slow, e.g. WebGPU compiles its GPU programs then.
  await seek(0.5 / FPS);
  await detector.detect(video, video.videoWidth, video.videoHeight);
  for (const n of frameList) {
    const t0 = performance.now();
    await seek((n + 0.5) / FPS);
    const t1 = performance.now();
    results[n] = await detector.detect(video, video.videoWidth, video.videoHeight);
    seekMs += t1 - t0;
    detectMs += performance.now() - t1;
  }
  const k = frameList.length;
  await report({
    results, msPerFrame: (seekMs + detectMs) / k, seekMs: seekMs / k, detectMs: detectMs / k,
    // Multi-threaded WASM only works on "cross-origin isolated" pages (special HTTP headers).
    crossOriginIsolated: self.crossOriginIsolated, wasmThreads: ort.env.wasm.numThreads, ep: detector.ep,
    userAgent: navigator.userAgent,
  });
}

// ?track=1: the whole pipeline on the whole video; reports crossing events + per-frame IDs.
async function checkTracking() {
  const ids = [];
  const { counter, seconds, frames } = await processVideo({ onFrame: (n, tracks) => ids.push(tracks.map((t) => t.trackId)) });
  await report({ events: counter.events, totals: counter.totals(), ids, seconds, frames, ep: detector.ep });
}

// ?snapshot=N: process N frames, then send the drawn canvas as a PNG.
async function snapshot(n) {
  await processVideo({ maxFrames: n });
  await report({ snapshot: canvas.toDataURL("image/png"), status: statusEl.textContent });
}

async function main() {
  setStatus("Loading model (first visit downloads ~30 MB, then it's cached)...");
  const params = new URLSearchParams(location.search);
  const backend = await detector.load(MODEL_URL, params.get("ep") || "auto");
  setStatus(`Model loaded (${backend === "webgpu" ? "WebGPU, graphics chip" : "WASM, CPU"}). Loading video...`);
  // Load the whole video into memory (a "blob") first: blob videos can always be seeked
  // frame by frame, whatever the web server supports. Uploaded files work the same way.
  const blob = await (await fetch(SAMPLE_URL)).blob();
  video.src = URL.createObjectURL(blob);
  await new Promise((resolve) => video.addEventListener("loadeddata", resolve, { once: true }));
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  if (params.get("check")) return checkDetections(params.get("check").split(",").map(Number));
  if (params.get("track")) return checkTracking();
  if (params.get("snapshot")) return snapshot(Number(params.get("snapshot")));
  const { counter, seconds, frames } = await processVideo();
  const { IN, OUT } = counter.totals();
  setStatus(`Done: ${frames} frames in ${seconds.toFixed(0)} s · IN ${IN} · OUT ${OUT}`);
}

main().catch((err) => {
  console.error(err);
  setStatus(`Error: ${err.message}`);
  report({ error: String(err.stack || err) }).catch(() => {});
});
