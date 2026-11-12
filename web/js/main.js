// Page logic: load the model, play the sample video, detect objects on each frame and draw them.
import { Detector } from "./detector.js";

const MODEL_URL = "model/yolo26n.onnx";
const SAMPLE_URL = "samples/highway.mp4";
const COLORS = { car: "#2d9cff", truck: "#ff7a28", bus: "#c83cc8", motorcycle: "#ffc83c", person: "#50dc50" };

const video = document.getElementById("video");
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const detector = new Detector();

function setStatus(text) {
  statusEl.textContent = text;
}

function draw(dets) {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  ctx.font = "14px system-ui, sans-serif";
  for (const d of dets) {
    const [x1, y1, x2, y2] = d.box;
    const color = COLORS[d.className] || "#ccc";
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${d.className} ${d.conf.toFixed(2)}`;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 18, ctx.measureText(label).width + 6, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 3, y1 - 4);
  }
}

// Wait until the video shows the frame at time t (seeking is asynchronous).
function seek(t) {
  return new Promise((resolve) => {
    video.addEventListener("seeked", resolve, { once: true });
    video.currentTime = t;
  });
}

// Process the video frame by frame (seek, detect, draw). Seeking instead of normal
// playback means no frame is skipped, however slow the device is.
async function run(snapshotAt = null) {
  const fps = 25; // the sample video's frame rate
  const total = snapshotAt ? snapshotAt : Math.floor(video.duration * fps);
  let frames = 0;
  const t0 = performance.now();
  for (let n = 0; n < total; n++) {
    await seek((n + 0.5) / fps); // +0.5: land in the middle of frame n, not on its edge
    const dets = await detector.detect(video, video.videoWidth, video.videoHeight);
    draw(dets);
    frames++;
    const speed = frames / ((performance.now() - t0) / 1000);
    const where = detector.ep === "webgpu" ? "WebGPU" : "CPU";
    setStatus(`Frame ${n + 1}/${total} · ${dets.length} objects · ${speed.toFixed(1)} FPS in your browser (${where})`);
  }
  // Developer option ?snapshot=N: after N frames, send the drawn canvas to the local test server.
  if (snapshotAt) {
    await fetch("/report", { method: "POST", body: JSON.stringify({ snapshot: canvas.toDataURL("image/png"), status: statusEl.textContent }) });
  }
}

// Developer check: ?check=0,60,120 runs detection on those frames and POSTs the boxes to
// /report (only the local test server has that endpoint). Used to compare with Python.
async function check(frameList) {
  const fps = 25, results = {};
  let seekMs = 0, detectMs = 0;
  // Warm-up (not timed): the first run is slow, e.g. WebGPU compiles its GPU programs then.
  await seek(0.5 / fps);
  await detector.detect(video, video.videoWidth, video.videoHeight);
  let adapter = null;
  if (navigator.gpu) {
    const a = await navigator.gpu.requestAdapter();
    adapter = a ? (a.info ? `${a.info.vendor} ${a.info.architecture} ${a.info.description}` : "available") : "none";
  }
  for (const n of frameList) {
    const t0 = performance.now();
    await seek((n + 0.5) / fps);
    const t1 = performance.now();
    results[n] = await detector.detect(video, video.videoWidth, video.videoHeight);
    seekMs += t1 - t0;
    detectMs += performance.now() - t1;
  }
  const k = frameList.length;
  await fetch("/report", {
    method: "POST",
    body: JSON.stringify({
      results, msPerFrame: (seekMs + detectMs) / k, seekMs: seekMs / k, detectMs: detectMs / k,
      // Multi-threaded WASM only works on "cross-origin isolated" pages (special HTTP headers).
      crossOriginIsolated: self.crossOriginIsolated, wasmThreads: ort.env.wasm.numThreads, ep: detector.ep, gpuAdapter: adapter,
      userAgent: navigator.userAgent,
    }),
  });
  setStatus("check done");
}

async function main() {
  setStatus("Loading model (first visit downloads ~30 MB, then it's cached)...");
  const ep = new URLSearchParams(location.search).get("ep") || "auto";
  const backend = await detector.load(MODEL_URL, ep);
  setStatus(`Model loaded (${backend === "webgpu" ? "WebGPU, graphics chip" : "WASM, CPU"}). Loading video...`);
  // Load the whole video into memory (a "blob") first: blob videos can always be seeked
  // frame by frame, whatever the web server supports. Uploaded files work the same way.
  const blob = await (await fetch(SAMPLE_URL)).blob();
  video.src = URL.createObjectURL(blob);
  await new Promise((resolve) => video.addEventListener("loadeddata", resolve, { once: true }));
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const params = new URLSearchParams(location.search);
  if (params.get("check")) return check(params.get("check").split(",").map(Number));
  await run(params.get("snapshot") ? Number(params.get("snapshot")) : null);
}

main().catch((err) => {
  console.error(err);
  setStatus(`Error: ${err.message}`);
  fetch("/report", { method: "POST", body: JSON.stringify({ error: String(err.stack || err) }) }).catch(() => {});
});
