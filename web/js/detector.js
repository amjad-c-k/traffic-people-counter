// YOLO26n object detection in the browser with ONNX Runtime Web.
// Mirrors what Ultralytics does in Python: letterbox -> model -> decode boxes -> NMS.
/* global ort */

export const IMGSZ = 640;
// COCO class IDs we care about (YOLO is trained on the 80 COCO classes).
export const CLASSES = { 0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck" };

const NUM_CLASSES = 80;
const STRIDE = 32; // the model's input sides must be multiples of 32

export class Detector {
  constructor({ rect = true, imgsz = IMGSZ } = {}) {
    this.session = null;
    this.imgsz = imgsz; // model input size (640). 480 was tested: no faster on WebGPU, misses small objects
    // rect = true: pad only up to the next multiple of 32 (e.g. 640x384 for 16:9 video) instead
    // of a full 640x640 square, like Ultralytics does with PyTorch. Needs a dynamic-shape model.
    this.rect = rect;
    // Reusable canvas for preprocessing (avoids allocating one per frame).
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  // model: URL or the downloaded model bytes (Uint8Array).
  // Backends (fastest first, where the browser supports them):
  //   "webnn"  - Windows' own ML engine (DirectML) via the WebNN API, runs on the graphics chip
  //   "webgpu" - general GPU programs written by ONNX Runtime
  //   "wasm"   - CPU, works everywhere
  // ep = "auto" tries them in that order and keeps the first that works.
  async load(model, ep = "auto") {
    this.model = model;
    const order = ep === "auto" ? ["webnn", "webgpu", "wasm"] : [ep];
    const failures = [];
    for (const backend of order) {
      if (backend === "webnn" && !("ml" in navigator)) { failures.push("webnn: not available in this browser"); continue; }
      if (backend === "webgpu" && !navigator.gpu) { failures.push("webgpu: not available in this browser"); continue; }
      if (backend === "webgpu" && ep === "auto") {
        // A "fallback" adapter is software pretending to be a GPU (e.g. hardware acceleration
        // switched off). Measured: 7.4 s per frame vs ~0.25 s on the CPU backend, so skip it.
        const gpu = await describeGpu();
        this.gpuName = gpu.name;
        if (gpu.software) { failures.push(`webgpu: software adapter (${gpu.name})`); this.softwareGpu = gpu.name; continue; }
      }
      try {
        this.ep = backend;
        // WebNN needs a fixed input size; 640x384 (16:9 video) is a test shape here, and
        // detect() makes a new session if the real video needs another shape.
        this.session = await this.createSession(backend === "webnn" ? [384, 640] : null);
        return backend;
      } catch (err) {
        console.warn(`Backend ${backend} failed, trying the next one`, err);
        failures.push(`${backend}: ${err.message}`);
      }
    }
    throw new Error(`Could not load the model with any backend. ${failures.join(" | ")}`);
  }

  async createSession(shape) {
    const opts = { executionProviders: [this.ep === "webnn" ? { name: "webnn", deviceType: "gpu" } : this.ep] };
    if (shape) opts.freeDimensionOverrides = { batch: 1, height: shape[0], width: shape[1] };
    const session = await ort.InferenceSession.create(this.model, opts);
    this.sessionShape = shape;
    return session;
  }

  // Letterbox: scale the frame to fit 640x640 WITHOUT stretching, pad the rest with grey
  // (114,114,114), exactly like Ultralytics' LetterBox. Returns the tensor + how to undo it.
  preprocess(source, srcW, srcH) {
    const S = this.imgsz;
    const r = Math.min(S / srcW, S / srcH);
    const newW = Math.round(srcW * r), newH = Math.round(srcH * r);
    // Padding: full square, or (rect) only up to the next multiple of 32 on each side.
    let padW = S - newW, padH = S - newH;
    if (this.rect) { padW %= STRIDE; padH %= STRIDE; }
    const inW = newW + padW, inH = newH + padH;
    const left = Math.round(padW / 2 - 0.1);
    const top = Math.round(padH / 2 - 0.1);

    if (this.canvas.width !== inW || this.canvas.height !== inH) {
      this.canvas.width = inW;
      this.canvas.height = inH;
    }
    this.ctx.fillStyle = "rgb(114,114,114)";
    this.ctx.fillRect(0, 0, inW, inH);
    this.ctx.drawImage(source, left, top, newW, newH);
    const rgba = this.ctx.getImageData(0, 0, inW, inH).data;

    // Canvas gives RGBA pixel by pixel (HWC); the model wants RGB planes (CHW), values 0..1.
    const plane = inW * inH;
    const data = new Float32Array(3 * plane);
    for (let i = 0, p = 0; p < plane; i += 4, p++) {
      data[p] = rgba[i] / 255;
      data[p + plane] = rgba[i + 1] / 255;
      data[p + 2 * plane] = rgba[i + 2] / 255;
    }
    return { tensor: new ort.Tensor("float32", data, [1, 3, inH, inW]), r, left, top };
  }

  // Run the model on a video frame / image. Returns [{box:[x1,y1,x2,y2], conf, classId, className}].
  async detect(source, srcW, srcH, { conf = 0.25, iou = 0.7, classIds = Object.keys(CLASSES).map(Number) } = {}) {
    const t0 = performance.now();
    const { tensor, r, left, top } = this.preprocess(source, srcW, srcH);
    // WebNN sessions have a fixed input size: rebuild once if this video needs a different one.
    const [, , inH, inW] = tensor.dims;
    if (this.ep === "webnn" && (this.sessionShape[0] !== inH || this.sessionShape[1] !== inW)) {
      this.session = await this.createSession([inH, inW]);
    }
    const t1 = performance.now();
    const feeds = { [this.session.inputNames[0]]: tensor };
    const output = (await this.session.run(feeds))[this.session.outputNames[0]];
    const t2 = performance.now();
    const numAnchors = output.dims[2]; // 8400 for 640x640, 5040 for 640x384
    const dets = postprocess(output.data, { numAnchors, r, left, top, srcW, srcH, conf, iou, classIds });
    // Time spent in each step of the last frame (ms), for the speed check.
    this.timings = { pre: t1 - t0, model: t2 - t1, post: performance.now() - t2 };
    return dets;
  }
}

// Which graphics adapter does WebGPU give us, and is it real hardware or software emulation?
export async function describeGpu() {
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  if (!adapter) return { name: "none", software: true };
  const info = adapter.info || {};
  const name = [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "unknown GPU";
  const software = Boolean(adapter.isFallbackAdapter || info.isFallbackAdapter || /swiftshader|basic render|llvmpipe/i.test(name));
  return { name, software };
}

// Model output has shape [1, 84, N]: for each of N candidates (8400 on a 640x640 input),
// 4 box numbers (centre x, centre y, width, height) then 80 class scores. Stored row by row,
// so value k of candidate i is at out[k * N + i].
export function postprocess(out, { numAnchors: NUM_ANCHORS, r, left, top, srcW, srcH, conf, iou, classIds }) {
  const wanted = new Set(classIds);
  const candidates = [];
  for (let i = 0; i < NUM_ANCHORS; i++) {
    // Best class for this candidate (same as Ultralytics: pick the top class first,
    // then drop it if that class isn't one we want).
    let best = 0, bestScore = -1;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const s = out[(4 + c) * NUM_ANCHORS + i];
      if (s > bestScore) { bestScore = s; best = c; }
    }
    if (bestScore <= conf || !wanted.has(best)) continue;

    const cx = out[i], cy = out[NUM_ANCHORS + i], w = out[2 * NUM_ANCHORS + i], h = out[3 * NUM_ANCHORS + i];
    // Undo the letterbox: remove padding, divide by the scale, keep inside the frame.
    const x1 = clamp((cx - w / 2 - left) / r, 0, srcW), y1 = clamp((cy - h / 2 - top) / r, 0, srcH);
    const x2 = clamp((cx + w / 2 - left) / r, 0, srcW), y2 = clamp((cy + h / 2 - top) / r, 0, srcH);
    candidates.push({ box: [x1, y1, x2, y2], conf: bestScore, classId: best, className: CLASSES[best] });
  }
  return nms(candidates, iou);
}

// Non-Maximum Suppression: many candidates overlap the same object. Keep the most
// confident box and remove others of the SAME class that overlap it more than `iouThr`.
export function nms(dets, iouThr, maxDet = 300) {
  dets.sort((a, b) => b.conf - a.conf);
  const keep = [];
  const removed = new Uint8Array(dets.length);
  for (let i = 0; i < dets.length && keep.length < maxDet; i++) {
    if (removed[i]) continue;
    keep.push(dets[i]);
    for (let j = i + 1; j < dets.length; j++) {
      if (!removed[j] && dets[j].classId === dets[i].classId && boxIoU(dets[i].box, dets[j].box) > iouThr) {
        removed[j] = 1;
      }
    }
  }
  return keep;
}

// Intersection over Union: overlap area / combined area (0 = no overlap, 1 = identical).
export function boxIoU(a, b) {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

function clamp(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}
