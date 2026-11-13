// YOLO26n object detection in the browser with ONNX Runtime Web.
// Mirrors what Ultralytics does in Python: letterbox -> model -> decode boxes -> NMS.
/* global ort */

export const IMGSZ = 640;
// COCO class IDs we care about (YOLO is trained on the 80 COCO classes).
export const CLASSES = { 0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck" };

const NUM_CLASSES = 80;
const NUM_ANCHORS = 8400; // candidate boxes the model scores on a 640x640 input

export class Detector {
  constructor() {
    this.session = null;
    // Reusable 640x640 canvas for preprocessing (avoids allocating one per frame).
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = IMGSZ;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  // model: URL or the downloaded model bytes (Uint8Array).
  // Backends: "webgpu" runs on the graphics chip (fast where supported), "wasm" on the CPU
  // (works everywhere). ep = "auto" tries WebGPU first and falls back to WASM.
  async load(model, ep = "auto") {
    const modelUrl = model;
    const order = ep === "auto" ? (navigator.gpu ? ["webgpu", "wasm"] : ["wasm"]) : [ep];
    for (const backend of order) {
      try {
        this.session = await ort.InferenceSession.create(modelUrl, { executionProviders: [backend] });
        this.ep = backend;
        return backend;
      } catch (err) {
        console.warn(`Backend ${backend} failed, trying the next one`, err);
      }
    }
    throw new Error("Could not load the model with any backend.");
  }

  // Letterbox: scale the frame to fit 640x640 WITHOUT stretching, pad the rest with grey
  // (114,114,114), exactly like Ultralytics' LetterBox. Returns the tensor + how to undo it.
  preprocess(source, srcW, srcH) {
    const r = Math.min(IMGSZ / srcW, IMGSZ / srcH);
    const newW = Math.round(srcW * r), newH = Math.round(srcH * r);
    const left = Math.round((IMGSZ - newW) / 2 - 0.1);
    const top = Math.round((IMGSZ - newH) / 2 - 0.1);

    this.ctx.fillStyle = "rgb(114,114,114)";
    this.ctx.fillRect(0, 0, IMGSZ, IMGSZ);
    this.ctx.drawImage(source, left, top, newW, newH);
    const rgba = this.ctx.getImageData(0, 0, IMGSZ, IMGSZ).data;

    // Canvas gives RGBA pixel by pixel (HWC); the model wants RGB planes (CHW), values 0..1.
    const plane = IMGSZ * IMGSZ;
    const data = new Float32Array(3 * plane);
    for (let i = 0, p = 0; p < plane; i += 4, p++) {
      data[p] = rgba[i] / 255;
      data[p + plane] = rgba[i + 1] / 255;
      data[p + 2 * plane] = rgba[i + 2] / 255;
    }
    return { tensor: new ort.Tensor("float32", data, [1, 3, IMGSZ, IMGSZ]), r, left, top };
  }

  // Run the model on a video frame / image. Returns [{box:[x1,y1,x2,y2], conf, classId, className}].
  async detect(source, srcW, srcH, { conf = 0.25, iou = 0.7, classIds = Object.keys(CLASSES).map(Number) } = {}) {
    const { tensor, r, left, top } = this.preprocess(source, srcW, srcH);
    const feeds = { [this.session.inputNames[0]]: tensor };
    const output = (await this.session.run(feeds))[this.session.outputNames[0]].data;
    return postprocess(output, { r, left, top, srcW, srcH, conf, iou, classIds });
  }
}

// Model output has shape [1, 84, 8400]: for each of 8400 candidates, 4 box numbers
// (centre x, centre y, width, height) then 80 class scores. Stored row by row, so
// value k of candidate i is at out[k * 8400 + i].
export function postprocess(out, { r, left, top, srcW, srcH, conf, iou, classIds }) {
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
