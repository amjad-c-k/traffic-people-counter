// ByteTrack-style multi-object tracker, ported from Ultralytics' BYTETracker so the browser
// gives the same track IDs as the Python app. Settings match bytetrack.yaml.
//
// Each frame:
//   1. Predict where every known track should be now (Kalman filter).
//   2. Match tracks to high-confidence detections by box overlap (IoU x score).
//   3. Give unmatched tracks a second chance with low-confidence detections (IoU only).
//   4. Brand-new tracks must be seen again next frame to be confirmed.
//   5. Unmatched detections start new tracks; tracks lost for > 30 frames are removed.
import * as kf from "./kalman.js";

const CFG = {
  trackHighThresh: 0.25, // detections at or above this go to the first matching stage
  trackLowThresh: 0.1,   // between low and high: only used in the second stage
  newTrackThresh: 0.25,  // minimum score to start a new track
  trackBuffer: 30,       // frames a lost track is remembered
  matchThresh: 0.8,      // max cost (1 - IoU*score) for a first-stage match
};

const State = { New: 0, Tracked: 1, Lost: 2, Removed: 3 };

// Box formats: xyxy = corners, xyah = centre x/y, aspect ratio w/h, height.
const xyxyToXyah = ([x1, y1, x2, y2]) => [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / (y2 - y1), y2 - y1];

class STrack {
  constructor(det) {
    this.detBox = det.box;      // raw detection box, used until the Kalman filter starts
    this.score = det.conf;
    this.classId = det.classId;
    this.mean = null;
    this.cov = null;
    this.state = State.New;
    this.isActivated = false;
    this.trackId = 0;
    this.frameId = 0;
    this.startFrame = 0;
  }

  // Current box: from the Kalman state once tracking started (this smooths jitter).
  get xyxy() {
    if (!this.mean) return [...this.detBox];
    const [cx, cy, a, h] = this.mean;
    const w = a * h;
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  }

  activate(nextId, frameId) {
    this.trackId = nextId();
    ({ mean: this.mean, cov: this.cov } = kf.initiate(xyxyToXyah(this.detBox)));
    this.state = State.Tracked;
    this.isActivated = frameId === 1; // only first-frame tracks are confirmed immediately
    this.frameId = frameId;
    this.startFrame = frameId;
  }

  update(det, frameId) { // matched while being tracked
    ({ mean: this.mean, cov: this.cov } = kf.update(this.mean, this.cov, xyxyToXyah(det.detBox)));
    this.state = State.Tracked;
    this.isActivated = true;
    this.frameId = frameId;
    this.score = det.score;
    this.classId = det.classId;
  }

  reActivate(det, frameId) { // matched again after being lost: keeps its old ID
    this.update(det, frameId);
  }
}

// IoU between every track box and every detection box -> cost = 1 - IoU (0 = perfect overlap).
function iouDistance(tracks, dets) {
  return tracks.map((t) => {
    const a = t.xyxy;
    return dets.map((d) => {
      const b = d.xyxy;
      const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
      const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
      const inter = iw * ih;
      const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
      return 1 - inter / (union + 1e-7);
    });
  });
}

// Fuse detection confidence into the cost: a confident detection is a more trustworthy match.
const fuseScore = (cost, dets) => cost.map((row) => row.map((c, j) => 1 - (1 - c) * dets[j].score));

// Optimal one-to-one matching (Hungarian algorithm) where any pair costing more than
// `thresh` is left unmatched. Same set-up as lap.lapjv(extend_cost=True, cost_limit=thresh):
// a square matrix where "no match" costs thresh/2 for each side.
// m (number of columns) is passed in because an empty cost matrix (no tracks yet) can't tell
// how many detections there are.
export function linearAssignment(cost, thresh, m = cost.length ? cost[0].length : 0) {
  const n = cost.length;
  if (n === 0 || m === 0) return { matches: [], uA: [...Array(n).keys()], uB: [...Array(m).keys()] };
  const N = n + m;
  const C = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => (i < n && j < m ? cost[i][j] : i >= n && j >= m ? 0 : thresh / 2)));
  const rowOf = hungarian(C); // rowOf[col] = row assigned to that column
  const colOf = new Array(N);
  rowOf.forEach((r, c) => { colOf[r] = c; });
  const matches = [], uA = [], uB = [];
  for (let i = 0; i < n; i++) (colOf[i] < m ? matches.push([i, colOf[i]]) : uA.push(i));
  for (let j = 0; j < m; j++) if (rowOf[j] >= n) uB.push(j);
  return { matches, uA, uB };
}

// Classic O(N^3) Hungarian algorithm with potentials, for a square cost matrix.
function hungarian(C) {
  const N = C.length, INF = Infinity;
  const u = new Array(N + 1).fill(0), v = new Array(N + 1).fill(0);
  const p = new Array(N + 1).fill(0), way = new Array(N + 1).fill(0); // p[col] = row (1-based)
  for (let i = 1; i <= N; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(N + 1).fill(INF), used = new Array(N + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF, j1 = 0;
      for (let j = 1; j <= N; j++) {
        if (used[j]) continue;
        const cur = C[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= N; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  return p.slice(1).map((r) => r - 1);
}

const joint = (a, b) => { const ids = new Set(a.map((t) => t.trackId)); return [...a, ...b.filter((t) => !ids.has(t.trackId))]; };
const subtract = (a, b) => { const ids = new Set(b.map((t) => t.trackId)); return a.filter((t) => !ids.has(t.trackId)); };

export class ByteTracker {
  constructor(cfg = {}) {
    this.cfg = { ...CFG, ...cfg };
    this.reset();
  }

  reset() {
    this.tracked = [];
    this.lost = [];
    this.removed = [];
    this.frameId = 0;
    let id = 0;
    this.nextId = () => ++id; // IDs start at 1, like Ultralytics
  }

  // dets: [{box:[x1,y1,x2,y2], conf, classId}] for one frame. frameSize: [width, height].
  // Returns confirmed tracks: [{trackId, box, conf, classId}] with Kalman-smoothed boxes,
  // clipped to the frame like Ultralytics does (a predicted box can stick out past the edge).
  update(dets, frameSize = null) {
    const c = this.cfg;
    this.frameId++;
    const activated = [], refind = [], lostNow = [], removedNow = [];

    const valid = dets.filter((d) => d.box[2] > d.box[0] && d.box[3] > d.box[1]);
    const high = valid.filter((d) => d.conf >= c.trackHighThresh).map((d) => new STrack(d));
    const low = valid.filter((d) => d.conf > c.trackLowThresh && d.conf < c.trackHighThresh).map((d) => new STrack(d));

    const unconfirmed = this.tracked.filter((t) => !t.isActivated);
    const confirmed = this.tracked.filter((t) => t.isActivated);
    const pool = joint(confirmed, this.lost);

    // 1. Predict. Lost tracks stop growing/shrinking (height speed set to 0).
    for (const t of pool) {
      const mean = [...t.mean];
      if (t.state !== State.Tracked) mean[7] = 0;
      ({ mean: t.mean, cov: t.cov } = kf.predict(mean, t.cov));
    }

    const apply = (track, det) => {
      if (track.state === State.Tracked) { track.update(det, this.frameId); activated.push(track); }
      else { track.reActivate(det, this.frameId); refind.push(track); }
    };

    // 2. First association: all tracks vs high-confidence detections.
    const first = linearAssignment(fuseScore(iouDistance(pool, high), high), c.matchThresh, high.length);
    for (const [ti, di] of first.matches) apply(pool[ti], high[di]);

    // 3. Second association: still-unmatched tracked tracks vs low-confidence detections.
    const remaining = first.uA.map((i) => pool[i]).filter((t) => t.state === State.Tracked);
    let stillUnmatched = remaining.map((_, i) => i);
    if (remaining.length && low.length) {
      const second = linearAssignment(iouDistance(remaining, low), 0.5, low.length);
      for (const [ti, di] of second.matches) apply(remaining[ti], low[di]);
      stillUnmatched = second.uA;
    }
    for (const i of stillUnmatched) { remaining[i].state = State.Lost; lostNow.push(remaining[i]); }

    // 4. Unconfirmed (new last frame) tracks vs leftover high detections.
    let leftover = first.uB.map((i) => high[i]);
    if (unconfirmed.length) {
      const third = linearAssignment(fuseScore(iouDistance(unconfirmed, leftover), leftover), 0.7, leftover.length);
      for (const [ti, di] of third.matches) { unconfirmed[ti].update(leftover[di], this.frameId); activated.push(unconfirmed[ti]); }
      for (const i of third.uA) { unconfirmed[i].state = State.Removed; removedNow.push(unconfirmed[i]); }
      leftover = third.uB.map((i) => leftover[i]);
    }

    // 5. New tracks from confident leftover detections.
    for (const det of leftover) {
      if (det.score < c.newTrackThresh) continue;
      det.activate(this.nextId, this.frameId);
      activated.push(det);
    }

    // 6. Forget tracks that have been lost for too long.
    for (const t of this.lost) {
      if (this.frameId - t.frameId > c.trackBuffer) { t.state = State.Removed; removedNow.push(t); }
    }

    // Bookkeeping (same order as Ultralytics' merge_track_pools).
    this.tracked = this.tracked.filter((t) => t.state === State.Tracked);
    this.tracked = joint(joint(this.tracked, activated), refind);
    this.lost = subtract(this.lost, this.tracked);
    this.lost.push(...lostNow);
    // Note: like Ultralytics, only tracks removed in EARLIER frames are subtracted here, so a
    // track removed this frame stays in `lost` for one more frame (kept for identical results).
    this.lost = subtract(this.lost, this.removed);
    [this.tracked, this.lost] = removeDuplicates(this.tracked, this.lost);
    this.removed.push(...removedNow);
    if (this.removed.length > 1000) this.removed = this.removed.slice(-1000);

    const clip = (box) => {
      if (!frameSize) return box;
      const [w, h] = frameSize;
      return [Math.min(Math.max(box[0], 0), w), Math.min(Math.max(box[1], 0), h),
              Math.min(Math.max(box[2], 0), w), Math.min(Math.max(box[3], 0), h)];
    };
    return this.tracked.filter((t) => t.isActivated)
      .map((t) => ({ trackId: t.trackId, box: clip(t.xyxy), conf: t.score, classId: t.classId }));
  }
}

// If a tracked and a lost track overlap almost completely (IoU > 0.85), they are the same
// object: keep the one that has existed longer.
function removeDuplicates(a, b) {
  const dist = iouDistance(a, b);
  const dropA = new Set(), dropB = new Set();
  dist.forEach((row, p) => row.forEach((d, q) => {
    if (d < 0.15) {
      const ageP = a[p].frameId - a[p].startFrame, ageQ = b[q].frameId - b[q].startFrame;
      if (ageP > ageQ) dropB.add(q); else dropA.add(p);
    }
  }));
  return [a.filter((_, i) => !dropA.has(i)), b.filter((_, i) => !dropB.has(i))];
}
