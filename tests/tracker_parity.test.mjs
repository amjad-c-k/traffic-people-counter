// Parity test: does the JavaScript tracker (web/js/tracker.js) behave like Ultralytics' ByteTrack?
// tests/fixtures/tracking_reference.json (made by scripts/make_tracking_reference.py) holds, for each
// frame of the sample video, the detections Python's tracker received and the tracks it returned.
// We feed the SAME detections into the JS tracker and compare IDs, boxes and line-crossing events.
// Run:  node --test "tests/*.test.mjs"
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ByteTracker, linearAssignment } from "../web/js/tracker.js";
import { LineCounter } from "../web/js/lineCounter.js";

const ref = JSON.parse(readFileSync(new URL("./fixtures/tracking_reference.json", import.meta.url)));
const CLASS_NAMES = { 0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck" };

function runJs() {
  const tracker = new ByteTracker();
  const counter = new LineCounter(...ref.line);
  const frames = ref.frames.map((f, i) => {
    const tracks = tracker.update(f.dets, [ref.width, ref.height]).map((t) => ({ ...t, className: CLASS_NAMES[t.classId] }));
    counter.update(tracks, i, ref.fps);
    return tracks;
  });
  return { frames, counter };
}

test("linear assignment leaves pairs above the threshold unmatched", () => {
  const { matches, uA, uB } = linearAssignment([[0.1, 0.9], [0.9, 0.95]], 0.8);
  assert.deepEqual(matches, [[0, 0]]);
  assert.deepEqual(uA, [1]);
  assert.deepEqual(uB, [1]);
});

test("same track IDs as Python on every frame", () => {
  const { frames } = runJs();
  const mismatches = [];
  frames.forEach((tracks, i) => {
    const js = tracks.map((t) => t.trackId).sort((a, b) => a - b).join(",");
    const py = ref.frames[i].tracks.map((t) => t.trackId).sort((a, b) => a - b).join(",");
    if (js !== py) mismatches.push(`frame ${i}: python [${py}] js [${js}]`);
  });
  assert.equal(mismatches.length, 0, mismatches.slice(0, 5).join("\n"));
});

test("same Kalman-smoothed boxes as Python (within 0.5 px)", () => {
  const { frames } = runJs();
  let worst = 0;
  frames.forEach((tracks, i) => {
    const py = new Map(ref.frames[i].tracks.map((t) => [t.trackId, t.box]));
    for (const t of tracks) {
      const b = py.get(t.trackId);
      if (b) worst = Math.max(worst, ...t.box.map((v, k) => Math.abs(v - b[k])));
    }
  });
  assert.ok(worst < 0.5, `largest box difference ${worst.toFixed(3)} px`);
});

test("same line-crossing events as Python", () => {
  const { counter } = runJs();
  const simplify = (e) => `${e.trackId} ${e.className} ${e.direction} @${e.frame}`;
  assert.deepEqual(counter.events.map(simplify), ref.events.map(simplify));
  assert.deepEqual(counter.totals(), ref.totals);
});
