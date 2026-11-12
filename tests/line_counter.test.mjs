// Same checks as tests/test_line_counter.py, for the JavaScript port (web/js/lineCounter.js).
// Run:  node --test "tests/*.test.mjs"
import { test } from "node:test";
import assert from "node:assert/strict";
import { LineCounter } from "../web/js/lineCounter.js";

const FPS = 25;
const boxAt = (x, y, s = 10) => [x - s, y - s, x + s, y + s]; // box whose centre is (x, y)

function run(path, counter = new LineCounter([0, 100], [200, 100]), trackId = 1, className = "car") {
  path.forEach(([x, y], i) => counter.update([{ trackId, className, box: boxAt(x, y) }], i, FPS));
  return counter;
}

test("moving down counts IN", () => {
  assert.deepEqual(run([[50, 80], [50, 95], [50, 105], [50, 120]]).totals(), { IN: 1, OUT: 0 });
});

test("moving up counts OUT", () => {
  assert.deepEqual(run([[50, 120], [50, 105], [50, 95], [50, 80]]).totals(), { IN: 0, OUT: 1 });
});

test("never counted twice", () => {
  assert.equal(run([[50, 80], [50, 120], [50, 80], [50, 120]]).events.length, 1);
});

test("no count without crossing", () => {
  assert.equal(run([[50, 80], [60, 85], [70, 90], [80, 95]]).events.length, 0);
});

test("passing beyond the line's end is not counted", () => {
  assert.equal(run([[300, 80], [300, 120]]).events.length, 0);
});

test("stopping exactly on the line counts once", () => {
  assert.deepEqual(run([[50, 90], [50, 100], [50, 110]]).totals(), { IN: 1, OUT: 0 });
});

test("two objects are counted separately", () => {
  const lc = new LineCounter([0, 100], [200, 100]);
  [[80, 120], [120, 80]].forEach(([y1, y2], i) => lc.update([
    { trackId: 1, className: "car", box: boxAt(50, y1) },
    { trackId: 2, className: "truck", box: boxAt(150, y2) },
  ], i, FPS));
  assert.deepEqual(Object.fromEntries(lc.counts), { "car|IN": 1, "truck|OUT": 1 });
});
