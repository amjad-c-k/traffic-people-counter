// Line-crossing counter: a direct port of counter/line_counter.py.
//
// Remember each track's last centre point. Every frame, check whether the path from the old
// centre to the new centre crosses the counting line. If it does, count it once as IN or OUT.

// Which side of line A->B is point P on? (2D cross product: >0 one side, <0 other, 0 on the line.)
// In image coordinates (y grows downward) with a left->right line, positive means BELOW the line.
export function sideOfLine(a, b, p) {
  return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
}

// True if segment p1->p2 (the object's movement) crosses segment a->b (the line).
// The ends of EACH segment must be on opposite sides of the other one; checking only one
// direction would also count objects passing beyond the line's ends.
export function segmentsIntersect(p1, p2, a, b) {
  const d1 = sideOfLine(a, b, p1), d2 = sideOfLine(a, b, p2);
  const d3 = sideOfLine(p1, p2, a), d4 = sideOfLine(p1, p2, b);
  return d1 * d2 < 0 && d3 * d4 <= 0;
}

export const center = ([x1, y1, x2, y2]) => [(x1 + x2) / 2, (y1 + y2) / 2];

// IN  = moving from the negative to the positive side of A->B
//       (for a left->right line: moving DOWN the image, towards the camera). OUT = opposite.
export class LineCounter {
  constructor(a, b, forgetAfter = 30) {
    this.a = a;
    this.b = b;
    this.lastPos = new Map();   // trackId -> last centre NOT exactly on the line
    this.lastSeen = new Map();  // trackId -> last frame seen
    this.counted = new Set();   // IDs already counted: never counted twice
    this.events = [];           // {trackId, className, direction, frame, timeS}
    this.counts = new Map();    // "class|direction" -> count
    this.forgetAfter = forgetAfter;
  }

  // tracks: [{trackId, box, className}] for this frame. Returns the new crossing events.
  update(tracks, frame, fps) {
    const newEvents = [];
    for (const t of tracks) {
      const now = center(t.box);
      const prev = this.lastPos.get(t.trackId);
      this.lastSeen.set(t.trackId, frame);

      if (prev && !this.counted.has(t.trackId) && segmentsIntersect(prev, now, this.a, this.b)) {
        const direction = sideOfLine(this.a, this.b, now) > 0 ? "IN" : "OUT";
        const ev = { trackId: t.trackId, className: t.className, direction, frame,
                     timeS: Math.round((frame / fps) * 100) / 100 };
        this.counted.add(t.trackId);
        this.events.push(ev);
        const key = `${t.className}|${direction}`;
        this.counts.set(key, (this.counts.get(key) || 0) + 1);
        newEvents.push(ev);
      }
      // A centre exactly ON the line keeps the old point, so the crossing is caught next frame.
      if (sideOfLine(this.a, this.b, now) !== 0) this.lastPos.set(t.trackId, now);
    }
    // Free memory for tracks that left the scene.
    for (const [id, f] of this.lastSeen) {
      if (frame - f > this.forgetAfter) { this.lastSeen.delete(id); this.lastPos.delete(id); }
    }
    return newEvents;
  }

  totals() {
    let IN = 0, OUT = 0;
    for (const [key, n] of this.counts) (key.endsWith("|IN") ? (IN += n) : (OUT += n));
    return { IN, OUT };
  }
}
