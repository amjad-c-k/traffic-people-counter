"""My own line-crossing counter (no library magic).

Idea: remember each track's last centre point. Every frame, check whether the
straight path from the old centre to the new centre crosses the counting line.
If it does, count it once as IN or OUT depending on the direction.
"""
from dataclasses import dataclass


def side_of_line(a, b, p):
    """Which side of line A->B is point P on?

    2D cross product of (B - A) and (P - A):
      > 0  -> one side, < 0 -> the other side, == 0 -> exactly on the line.
    In image coordinates (y grows downward) and a line drawn left -> right,
    a positive value means P is BELOW the line.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1, p2, a, b):
    """True if segment p1->p2 (the object's movement) crosses segment a->b (the line).

    Two segments cross when the ends of EACH one are on opposite sides of the other.
    Checking only one direction would also count objects passing beyond the line's ends.
    """
    d1, d2 = side_of_line(a, b, p1), side_of_line(a, b, p2)
    d3, d4 = side_of_line(p1, p2, a), side_of_line(p1, p2, b)
    return d1 * d2 < 0 and d3 * d4 <= 0


@dataclass
class CrossingEvent:
    track_id: int
    cls_name: str
    direction: str  # "IN" or "OUT"
    frame: int
    time_s: float   # seconds from the start of the video


class LineCounter:
    """Counts tracked objects crossing a line, separately for IN and OUT.

    IN  = moving from the negative side to the positive side of A->B
          (for a left->right line: moving DOWN the image, i.e. towards the camera).
    OUT = the opposite direction.
    """

    def __init__(self, a, b, forget_after=30):
        self.a, self.b = a, b
        self.last_pos = {}       # track_id -> last centre that was NOT exactly on the line
        self.last_seen = {}      # track_id -> frame number it was last seen
        self.counted = set()     # IDs already counted: an ID is never counted twice
        self.events = []         # list of CrossingEvent
        self.counts = {}         # (cls_name, direction) -> count
        self.forget_after = forget_after  # drop memory of IDs unseen for this many frames

    def update(self, tracks, frame_idx, fps):
        """Feed this frame's tracks. Returns the list of new crossings in this frame."""
        new_events = []
        for t in tracks:
            tid, now = t.track_id, t.center
            prev = self.last_pos.get(tid)
            self.last_seen[tid] = frame_idx

            if prev is not None and tid not in self.counted and segments_intersect(prev, now, self.a, self.b):
                direction = "IN" if side_of_line(self.a, self.b, now) > 0 else "OUT"
                ev = CrossingEvent(tid, t.cls_name, direction, frame_idx, round(frame_idx / fps, 2))
                self.counted.add(tid)
                self.events.append(ev)
                key = (t.cls_name, direction)
                self.counts[key] = self.counts.get(key, 0) + 1
                new_events.append(ev)

            # If the centre sits exactly ON the line, keep the old point so the
            # crossing is still detected on the next frame.
            if side_of_line(self.a, self.b, now) != 0:
                self.last_pos[tid] = now

        self._forget_old(frame_idx)
        return new_events

    def _forget_old(self, frame_idx):
        """Free memory for tracks that left the scene (keeps long videos light)."""
        stale = [tid for tid, f in self.last_seen.items() if frame_idx - f > self.forget_after]
        for tid in stale:
            self.last_pos.pop(tid, None)
            self.last_seen.pop(tid, None)

    def totals(self):
        """Total IN and OUT across all classes."""
        tin = sum(n for (_, d), n in self.counts.items() if d == "IN")
        tout = sum(n for (_, d), n in self.counts.items() if d == "OUT")
        return {"IN": tin, "OUT": tout}
