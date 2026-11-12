"""Drawing helpers: boxes, IDs, trails, counting line and count panel."""
from collections import defaultdict, deque

import cv2
import numpy as np

# BGR colours (OpenCV uses Blue-Green-Red order), one per class.
CLASS_COLORS = {
    "car": (255, 160, 40),        # blue
    "truck": (40, 120, 255),      # orange
    "bus": (200, 60, 200),        # purple
    "motorcycle": (60, 200, 255), # yellow
    "person": (80, 220, 80),      # green
}
DEFAULT_COLOR = (200, 200, 200)
COUNTED_COLOR = (60, 60, 255)    # red: object already counted
LINE_COLOR = (0, 230, 255)       # yellow
FONT = cv2.FONT_HERSHEY_SIMPLEX


class TrailBuffer:
    """Keeps the last N centre points of each track for drawing trails."""

    def __init__(self, length=20):
        self.points = defaultdict(lambda: deque(maxlen=length))  # old points fall off automatically

    def update(self, tracks):
        for t in tracks:
            self.points[t.track_id].append(tuple(map(int, t.center)))

    def forget(self, active_ids):
        for tid in list(self.points):
            if tid not in active_ids:
                del self.points[tid]


def _label(img, text, org, color, scale=0.5, thickness=1):
    """Text on a filled rectangle so it stays readable on any background."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    y = max(y, th + 4)  # keep inside the top of the image
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_tracks(img, tracks, counted_ids, trails):
    for t in tracks:
        color = COUNTED_COLOR if t.track_id in counted_ids else CLASS_COLORS.get(t.cls_name, DEFAULT_COLOR)
        x1, y1, x2, y2 = map(int, t.box)

        # Trail: thicker for newer points, so the direction of motion is visible.
        pts = trails.points.get(t.track_id, [])
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, max(1, i * 3 // len(pts)), cv2.LINE_AA)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, tuple(map(int, t.center)), 3, color, -1)
        _label(img, f"#{t.track_id} {t.cls_name}", (x1, y1 - 2), color, scale=0.45)


def draw_line(img, a, b):
    """Counting line with small arrows showing which way is IN and OUT."""
    cv2.line(img, a, b, LINE_COLOR, 2, cv2.LINE_AA)
    # Normal vector (perpendicular to the line) pointing to the IN side (positive side).
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    nx, ny = -dy / length, dx / length
    # Put the arrows near the start of the line (usually the frame edge) so
    # they don't cover vehicles in the middle of the road.
    mx, my = int(a[0] + dx * 0.05), int(a[1] + dy * 0.05)
    tip_in = (int(mx + nx * 30), int(my + ny * 30))
    tip_out = (int(mx - nx * 30), int(my - ny * 30))
    cv2.arrowedLine(img, (mx, my), tip_in, LINE_COLOR, 2, tipLength=0.4)
    cv2.arrowedLine(img, (mx, my), tip_out, LINE_COLOR, 2, tipLength=0.4)
    _label(img, "IN", (tip_in[0] + 6, tip_in[1] + 6), (0, 140, 0))
    _label(img, "OUT", (tip_out[0] + 6, tip_out[1] + 6), (0, 0, 170))


def draw_panel(img, counts, classes):
    """Semi-transparent table in the top-left: one row per class, IN / OUT columns."""
    rows = [c for c in classes if any(counts.get((c, d), 0) for d in ("IN", "OUT"))] or classes[:1]
    row_h, pad = 24, 10
    w, h = 230, pad * 2 + row_h * (len(rows) + 2)
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + w, 10 + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)  # blend = transparency

    def row(y, name, vin, vout, color=(255, 255, 255)):
        cv2.putText(img, name, (20, y), FONT, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(img, str(vin), (130, y), FONT, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(img, str(vout), (185, y), FONT, 0.55, color, 1, cv2.LINE_AA)

    y = 10 + pad + 16
    row(y, "class", "IN", "OUT", (180, 180, 180))
    for c in rows:
        y += row_h
        row(y, c, counts.get((c, "IN"), 0), counts.get((c, "OUT"), 0), CLASS_COLORS.get(c, DEFAULT_COLOR))
    y += row_h
    tin = sum(n for (_, d), n in counts.items() if d == "IN")
    tout = sum(n for (_, d), n in counts.items() if d == "OUT")
    row(y, "TOTAL", tin, tout, LINE_COLOR)
    return np.asarray(img)
