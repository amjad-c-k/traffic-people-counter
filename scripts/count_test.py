"""Line-count test: count objects crossing a horizontal line in a video.

Prints every crossing event (ID, class, IN/OUT, time) and saves the totals.

Usage (from the project folder, venv active):
    python scripts/count_test.py            # line at 50% of the frame height
    python scripts/count_test.py 0.6        # line at 60% of the frame height
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from counter.detector import Detector  # noqa: E402
from counter.line_counter import LineCounter  # noqa: E402

VIDEO = ROOT / "samples" / "highway.mp4"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def main(line_y_frac=0.5):
    det = Detector("yolo26n.pt")
    cap = cv2.VideoCapture(str(VIDEO))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Horizontal line across the full width, given as a fraction of height
    # so the same setting works for any video resolution.
    y = int(h * line_y_frac)
    a, b = (0, y), (w, y)
    counter = LineCounter(a, b)
    print(f"Line from {a} to {b}. IN = moving down (towards camera), OUT = moving up.\n")

    out_path = RESULTS / "line_count_demo.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracks = det.track(frame)
        for ev in counter.update(tracks, frame_idx, fps):
            print(f"t={ev.time_s:5.2f}s frame {ev.frame:3d}: #{ev.track_id} {ev.cls_name} {ev.direction}")

        # Minimal drawing so the counts can be checked by eye (full overlay lives in counter/draw.py).
        cv2.line(frame, a, b, (0, 255, 255), 2)
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)
            color = (0, 0, 255) if t.track_id in counter.counted else (0, 200, 0)  # red once counted
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, tuple(map(int, t.center)), 3, color, -1)
            cv2.putText(frame, f"#{t.track_id}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        tot = counter.totals()
        cv2.putText(frame, f"IN {tot['IN']}  OUT {tot['OUT']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    summary = {
        "video": VIDEO.name,
        "line_y_frac": line_y_frac,
        "totals": counter.totals(),
        "by_class": {f"{c} {d}": n for (c, d), n in sorted(counter.counts.items())},
        "events": [asdict(e) for e in counter.events],
    }
    (RESULTS / "line_counts.json").write_text(json.dumps(summary, indent=2))
    print(f"\nTotals: {summary['totals']}  by class: {summary['by_class']}")
    print(f"Saved video -> {out_path.relative_to(ROOT)}  and results/line_counts.json")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
