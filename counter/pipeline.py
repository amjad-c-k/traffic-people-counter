"""Full pipeline: video in -> tracked, counted, annotated video + CSV out.

Used by both scripts/run_video.py (command line) and app.py (Gradio).
"""
import csv
import time
from pathlib import Path

import cv2

from counter.detector import CLASSES, Detector
from counter.draw import TrailBuffer, draw_line, draw_panel, draw_tracks
from counter.line_counter import LineCounter


def process_video(video_path, out_video, out_csv, line=((0.0, 0.5), (1.0, 0.5)),
                  classes=None, model_path="yolo26n.pt", detector=None, progress=None):
    """Run detection + tracking + counting on one video.

    line: two points given as FRACTIONS of width/height, e.g. ((0, 0.5), (1, 0.5))
          is a horizontal line across the middle. Fractions work for any resolution.
    classes: list of class names to count, e.g. ["car", "truck"]. None = all five.
    detector: reuse an already-loaded Detector (saves reloading the model in the app).
    progress: optional function(fraction_done) for progress bars.
    Returns a summary dict.
    """
    name_to_id = {v: k for k, v in CLASSES.items()}
    class_names = classes or list(CLASSES.values())
    class_ids = [name_to_id[c] for c in class_names]

    if detector is None:
        detector = Detector(model_path, classes=class_ids)
    else:
        detector.class_ids = class_ids
        detector.reset()  # new video: forget tracks from the previous one

    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert line fractions to pixel coordinates.
    a = (int(line[0][0] * w), int(line[0][1] * h))
    b = (int(line[1][0] * w), int(line[1][1] * h))
    counter = LineCounter(a, b)
    trails = TrailBuffer(length=20)

    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx, t_start = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracks = detector.track(frame)
        counter.update(tracks, frame_idx, fps)
        trails.update(tracks)
        trails.forget({t.track_id for t in tracks})

        draw_line(frame, a, b)
        draw_tracks(frame, tracks, counter.counted, trails)
        draw_panel(frame, counter.counts, class_names)
        writer.write(frame)

        frame_idx += 1
        if progress and total:
            progress(frame_idx / total)

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t_start

    # CSV: one row per crossing event.
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["track_id", "class", "direction", "frame", "time_s"])
        for e in counter.events:
            wr.writerow([e.track_id, e.cls_name, e.direction, e.frame, e.time_s])

    return {
        "frames": frame_idx,
        "resolution": f"{w}x{h}",
        "processing_fps": round(frame_idx / elapsed, 2),  # whole pipeline incl. drawing + writing
        "totals": counter.totals(),
        "by_class": {f"{c} {d}": n for (c, d), n in sorted(counter.counts.items())},
        "events": len(counter.events),
    }
