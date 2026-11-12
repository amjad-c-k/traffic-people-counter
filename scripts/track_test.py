"""Tracking test: run YOLO + ByteTrack on a video and check that IDs stay stable.

Records how many frames each track ID lives; many very short-lived IDs would
point to ID switches or flickering detections.

Usage (from the project folder, venv active):
    python scripts/track_test.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so "counter" can be imported when run as a script
from counter.detector import Detector  # noqa: E402

VIDEO = ROOT / "samples" / "highway.mp4"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    det = Detector("yolo26n.pt")
    cap = cv2.VideoCapture(str(VIDEO))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    out_path = RESULTS / "tracking_demo.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    frames_seen = defaultdict(list)  # track_id -> list of frame numbers it appeared in
    track_class = {}                 # track_id -> class name
    frame_idx, track_time = 0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        tracks = det.track(frame)
        track_time += time.perf_counter() - t0

        for t in tracks:
            frames_seen[t.track_id].append(frame_idx)
            track_class[t.track_id] = t.cls_name
            # Minimal drawing just to check IDs by eye (full overlay lives in counter/draw.py).
            x1, y1, x2, y2 = map(int, t.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"#{t.track_id} {t.cls_name}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        if frame_idx % 50 == 0:
            ids = ", ".join(f"#{t.track_id}" for t in tracks) or "-"
            print(f"Frame {frame_idx:3d}: {len(tracks)} tracks  IDs: {ids}")
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # How long did each ID live? Short-lived IDs often mean the tracker lost an
    # object and gave it a new ID (an "ID switch"), or a brief false detection.
    print(f"\n{'ID':>4} {'class':<11} {'first':>5} {'last':>5} {'frames':>6}")
    lifetimes = {}
    for tid in sorted(frames_seen):
        f = frames_seen[tid]
        lifetimes[tid] = len(f)
        print(f"{tid:>4} {track_class[tid]:<11} {f[0]:>5} {f[-1]:>5} {len(f):>6}")

    short = [tid for tid, n in lifetimes.items() if n < 10]
    stats = {
        "video": VIDEO.name,
        "frames": frame_idx,
        "unique_ids": len(lifetimes),
        "ids_by_class": {c: sum(1 for t in track_class.values() if t == c) for c in set(track_class.values())},
        "avg_track_length_frames": round(sum(lifetimes.values()) / max(len(lifetimes), 1), 1),
        "short_tracks_under_10_frames": len(short),
        "avg_tracking_fps": round(frame_idx / track_time, 2),
    }
    (RESULTS / "tracking_stats.json").write_text(json.dumps(stats, indent=2))
    print("\n" + json.dumps(stats, indent=2))
    print(f"Saved video -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
