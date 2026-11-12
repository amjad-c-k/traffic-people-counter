"""Record what the Python tracker does on the sample video, as a reference for the JS port.

For every frame it saves:
  - the detections the tracker received (box, confidence, class),
  - the tracks Ultralytics' ByteTrack returned (ID + Kalman-smoothed box),
and at the end the line-crossing events from counter/line_counter.py.
The JS tests (tests/tracker_parity.test.mjs) feed the SAME detections into web/js/tracker.js
and check that they get the same IDs, boxes and crossing events.

Usage (from the project folder, venv active):
    python scripts/make_tracking_reference.py
Saves tests/fixtures/tracking_reference.json.
"""
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from counter.detector import CLASSES, Track  # noqa: E402
from counter import lap_fallback  # noqa: E402
from counter.line_counter import LineCounter  # noqa: E402

lap_fallback.install()
MODEL = ROOT / "models" / "yolo26n_fp32.onnx"  # same model file the browser uses
VIDEO = ROOT / "samples" / "highway.mp4"
OUT = ROOT / "tests" / "fixtures" / "tracking_reference.json"
ARGS = dict(classes=list(CLASSES), conf=0.25, iou=0.7, device="cpu", verbose=False)

r2 = lambda xs: [round(float(x), 3) for x in xs]  # noqa: E731  (keeps the JSON small)


def main():
    # Two separate model objects: tracking registers callbacks on its predictor, so plain
    # predictions must come from a different instance to stay untouched.
    detector = YOLO(MODEL, task="detect")
    tracker = YOLO(MODEL, task="detect")

    cap = cv2.VideoCapture(str(VIDEO))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    a, b = (0, int(0.5 * h)), (w, int(0.5 * h))
    counter = LineCounter(a, b)

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        d = detector.predict(frame, **ARGS)[0].boxes
        dets = [{"box": r2(x), "conf": round(float(c), 4), "classId": int(k)}
                for x, c, k in zip(d.xyxy.tolist(), d.conf.tolist(), d.cls.tolist())]

        t = tracker.track(frame, persist=True, tracker="bytetrack.yaml", **ARGS)[0].boxes
        tracks = []
        if t.id is not None:
            for tid, x, k in zip(t.id.int().tolist(), t.xyxy.tolist(), t.cls.int().tolist()):
                tracks.append({"trackId": tid, "box": r2(x), "classId": k})
        counter.update([Track(tr["trackId"], CLASSES[tr["classId"]], 0, tuple(tr["box"])) for tr in tracks],
                       len(frames), fps)
        frames.append({"dets": dets, "tracks": tracks})
    cap.release()

    ref = {
        "video": VIDEO.name, "model": MODEL.name, "fps": fps, "width": w, "height": h,
        "line": [a, b],
        "frames": frames,
        "events": [{"trackId": e.track_id, "className": e.cls_name, "direction": e.direction,
                    "frame": e.frame} for e in counter.events],
        "totals": counter.totals(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ref, separators=(",", ":")))
    ids = {tr["trackId"] for f in frames for tr in f["tracks"]}
    print(f"{len(frames)} frames, {len(ids)} track IDs, totals {ref['totals']}, "
          f"{len(ref['events'])} events -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
