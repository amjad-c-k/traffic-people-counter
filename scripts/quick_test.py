"""Milestone 1: run YOLO on one image and one short video, print detections.

Usage (from the project folder, venv active):
    python scripts/quick_test.py
"""
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "yolo26n.pt"  # smallest "nano" model; auto-downloads on first run
IMAGE = "https://ultralytics.com/images/bus.jpg"  # Ultralytics' standard sample image
VIDEO = ROOT / "samples" / "highway.mp4"
MAX_VIDEO_FRAMES = 150  # enough for a speed check without waiting long


def print_detections(result):
    """Print one line per detected object: class, confidence, box corners."""
    names = result.names  # id -> class name, e.g. {0: 'person', 2: 'car', ...}
    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # top-left and bottom-right corners (pixels)
        print(f"  {names[cls_id]:<12} conf={conf:.2f}  box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")


def run_image(model):
    print(f"\n=== IMAGE: {IMAGE} ===")
    result = model(IMAGE, verbose=False)[0]
    print(f"Found {len(result.boxes)} objects:")
    print_detections(result)
    out = RESULTS / "m1_image_detections.jpg"
    cv2.imwrite(str(out), result.plot())  # plot() draws boxes + labels on the image
    print(f"Saved annotated image -> {out.relative_to(ROOT)}")


def run_video(model):
    print(f"\n=== VIDEO: {VIDEO.relative_to(ROOT)} ===")
    cap = cv2.VideoCapture(str(VIDEO))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height}, {fps_in:.1f} fps, {total} frames")

    out_path = RESULTS / "m1_video_detections.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    # Warm-up: the first inference is slow (memory allocation), so don't time it.
    ok, frame = cap.read()
    model(frame, verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx, infer_time = 0, 0.0
    while frame_idx < MAX_VIDEO_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        result = model(frame, verbose=False)[0]
        infer_time += time.perf_counter() - t0

        if frame_idx % 30 == 0:  # print a sample every ~1 second of video
            print(f"Frame {frame_idx}: {len(result.boxes)} objects")
            print_detections(result)
        writer.write(result.plot())
        frame_idx += 1

    cap.release()
    writer.release()
    avg_fps = frame_idx / infer_time
    print(f"\nProcessed {frame_idx} frames. Average model speed: {avg_fps:.1f} FPS (inference only, CPU)")
    print(f"Saved annotated video -> {out_path.relative_to(ROOT)}")
    return {"model": MODEL_NAME, "video": VIDEO.name, "resolution": f"{width}x{height}",
            "frames": frame_idx, "avg_inference_fps": round(avg_fps, 2)}


if __name__ == "__main__":
    model = YOLO(MODEL_NAME)
    run_image(model)
    stats = run_video(model)
    (RESULTS / "m1_quick_test.json").write_text(json.dumps(stats, indent=2))
    print(f"Saved stats -> results/m1_quick_test.json")
