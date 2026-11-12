"""Benchmark every model format on the same video, on CPU.

For each format we measure:
  1. Model-only speed: run detection on the same 150 frames (already in memory,
     so disk reading isn't timed), repeated 3 times; we report the MEDIAN run.
  2. Full-pipeline speed: detect + track + count + draw + write on the whole video
     (same code as the app), plus the final IN/OUT counts.
Comparing counts against PyTorch shows whether the faster formats (especially
INT8) change the actual results.

Usage (from the project folder, venv active, after scripts/export_models.py):
    python scripts/benchmark.py
Saves results/benchmark.csv and results/benchmark_env.json.
"""
import csv
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from counter.detector import CLASSES, Detector  # noqa: E402
from counter.pipeline import process_video  # noqa: E402

VIDEO = ROOT / "samples" / "highway.mp4"
N_FRAMES, REPEATS, WARMUP = 150, 3, 10

FORMATS = [
    # (label,             engine,          precision, path)
    ("PyTorch",           "PyTorch",       "FP32", ROOT / "yolo26n.pt"),
    ("ONNX FP32",         "ONNX Runtime",  "FP32", ROOT / "models" / "yolo26n_fp32.onnx"),
    ("ONNX INT8",         "ONNX Runtime",  "INT8", ROOT / "models" / "yolo26n_int8.onnx"),
    ("OpenVINO FP32",     "OpenVINO",      "FP32", ROOT / "models" / "yolo26n_fp32_openvino_model"),
    ("OpenVINO INT8",     "OpenVINO",      "INT8", ROOT / "models" / "yolo26n_int8_openvino_model"),
]


def load_frames(n):
    cap = cv2.VideoCapture(str(VIDEO))
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def model_only_fps(det, frames):
    """Median FPS of plain detection (no tracking) over several repeats."""
    predict = lambda f: det.model.predict(f, classes=det.class_ids, conf=det.conf,  # noqa: E731
                                          device=det.device, verbose=False)
    for f in frames[:WARMUP]:  # warm-up runs are not timed (memory allocation, graph compile)
        predict(f)
    runs = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for f in frames:
            predict(f)
        runs.append(len(frames) / (time.perf_counter() - t0))
    return statistics.median(runs), runs


def size_mb(path):
    files = path.rglob("*") if path.is_dir() else [path]
    return round(sum(f.stat().st_size for f in files if f.is_file()) / 1e6, 2)


def cpu_name():
    try:
        import openvino as ov
        return ov.Core().get_property("CPU", "FULL_DEVICE_NAME")
    except Exception:
        return platform.processor()


def main():
    frames = load_frames(N_FRAMES)
    print(f"Video {VIDEO.name}: {len(frames)} frames in memory for model-only timing\n")
    rows, base_counts = [], None
    out_dir = Path(tempfile.mkdtemp(prefix="bench_"))

    for label, engine, precision, path in FORMATS:
        if not path.exists():
            print(f"SKIP {label}: {path.relative_to(ROOT)} not found (run scripts/export_models.py)")
            continue
        print(f"=== {label} ===")
        det = Detector(str(path))
        fps, runs = model_only_fps(det, frames)
        print(f"  model-only: median {fps:.1f} FPS  (runs: {', '.join(f'{r:.1f}' for r in runs)})")

        # Fresh detector so tracking starts clean, then run the real app pipeline.
        summary = process_video(VIDEO, out_dir / f"{label}.mp4", out_dir / f"{label}.csv",
                                detector=Detector(str(path)), classes=list(CLASSES.values()))
        counts = summary["totals"]
        if base_counts is None:
            base_counts = counts  # PyTorch is the reference
        print(f"  pipeline: {summary['processing_fps']} FPS, counts IN {counts['IN']} OUT {counts['OUT']}")

        rows.append({
            "format": label,
            "engine": engine,
            "precision": precision,
            "model_size_mb": size_mb(path),
            "model_only_fps": round(fps, 1),
            "model_only_ms_per_frame": round(1000 / fps, 1),
            "pipeline_fps": summary["processing_fps"],
            "count_in": counts["IN"],
            "count_out": counts["OUT"],
            # abs() so +1 IN and -1 OUT don't cancel out and hide a difference
            "count_diff_vs_pytorch": abs(counts["IN"] - base_counts["IN"]) + abs(counts["OUT"] - base_counts["OUT"]),
        })

    base_fps = rows[0]["model_only_fps"]
    for r in rows:
        r["speedup_vs_pytorch"] = round(r["model_only_fps"] / base_fps, 2)

    with open(ROOT / "results" / "benchmark.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    import onnxruntime
    import openvino
    import torch
    import ultralytics
    env = {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
        "openvino": openvino.__version__, "ultralytics": ultralytics.__version__,
        "video": f"{VIDEO.name} ({len(frames)} frames for model-only timing, full video for pipeline)",
        "imgsz": 640, "repeats": REPEATS, "warmup_frames": WARMUP,
    }
    (ROOT / "results" / "benchmark_env.json").write_text(json.dumps(env, indent=2))

    print(f"\n{'format':<15}{'size MB':>8}{'model FPS':>11}{'ms/frame':>10}{'speedup':>9}{'pipeline FPS':>14}{'IN':>4}{'OUT':>5}{'diff':>6}")
    for r in rows:
        print(f"{r['format']:<15}{r['model_size_mb']:>8}{r['model_only_fps']:>11}{r['model_only_ms_per_frame']:>10}"
              f"{r['speedup_vs_pytorch']:>8}x{r['pipeline_fps']:>14}{r['count_in']:>4}{r['count_out']:>5}{r['count_diff_vs_pytorch']:>6}")
    print(f"\nCPU: {env['cpu']}  ({env['logical_cores']} logical cores)")
    print("Saved results/benchmark.csv and results/benchmark_env.json")


if __name__ == "__main__":
    main()
