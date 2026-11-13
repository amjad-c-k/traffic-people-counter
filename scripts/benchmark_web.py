"""Measure the in-browser app's speed for each backend/model setup (headless Microsoft Edge).

Each row runs scripts/check_web.py's detection check (7 frames of the sample video, after a
warm-up frame) and records the time per frame and how many boxes match Python.
Speeds depend on the computer: these numbers are for the machine in results/benchmark_env.json.

Usage (from the project folder, venv active):
    python scripts/benchmark_web.py
Saves results/web_benchmark.csv.
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

SETUPS = [
    # (label, model, backend, extra Edge flags, note)
    ("WebGPU, square 640x640 input", "models/yolo26n_fp32.onnx", "webgpu", "", "first browser version"),
    ("WebGPU, rect 640x384 input", "models/yolo26n_dynamic.onnx", "webgpu", "", "used on the live site"),
    ("CPU (WASM, 1 thread), rect", "models/yolo26n_dynamic.onnx", "wasm", "", "fallback without a GPU"),
    ("WebGPU on software GPU, rect", "models/yolo26n_dynamic.onnx", "webgpu", "--disable-gpu",
     "graphics acceleration off: the app now switches to CPU instead"),
    ("WebNN, rect", "models/yolo26n_dynamic.onnx", "webnn", "--enable-features=WebMachineLearningNeuralNetwork",
     "experimental flag in Edge 157; used automatically once browsers enable it"),
]


def main():
    rows = []
    for label, model, ep, flags, note in SETUPS:
        print(f"=== {label}")
        env = {**os.environ, "EDGE_FLAGS": flags}
        res = subprocess.run([PY, str(ROOT / "scripts" / "check_web.py"), model, ep], env=env,
                             capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if res.returncode != 0:
            print("  failed:", (res.stdout + res.stderr).strip().splitlines()[-1:])
            continue
        r = json.loads((ROOT / "results" / "web_check.json").read_text())
        parts = r.get("browser_detect_parts_ms") or {}
        per_frame = r["browser_detect_ms"] + r["browser_seek_ms"]
        rows.append({
            "setup": label, "backend_used": r["execution_provider"],
            "model_ms": parts.get("model"), "detect_ms": r["browser_detect_ms"], "seek_ms": r["browser_seek_ms"],
            "approx_fps_detection_only": round(1000 / per_frame, 1),
            "boxes_matched_vs_python": f"{r['total_matched']}/{r['total_python_boxes']}",
            "note": note,
        })
        print(f"  {rows[-1]}")

    out = ROOT / "results" / "web_benchmark.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
