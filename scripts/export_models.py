"""Export YOLO26n to faster CPU formats: ONNX and OpenVINO, each in FP32 and INT8.

FP32 = weights stored as 32-bit floats (same maths as PyTorch, different engine).
INT8 = weights and activations squeezed into 8-bit integers. Smaller and usually
faster, but slightly less precise. To choose good integer ranges, INT8 needs to
see real images first ("calibration"); we use the 128 images of COCO128.

Usage (from the project folder, venv active):
    python scripts/export_models.py
Exports go to models/ (git-ignored; they can always be rebuilt with this script).
"""
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SOURCE = "yolo26n.pt"
IMGSZ = 640               # same input size as the PyTorch model, for a fair comparison
CALIB_DATA = "coco128.yaml"  # 128 COCO images (cars, people, buses...) for INT8 calibration

EXPORTS = [
    # (output name,               format,     extra export args)
    ("yolo26n_fp32.onnx",            "onnx",     {}),
    ("yolo26n_int8.onnx",            "onnx",     {"quantize": 8, "data": CALIB_DATA, "fraction": 1.0}),
    ("yolo26n_fp32_openvino_model",  "openvino", {}),
    ("yolo26n_int8_openvino_model",  "openvino", {"quantize": 8, "data": CALIB_DATA, "fraction": 1.0}),
]


def size_mb(path):
    path = Path(path)
    files = path.rglob("*") if path.is_dir() else [path]
    return round(sum(f.stat().st_size for f in files if f.is_file()) / 1e6, 2)


def main():
    MODELS.mkdir(exist_ok=True)
    sizes = {"yolo26n.pt (PyTorch)": size_mb(ROOT / SOURCE)}

    for name, fmt, extra in EXPORTS:
        print(f"\n=== Exporting {name} ===")
        # Fresh model object each time: export can modify the model in place.
        exported = Path(YOLO(ROOT / SOURCE).export(format=fmt, imgsz=IMGSZ, device="cpu", **extra))
        target = MODELS / name
        if target.exists():  # replace an older export of the same name
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(exported), target)  # Ultralytics writes next to the .pt; move it into models/
        sizes[name] = size_mb(target)
        print(f"Saved {target.relative_to(ROOT)} ({sizes[name]} MB)")

    (ROOT / "results" / "model_sizes.json").write_text(json.dumps(sizes, indent=2))
    print("\nModel sizes (MB):", json.dumps(sizes, indent=2))


if __name__ == "__main__":
    main()
