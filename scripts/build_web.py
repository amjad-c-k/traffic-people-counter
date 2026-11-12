"""Assemble the in-browser app into web_dist/ (what gets served and deployed).

web/ holds the code; the model and sample video live elsewhere in the repo
(models/ is git-ignored). This copies everything into one folder:
    web_dist/index.html, js/, model/yolo26n.onnx, samples/highway.mp4

Usage (from the project folder, venv active):
    python scripts/build_web.py                       # FP32 model (default)
    python scripts/build_web.py models/yolo26n_int8.onnx
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "web_dist"


def build(model=ROOT / "models" / "yolo26n_fp32.onnx"):
    model = Path(model)
    if not model.exists():
        sys.exit(f"Missing {model} - run scripts/export_models.py first.")
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(ROOT / "web", DIST)
    (DIST / "model").mkdir()
    shutil.copy2(model, DIST / "model" / "yolo26n.onnx")
    (DIST / "samples").mkdir()
    shutil.copy2(ROOT / "samples" / "highway.mp4", DIST / "samples" / "highway.mp4")
    print(f"Built {DIST.relative_to(ROOT)} with model {model.name}")
    return DIST


if __name__ == "__main__":
    build(sys.argv[1]) if len(sys.argv) > 1 else build()
