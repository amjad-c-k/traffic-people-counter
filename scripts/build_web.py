"""Assemble the in-browser app into web_dist/ (what gets served and deployed).

web/ holds the code; the model and sample video live elsewhere in the repo
(models/ is git-ignored). This copies everything into one folder:
    web_dist/index.html, js/, model/yolo26n.onnx, samples/highway.mp4

Cache-busting: a short version code (hash of the code + model) replaces __VERSION__ and is
added to every JS import (e.g. "./tracker.js?v=3f2a1c9b"), so a browser can never mix an
old cached file with a new one after a deploy. The version is shown in the page footer.

Usage (from the project folder, venv active):
    python scripts/build_web.py                                # dynamic-shape model (default)
    python scripts/build_web.py models/yolo26n_fp32.onnx       # a fixed 640x640 model
"""
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "web_dist"
DEFAULT_MODEL = ROOT / "models" / "yolo26n_dynamic.onnx"


def build(model=DEFAULT_MODEL):
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

    # Version = hash of all code files + the model, so it changes whenever anything changes.
    h = hashlib.sha256()
    for f in sorted(DIST.rglob("*")):
        if f.is_file() and f.suffix in {".html", ".js", ".onnx"}:
            h.update(f.read_bytes())
    version = h.hexdigest()[:8]
    # Separate code for the model, so the 10 MB file is only re-downloaded when it changes.
    model_version = hashlib.sha256((DIST / "model" / "yolo26n.onnx").read_bytes()).hexdigest()[:8]
    for f in [DIST / "index.html", *(DIST / "js").glob("*.js")]:
        text = f.read_text(encoding="utf-8").replace("__MODEL_VERSION__", model_version).replace("__VERSION__", version)
        text = re.sub(r'(from\s+"\./[\w.]+\.js)(")', rf"\1?v={version}\2", text)  # versioned imports
        f.write_text(text, encoding="utf-8")

    print(f"Built {DIST.relative_to(ROOT)} with model {model.name}, version {version}, model version {model_version}")
    return DIST


if __name__ == "__main__":
    build(sys.argv[1]) if len(sys.argv) > 1 else build()
