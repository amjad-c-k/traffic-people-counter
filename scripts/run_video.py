"""Process a video end to end: detect, track, count, draw.

Saves the annotated video, a CSV of crossing events and a JSON summary to results/.

Usage (from the project folder, venv active):
    python scripts/run_video.py
    python scripts/run_video.py --video samples/highway.mp4 --line-y 0.45 --classes car truck
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from counter.pipeline import process_video  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--video", default=str(ROOT / "samples" / "highway.mp4"))
parser.add_argument("--line-y", type=float, default=0.5, help="horizontal line height as fraction 0..1")
parser.add_argument("--classes", nargs="*", default=None, help="e.g. car truck person (default: all five)")
args = parser.parse_args()

stem = Path(args.video).stem
out_video = ROOT / "results" / f"{stem}_counted.mp4"
out_csv = ROOT / "results" / f"{stem}_events.csv"

summary = process_video(args.video, out_video, out_csv,
                        line=((0.0, args.line_y), (1.0, args.line_y)), classes=args.classes)
(ROOT / "results" / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"Saved: {out_video.relative_to(ROOT)}, {out_csv.relative_to(ROOT)}")
