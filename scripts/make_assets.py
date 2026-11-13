"""Make the README / portfolio images in assets/ from real outputs (nothing hand-edited).

  assets/demo.gif           ~6 s of the Python app's annotated output video
  assets/python_output.png  one annotated frame from that video
  assets/benchmark.png      model-only CPU speed per format, from results/benchmark.csv
  assets/web_app_frame.png  the in-browser app's canvas after 165 frames (headless Edge)

Usage (from the project folder, venv active, after scripts/run_video.py and
scripts/benchmark.py have been run):
    python scripts/make_assets.py
"""
import base64
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
VIDEO = ROOT / "results" / "highway_counted.mp4"  # made by scripts/run_video.py

# Chart colours (one hue for one measure; text in neutral inks, not the bar colour).
BLUE, SURFACE, INK, INK_2 = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e"


def read_frames(start, end, step, width):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for i in range(start, end):
        ok, f = cap.read()
        if not ok:
            break
        if (i - start) % step == 0:
            h = int(f.shape[0] * width / f.shape[1])
            frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def make_gif():
    # Frames 100-250 (6 s) contain the busiest part: 6 cars crossing. Every 2nd frame at 480 px,
    # reduced to 128 colours: keeps the GIF small enough for a README.
    frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in read_frames(100, 250, 2, 480)]
    out = ASSETS / "demo.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    print(f"{out.relative_to(ROOT)}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


def make_frame_png():
    img = read_frames(165, 166, 1, 1280)[0]
    out = ASSETS / "python_output.png"
    img.save(out, optimize=True)
    print(f"{out.relative_to(ROOT)}")


def make_benchmark_chart():
    rows = list(csv.DictReader(open(ROOT / "results" / "benchmark.csv")))
    rows.sort(key=lambda r: float(r["model_only_fps"]))
    labels = [r["format"] for r in rows]
    fps = [float(r["model_only_fps"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(labels, fps, color=BLUE, height=0.55)
    for bar, v in zip(bars, fps):  # direct labels at the bar ends, in text ink
        ax.text(v + 0.4, bar.get_y() + bar.get_height() / 2, f"{v:g} FPS", va="center", color=INK, fontsize=10)
    ax.set_xlabel("frames per second (model only, higher is better)", color=INK_2)
    ax.set_title("YOLO26n on a laptop CPU (Intel i5-10300H), same video, 640 px input",
                 color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK_2)
    ax.tick_params(axis="y", colors=INK, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d4")
    ax.grid(axis="x", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(fps) * 1.18)
    fig.tight_layout()
    out = ASSETS / "benchmark.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"{out.relative_to(ROOT)}")


def make_web_frame():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_web  # noqa: E402  (serves web_dist and drives headless Edge)
    r = check_web.run_in_browser(check_web.build(), "snapshot=165", timeout=400)
    out = ASSETS / "web_app_frame.png"
    out.write_bytes(base64.b64decode(r["snapshot"].split(",")[1]))
    print(f"{out.relative_to(ROOT)}  ({r['status']})")


if __name__ == "__main__":
    make_gif()
    make_frame_png()
    make_benchmark_chart()
    make_web_frame()
