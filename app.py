"""Gradio web app: upload a video, count vehicles/people crossing a line.

Run (from the project folder, venv active):
    python app.py        then open http://127.0.0.1:7860
"""
import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

from counter.detector import CLASSES, Detector
from counter.draw import draw_line
from counter.pipeline import first_frame, process_video

ROOT = Path(__file__).resolve().parent
SAMPLES = sorted(str(p) for p in (ROOT / "samples").glob("*.mp4"))
CLASS_NAMES = list(CLASSES.values())  # person, car, motorcycle, bus, truck

# Fastest CPU model from our benchmark (results/benchmark.csv): OpenVINO INT8.
# Falls back to the PyTorch model if the export isn't there (run scripts/export_models.py).
INT8_MODEL = ROOT / "models" / "yolo26n_int8_openvino_model"
MODEL_PATH = os.getenv("MODEL_PATH") or (str(INT8_MODEL) if INT8_MODEL.exists() else "yolo26n.pt")
MODEL_LABEL = "OpenVINO INT8" if "int8_openvino" in MODEL_PATH else Path(MODEL_PATH).name

# Limits keep the free shared CPU responsive. Change with environment variables.
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "20"))
MAX_WIDTH = int(os.getenv("MAX_WIDTH", "960"))

# Load the model ONCE when the app starts, not on every click.
DETECTOR = Detector(MODEL_PATH)


def line_points(left_y, right_y):
    """Line from the left edge to the right edge, heights as fractions (0 = top, 1 = bottom)."""
    return ((0.0, left_y), (1.0, right_y))


def preview(video, left_y, right_y):
    """Show the first frame with the counting line, so the user can place it before running."""
    if not video:
        return None
    frame = first_frame(video)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    (ax, ay), (bx, by) = line_points(left_y, right_y)
    draw_line(frame, (int(ax * w), int(ay * h)), (int(bx * w), int(by * h)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV is BGR, Gradio expects RGB


def run(video, classes, left_y, right_y, progress=gr.Progress()):
    if not video:
        raise gr.Error("Please upload a video or pick a sample first.")
    if not classes:
        raise gr.Error("Please select at least one class to count.")

    out_dir = Path(tempfile.mkdtemp(prefix="traffic_"))  # unique folder per run
    stem = Path(video).stem
    out_video, out_csv = out_dir / f"{stem}_counted.mp4", out_dir / f"{stem}_events.csv"

    progress(0, desc="Processing video...")
    summary = process_video(video, out_video, out_csv, line=line_points(left_y, right_y),
                            classes=classes, detector=DETECTOR,
                            progress=lambda f: progress(f, desc="Detecting, tracking, counting..."),
                            max_seconds=MAX_SECONDS, max_width=MAX_WIDTH)

    # Summary table: one row per selected class + total.
    by = summary["by_class"]
    rows = [[c, by.get(f"{c} IN", 0), by.get(f"{c} OUT", 0)] for c in classes]
    rows.append(["TOTAL", summary["totals"]["IN"], summary["totals"]["OUT"]])
    table = pd.DataFrame(rows, columns=["class", "IN", "OUT"])

    info = (f"**{summary['frames']} frames** ({summary['seconds_processed']} s) at {summary['resolution']} · "
            f"processed at **{summary['processing_fps']} FPS** on CPU with {MODEL_LABEL} · "
            f"{summary['events']} crossing events")
    if summary["trimmed"]:
        info += f"\n\n⚠️ Video was longer than {MAX_SECONDS:g} s, so only the first {MAX_SECONDS:g} s were processed."
    return str(out_video), table, str(out_csv), info


with gr.Blocks(title="Smart Traffic & People Counter") as demo:
    gr.Markdown(
        "# 🚦 Smart Traffic & People Counter\n"
        "Detects cars, bikes, buses, trucks and people with **YOLO26n**, tracks each one with "
        "**ByteTrack**, and counts how many cross the yellow line in each direction. Runs on CPU."
    )
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Input video", sources=["upload"], height=280)
            if SAMPLES:
                gr.Examples(SAMPLES, inputs=video_in, label="Or pick a sample video")
            classes_in = gr.CheckboxGroup(CLASS_NAMES, value=CLASS_NAMES, label="Classes to count")
            with gr.Row():
                left_in = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Line height — left edge")
                right_in = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Line height — right edge")
            line_preview = gr.Image(label="Line preview (first frame)", height=220, interactive=False)
            run_btn = gr.Button("Count", variant="primary")
        with gr.Column(scale=1):
            video_out = gr.Video(label="Result", height=360, autoplay=True)
            info_out = gr.Markdown()
            table_out = gr.Dataframe(label="Counts", interactive=False)
            csv_out = gr.File(label="Download crossing events (CSV)")

    gr.Markdown(
        "IN = crossing the line moving **down** the image (towards the camera), OUT = moving **up**.  \n"
        f"Model: YOLO26n ({MODEL_LABEL}). To keep the free CPU fast, only the first "
        f"**{MAX_SECONDS:g} s** of a video are processed and frames are resized to at most **{MAX_WIDTH} px** wide."
    )

    # Update the preview whenever the video or a slider changes.
    for comp in (video_in, left_in, right_in):
        comp.change(preview, [video_in, left_in, right_in], line_preview)
    run_btn.click(run, [video_in, classes_in, left_in, right_in], [video_out, table_out, csv_out, info_out])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
