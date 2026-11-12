"""Full pipeline: video in -> tracked, counted, annotated video + CSV out.

Used by both scripts/run_video.py (command line) and app.py (Gradio).
"""
import csv
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg

from counter.detector import CLASSES, Detector
from counter.draw import TrailBuffer, draw_line, draw_panel, draw_tracks
from counter.line_counter import LineCounter


def to_h264(src, dst):
    """Re-encode a video to H.264, the codec web browsers can play.

    OpenCV's pip build can only write 'mp4v', which browsers usually refuse.
    imageio-ffmpeg ships its own ffmpeg program, so this works on Windows and Linux.
    Returns True on success.
    """
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(dst)]  # faststart: video can start playing before fully downloaded
    return subprocess.run(cmd).returncode == 0


def first_frame(video_path):
    """Read the first frame of a video (for the line preview)."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def process_video(video_path, out_video, out_csv, line=((0.0, 0.5), (1.0, 0.5)),
                  classes=None, model_path="yolo26n.pt", detector=None, progress=None,
                  max_seconds=None, max_width=None):
    """Run detection + tracking + counting on one video.

    line: two points given as FRACTIONS of width/height, e.g. ((0, 0.5), (1, 0.5))
          is a horizontal line across the middle. Fractions work for any resolution.
    classes: list of class names to count, e.g. ["car", "truck"]. None = all five.
    detector: reuse an already-loaded Detector (saves reloading the model in the app).
    progress: optional function(fraction_done) for progress bars.
    max_seconds: only process the first N seconds (keeps a shared CPU responsive).
    max_width: shrink wider frames to this width. YOLO resizes to 640 px internally
               anyway, so detection barely changes, but decoding/drawing/encoding get cheaper.
    Returns a summary dict.
    """
    name_to_id = {v: k for k, v in CLASSES.items()}
    class_names = classes or list(CLASSES.values())
    class_ids = [name_to_id[c] for c in class_names]

    if detector is None:
        detector = Detector(model_path, classes=class_ids)
    else:
        detector.class_ids = class_ids
        detector.reset()  # new video: forget tracks from the previous one

    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_size = f"{w}x{h}"

    # Optional limits: stop after max_seconds, shrink frames wider than max_width.
    max_frames = int(max_seconds * fps) if max_seconds else None
    if max_frames and total:
        total = min(total, max_frames)
    scale = max_width / w if (max_width and w > max_width) else 1.0
    if scale < 1.0:
        w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2  # even sizes: H.264 requires them

    # Convert line fractions to pixel coordinates.
    a = (int(line[0][0] * w), int(line[0][1] * h))
    b = (int(line[1][0] * w), int(line[1][1] * h))
    counter = LineCounter(a, b)
    trails = TrailBuffer(length=20)

    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    raw_video = out_video.with_name(out_video.stem + "_raw.mp4")  # mp4v first, converted at the end
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx, t_start = 0, time.perf_counter()
    while max_frames is None or frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)  # INTER_AREA: best for shrinking
        tracks = detector.track(frame)
        counter.update(tracks, frame_idx, fps)
        trails.update(tracks)
        trails.forget({t.track_id for t in tracks})

        draw_line(frame, a, b)
        draw_tracks(frame, tracks, counter.counted, trails)
        draw_panel(frame, counter.counts, class_names)
        writer.write(frame)

        frame_idx += 1
        if progress and total:
            progress(frame_idx / total)

    elapsed = time.perf_counter() - t_start  # measured before conversion: pipeline speed only
    trimmed = max_frames is not None and frame_idx >= max_frames and cap.read()[0]  # frames were left over?
    cap.release()
    writer.release()

    if to_h264(raw_video, out_video):
        raw_video.unlink()
    else:  # conversion failed: keep the mp4v file so the user still gets a video
        raw_video.replace(out_video)

    # CSV: one row per crossing event.
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["track_id", "class", "direction", "frame", "time_s"])
        for e in counter.events:
            wr.writerow([e.track_id, e.cls_name, e.direction, e.frame, e.time_s])

    return {
        "frames": frame_idx,
        "seconds_processed": round(frame_idx / fps, 1),
        "trimmed": bool(trimmed),  # True if the video was longer than max_seconds
        "original_resolution": orig_size,
        "resolution": f"{w}x{h}",
        "processing_fps": round(frame_idx / elapsed, 2),  # whole pipeline incl. drawing + writing
        "totals": counter.totals(),
        "by_class": {f"{c} {d}": n for (c, d), n in sorted(counter.counts.items())},
        "events": len(counter.events),
    }
