"""Check that the browser version detects the same objects as Python.

1. Builds web_dist/ and serves it on a local web server.
2. Opens the page in headless Microsoft Edge with ?check=<frames>; the page runs
   detection on those frames and POSTs its boxes back to this server.
3. Runs the same ONNX model in Python (Ultralytics) on the same frames.
4. Matches boxes between the two by IoU and reports how well they agree.

Usage (from the project folder, venv active):
    python scripts/check_web.py
    python scripts/check_web.py models/yolo26n_int8.onnx
    python scripts/check_web.py models/yolo26n_fp32.onnx webgpu          # GPU backend
    python scripts/check_web.py models/yolo26n_fp32.onnx wasm isolate    # multi-threaded CPU
    python scripts/check_web.py track     # whole pipeline: crossing events vs Python
Saves results/web_check.json (or results/web_tracking_check.json for "track").
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_web import build  # noqa: E402
from counter.detector import CLASSES  # noqa: E402

FRAMES = [0, 60, 120, 150, 180, 240, 300]
IMGSZ = int(os.environ.get("IMGSZ", 640))  # model input size to test (e.g. IMGSZ=480)
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
PORT = 8765
report = {}
got_report = threading.Event()


class Handler(SimpleHTTPRequestHandler):
    isolate = False  # send cross-origin isolation headers (enables multi-threaded WASM)

    def end_headers(self):
        if self.isolate:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_POST(self):  # the page sends its results here
        body = self.rfile.read(int(self.headers["Content-Length"]))
        report.update(json.loads(body))
        self.send_response(200)
        self.end_headers()
        got_report.set()

    def log_message(self, *args):  # keep the console quiet
        pass


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def python_detections(model_path):
    from ultralytics import YOLO
    model = YOLO(model_path, task="detect")
    cap = cv2.VideoCapture(str(ROOT / "samples" / "highway.mp4"))
    out = {}
    for n in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        r = model.predict(frame, classes=list(CLASSES), conf=0.25, iou=0.7, imgsz=IMGSZ, device="cpu", verbose=False)[0]
        out[n] = [{"box": b, "conf": c, "cls": int(k)} for b, c, k in
                  zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    cap.release()
    return out


def run_in_browser(dist, query, timeout):
    """Serve `dist`, open index.html?<query> in headless Edge, wait for the page's report."""
    server = ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=str(dist)))
    server.handle_error = lambda *a: None  # browser closing a download early is harmless noise
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = tempfile.mkdtemp(prefix="edge_check_")  # throwaway profile: doesn't touch your real Edge
    edge = subprocess.Popen([str(EDGE), "--headless=new", f"--user-data-dir={profile}",
                             "--autoplay-policy=no-user-gesture-required", "--enable-unsafe-webgpu",
                             *os.environ.get("EDGE_FLAGS", "").split(),  # extra browser flags for experiments
                             f"http://127.0.0.1:{PORT}/index.html?{query}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finished = got_report.wait(timeout=timeout)
    edge.kill()
    server.shutdown()
    if not finished:
        sys.exit(f"No report from the browser within {timeout} s.")
    if "error" in report:
        sys.exit(f"Browser error:\n{report['error']}")
    return report


def check_tracking():
    """Whole pipeline in the browser vs the Python reference (tests/fixtures/tracking_reference.json)."""
    ref = json.loads((ROOT / "tests" / "fixtures" / "tracking_reference.json").read_text())
    web = run_in_browser(build(), "track=1", timeout=600)
    fmt = lambda e: f"#{e['trackId']} {e['className']} {e['direction']} @frame {e['frame']}"  # noqa: E731
    py_events, web_events = [fmt(e) for e in ref["events"]], [fmt(e) for e in web["events"]]
    same_ids = sum(sorted(w) == sorted(t["trackId"] for t in f["tracks"]) for w, f in zip(web["ids"], ref["frames"]))
    print("Python events :", py_events)
    print("Browser events:", web_events)
    summary = {
        "python_totals": ref["totals"], "browser_totals": web["totals"],
        "identical_events": py_events == web_events,
        "frames_with_identical_track_ids": f"{same_ids}/{len(ref['frames'])}",
        "python_unique_ids": len({t["trackId"] for f in ref["frames"] for t in f["tracks"]}),
        "browser_unique_ids": len({i for f in web["ids"] for i in f}),
        "browser_seconds": round(web["seconds"], 1),
        "browser_fps": round(web["frames"] / web["seconds"], 2),
        "execution_provider": web["ep"],
    }
    (ROOT / "results" / "web_tracking_check.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main(model=ROOT / "models" / "yolo26n_dynamic.onnx", ep="auto", isolate=False):
    Handler.isolate = isolate
    rect = 1 if "dynamic" in Path(model).name else 0  # fixed 640x640 models need square input
    run_in_browser(build(model), f"ep={ep}&rect={rect}&imgsz={IMGSZ}&check={','.join(map(str, FRAMES))}", timeout=180)
    py = python_detections(model)
    rows, all_ious, all_dconf = [], [], []
    for n in FRAMES:
        web = report["results"][str(n)]
        matched = 0
        for p in py[n]:  # best-matching browser box of the same class for each Python box
            best = max((iou(p["box"], w["box"]) for w in web if w["classId"] == p["cls"]), default=0)
            if best >= 0.5:
                matched += 1
                all_ious.append(best)
                w = max((w for w in web if w["classId"] == p["cls"]), key=lambda w: iou(p["box"], w["box"]))
                all_dconf.append(abs(w["conf"] - p["conf"]))
        rows.append({"frame": n, "python_boxes": len(py[n]), "browser_boxes": len(web), "matched": matched})
        print(f"frame {n:3d}: python {len(py[n])} boxes, browser {len(web)} boxes, matched {matched}")

    summary = {
        "model": Path(model).name,
        "frames": rows,
        "total_python_boxes": sum(r["python_boxes"] for r in rows),
        "total_browser_boxes": sum(r["browser_boxes"] for r in rows),
        "total_matched": sum(r["matched"] for r in rows),
        "mean_iou_of_matches": round(sum(all_ious) / len(all_ious), 4) if all_ious else None,
        "mean_conf_difference": round(sum(all_dconf) / len(all_dconf), 4) if all_dconf else None,
        "browser_ms_per_frame": round(report["msPerFrame"], 1),
        "browser_seek_ms": round(report["seekMs"], 1),
        "browser_detect_ms": round(report["detectMs"], 1),
        "browser_detect_parts_ms": report.get("parts"),  # pre = image prep, model = inference, post = decode + NMS
        "cross_origin_isolated": report["crossOriginIsolated"],
        "wasm_threads": report["wasmThreads"],
        "execution_provider": report["ep"],
        "gpu_adapter": report.get("gpuAdapter"),
        "browser": report["userAgent"],
    }
    (ROOT / "results" / "web_check.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "track":
        check_tracking()
        sys.exit()
    main(model=args[0] if args else ROOT / "models" / "yolo26n_dynamic.onnx",
         ep=args[1] if len(args) > 1 else "auto",
         isolate=len(args) > 2 and args[2] == "isolate")
