# Smart Traffic & People Counter

Upload a video → the app detects cars, bikes, buses, trucks and people, tracks each one with an ID,
counts how many cross a line (IN / OUT), and runs on a normal CPU.

> 🚧 Work in progress — live demo, demo GIF, benchmark table and full docs are coming.

## Features (so far)
- YOLO26n detection + ByteTrack tracking with stable IDs
- Own line-crossing logic: IN / OUT per class, each object counted once
- Annotated output video (boxes, IDs, trails, count panel) + CSV of crossing events
- Gradio web app: upload or sample video, class filter, adjustable (slanted) counting line

## Run locally (Windows, Python 3.12)
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py          # open http://127.0.0.1:7860
```

## Credits & licence
- Detection and tracking: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- Sample video: [Elevated View of Highway with Traffic](https://www.pexels.com/video/elevated-view-of-highway-with-traffic-10995755/) on Pexels (free licence).
