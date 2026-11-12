---
title: Traffic & People Counter
emoji: 🚦
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Count vehicles & people crossing a line with YOLO26 + ByteTrack
tags:
  - computer-vision
  - object-detection
  - object-tracking
  - yolo
  - openvino
---

# 🚦 Smart Traffic & People Counter

Upload a short video (or pick the sample). The app detects cars, motorcycles, buses, trucks and
people with **YOLO26n**, tracks each one with **ByteTrack**, and counts how many cross the yellow
line in each direction (IN / OUT). You get an annotated video, a counts table and a CSV of every
crossing event.

- Runs on the free CPU tier using an **OpenVINO INT8** export of YOLO26n
  (1.79× faster than PyTorch for the model alone, measured on an Intel i5-10300H laptop CPU).
- To keep the shared CPU responsive, only the first 20 s of a video are processed and frames are
  resized to at most 960 px wide.

Source code, benchmark and details: https://github.com/amjad-c-k/traffic-people-counter

Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
Sample video: [Elevated View of Highway with Traffic](https://www.pexels.com/video/elevated-view-of-highway-with-traffic-10995755/) (Pexels, free licence).
