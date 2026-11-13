# Interview notes — Smart Traffic & People Counter

Ten questions an interviewer may ask, with short answers. Numbers come from `results/`.

**1. Walk me through the pipeline.**
Each frame goes through YOLO26n (boxes, class, confidence), then ByteTrack (gives each object a
stable ID across frames), then my line counter (counts an ID once when its centre crosses the line,
as IN or OUT). The output is an annotated video, a counts table and a CSV of crossing events.

**2. How does ByteTrack keep the same ID for a car?**
For every known track a Kalman filter predicts where the box should be in the new frame. Predictions
are matched to new detections by overlap (cost = 1 − IoU × confidence) with the Hungarian algorithm.
Matched detections keep the old ID; unmatched ones start new tracks (confirmed after a second
sighting); tracks missing for 30 frames are dropped. ByteTrack also gives unmatched tracks a second
chance with low-confidence boxes.

**3. How does your line counter decide IN vs OUT?**
The 2D cross product of the line direction and the point tells which side of the line a point is on
(sign). I remember each ID's previous centre; if the segment from old to new centre intersects the
line segment, the object crossed. The sign of the new side gives IN or OUT. A set of counted IDs
makes sure nothing is counted twice.

**4. Why export to ONNX / OpenVINO, and what did INT8 give you?**
PyTorch is built for training; inference engines are optimised for running a fixed model on specific
hardware. On my laptop CPU, OpenVINO INT8 ran the model at 31.6 FPS vs 17.7 for PyTorch (1.79×) and
was 3× smaller, with identical counts on the test video.

**5. ONNX INT8 was slower than FP32. Why?**
0.32× speed. INT8 only pays off with fast integer instructions (VNNI), which this CPU doesn't have,
and ONNX Runtime's quantized model adds quantize/dequantize steps around layers. OpenVINO fuses them
and has better Intel kernels. Lesson: always benchmark on the target hardware.

**6. How did you make the benchmark fair?**
Same video, same 640 px input, same CPU and confidence settings; frames preloaded so disk reads
aren't timed; 10 warm-up frames not timed; median of 3 runs. I also compared the counts, not just
the speed.

**7. Why does the live demo run in the browser?**
Free Gradio hosting on Hugging Face started requiring a paid plan (I got a 402 error). Static Spaces
are still free, so I ported the pipeline to JavaScript with ONNX Runtime Web: zero hosting cost,
videos stay on the visitor's device. The trade-off is that speed depends on their hardware.

**8. How do you know the JavaScript version is correct?**
Three checks. Detection: browser boxes match Python on sampled frames (28/28, mean IoU 0.99).
Tracker: fed Python's exact detections, my JS ByteTrack reproduces Python's IDs on all 330 frames.
Whole pipeline: same 9 crossings on the same frames. These checks caught real bugs, e.g. missing
box clipping at the frame edge.

**9. What did you try to speed up the browser version, and what didn't work?**
Feeding 16:9 video as 640×384 instead of a padded 640×640 square (dynamic-shape ONNX) cut model time
from 160 to 110 ms (≈1.5×). A 480 px input was not faster on WebGPU and missed a small car. Skipping
every 2nd frame broke tracking (0 of 9 crossings), because IoU matching needs boxes to overlap
between frames. WebNN was 48 ms but is still behind a flag, so it's used automatically when available.

**10. What would you improve next?**
Validate counts against a hand count on several busier videos; test more classes (people, bikes);
add a re-identification model for long occlusions; try WebNN / FP16 when browsers support them;
let users export the annotated video from the browser.
