"""Load YOLO and run detection + ByteTrack tracking on single frames."""
from dataclasses import dataclass

from ultralytics import YOLO

from counter import lap_fallback

lap_fallback.install()  # must run before Ultralytics loads its tracker code

# COCO class IDs we care about (YOLO is trained on the 80 COCO classes).
CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Track:
    track_id: int      # stable ID given by the tracker
    cls_name: str      # e.g. "car"
    conf: float        # detector confidence 0..1
    box: tuple         # (x1, y1, x2, y2) in pixels

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class Detector:
    def __init__(self, model_path="yolo26n.pt", classes=None, conf=0.25):
        self.model = YOLO(model_path)
        # Only ask YOLO for the classes we want to count.
        self.class_ids = classes if classes is not None else list(CLASSES)
        self.conf = conf

    def track(self, frame):
        """Detect + track objects in one frame. Call on consecutive frames of ONE video.

        persist=True keeps the tracker's memory between calls, so the same
        car keeps the same ID from frame to frame.
        """
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=self.class_ids,
            conf=self.conf,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes.id is None:  # no confirmed tracks in this frame yet
            return []

        tracks = []
        for tid, cls_id, conf, xyxy in zip(boxes.id.int().tolist(), boxes.cls.int().tolist(),
                                           boxes.conf.tolist(), boxes.xyxy.tolist()):
            tracks.append(Track(tid, result.names[cls_id], conf, tuple(xyxy)))
        return tracks

    def reset(self):
        """Forget all tracks (call before starting a new video)."""
        if self.model.predictor is not None and hasattr(self.model.predictor, "trackers"):
            for tracker in self.model.predictor.trackers:
                tracker.reset()
