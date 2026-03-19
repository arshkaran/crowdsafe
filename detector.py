"""
detector.py — Runs YOLOv8 on a frame and returns detected people + count.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class CrowdDetector:
    """
    Wraps YOLOv8 to detect only people (class 0) in a frame.
    On first use, YOLOv8 automatically downloads yolov8n.pt (~6 MB).
    """

    # Available model sizes — trade speed for accuracy:
    # yolov8n.pt  → nano   (fastest, good for demo)
    # yolov8s.pt  → small
    # yolov8m.pt  → medium (better accuracy)
    # yolov8l.pt  → large
    # yolov8x.pt  → xlarge (best, slowest)
    MODEL_OPTIONS = {
        "nano":   "yolov8n.pt",
        "small":  "yolov8s.pt",
        "medium": "yolov8m.pt",
    }

    def __init__(self, model_size: str = "nano", confidence: float = 0.4):
        """
        model_size:  "nano", "small", or "medium"
        confidence:  minimum detection confidence (0.0–1.0)
                     Lower = more detections (including false positives)
                     Higher = only high-confidence detections
        """
        model_file = self.MODEL_OPTIONS.get(model_size, "yolov8n.pt")
        print(f"[detector] Loading YOLOv8 model: {model_file}")
        self.model = YOLO(model_file)
        self.confidence = confidence
        print(f"[detector] Model loaded. Confidence threshold: {confidence}")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a single frame.

        Returns a dict:
        {
            "count":      int,           # number of people detected
            "boxes":      list[list],    # [[x1, y1, x2, y2, conf], ...]
            "annotated":  np.ndarray,    # frame with bounding boxes drawn
        }
        """
        # Run YOLO — class 0 = person
        results = self.model(
            frame,
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )[0]

        boxes = []
        annotated = frame.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            boxes.append([x1, y1, x2, y2, conf])

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 100), 2)

            # Confidence label above box
            label = f"{conf:.0%}"
            cv2.putText(
                annotated, label,
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 200, 100), 1, cv2.LINE_AA,
            )

        return {
            "count":     len(boxes),
            "boxes":     boxes,
            "annotated": annotated,
        }

    def draw_overlay(self, frame: np.ndarray, detection: dict, density_label: str, density_color: tuple) -> np.ndarray:
        """
        Draws the crowd count + density label as a top-left overlay on the frame.
        density_color: BGR tuple, e.g. (0, 200, 0) for green
        """
        overlay = detection["annotated"].copy()
        h, w = overlay.shape[:2]

        # Semi-transparent dark banner at top
        banner = overlay.copy()
        cv2.rectangle(banner, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.addWeighted(banner, 0.6, overlay, 0.4, 0, overlay)

        # Person count
        count = detection["count"]
        cv2.putText(
            overlay, f"People: {count}",
            (14, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Density label with color
        cv2.putText(
            overlay, f"Density: {density_label}",
            (14, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, density_color, 2, cv2.LINE_AA,
        )

        return overlay
