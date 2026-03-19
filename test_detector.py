"""
test_detector.py — Quick sanity test without Streamlit.
Uses your webcam (or a local video file) and shows output in an OpenCV window.

Run with:  python test_detector.py
Press Q to quit.
"""

import cv2
from stream import FrameGrabber
from detector import CrowdDetector
from classifier import DensityClassifier
import time

# ── Config ────────────────────────────────────────────────────────────────────
# Change this to test different sources:
#   "webcam"               → your laptop camera
#   "path/to/video.mp4"    → local file
#   "https://youtube..."   → live stream
SOURCE = "webcam"

# ── Init ──────────────────────────────────────────────────────────────────────
print("Starting crowd detector test...")
grabber    = FrameGrabber(source=SOURCE, fps_limit=2)
detector   = CrowdDetector(model_size="nano", confidence=0.4)
classifier = DensityClassifier()

grabber.start()
print("Waiting for first frame... (press Q in the window to quit)")

# ── Loop ──────────────────────────────────────────────────────────────────────
while True:
    frame = grabber.get_frame()

    if frame is None:
        time.sleep(0.1)
        continue

    # Detect
    detection = detector.detect(frame)
    result    = classifier.classify(detection["count"])

    # Draw overlay
    annotated = detector.draw_overlay(
        frame, detection, result.level, result.color_bgr
    )

    # Print to terminal as well
    print(f"[{result.timestamp}] People: {result.count} | {result.level} | {'⚠️ ALERT' if result.alert else 'OK'}")

    # Show in window
    cv2.imshow("CrowdSafe — press Q to quit", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

grabber.stop()
cv2.destroyAllWindows()
print("Done.")
