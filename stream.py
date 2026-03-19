"""
stream.py — Grabs frames from a live YouTube stream or local webcam.
Usage: from stream import FrameGrabber
"""

import cv2
import subprocess
import threading
import time


class FrameGrabber:
    """
    Continuously reads frames from a live stream in a background thread.
    Always gives you the latest frame — no buffering lag.
    """

    def __init__(self, source: str = "webcam", fps_limit: int = 1):
        """
        source:
          "webcam"         → uses your local webcam (great for testing)
          "https://..."    → any YouTube live stream URL
          "path/to/file"   → a local video file (for offline demo/testing)

        fps_limit: how many frames per second to process (default 1 = 1 frame/sec)
        """
        self.source = source
        self.fps_limit = fps_limit
        self.frame = None
        self.running = False
        self._thread = None
        self._lock = threading.Lock()

    def _get_stream_url(self, youtube_url: str) -> str:
        """Use yt-dlp to extract the direct stream URL from YouTube."""
        print(f"[stream] Resolving stream URL with yt-dlp...")
        try:
            result = subprocess.check_output(
                ["yt-dlp", "-g", "--no-playlist", youtube_url],
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            url = result.decode().strip().split("\n")[0]
            print(f"[stream] Got stream URL.")
            return url
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"yt-dlp failed. Is the stream live? Error: {e}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                "yt-dlp not found. Install it with: pip install yt-dlp"
            )

    def start(self):
        """Open the video capture and start background reading."""
        if self.source == "webcam":
            cap_source = 0
        elif self.source.startswith("http"):
            cap_source = self._get_stream_url(self.source)
        else:
            cap_source = self.source  # local file path

        self._cap = cv2.VideoCapture(cap_source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        self.running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"[stream] Started. Grabbing ~{self.fps_limit} frame(s)/sec.")

    def _read_loop(self):
        """Background loop — reads frames at the configured fps limit."""
        interval = 1.0 / self.fps_limit
        while self.running:
            ret, frame = self._cap.read()
            if not ret:
                print("[stream] Stream ended or frame read failed.")
                self.running = False
                break
            with self._lock:
                self.frame = frame
            time.sleep(interval)

    def get_frame(self):
        """Returns the latest frame (numpy array, BGR), or None if not ready."""
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stop the stream and release resources."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        if hasattr(self, "_cap"):
            self._cap.release()
        print("[stream] Stopped.")
