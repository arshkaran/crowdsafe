"""
classifier.py — Classifies crowd density based on person count.
Thresholds are configurable for different location types.
"""

from dataclasses import dataclass
from datetime import datetime


# BGR colors for OpenCV
COLOR_GREEN  = (60, 200, 60)
COLOR_ORANGE = (30, 160, 255)
COLOR_RED    = (50, 50, 230)


@dataclass
class DensityResult:
    level:       str    # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    count:       int
    color_bgr:   tuple
    alert:       bool
    message:     str
    timestamp:   str


class DensityClassifier:
    """
    Classifies crowd density into 4 levels based on person count.

    Default thresholds (you can tune these):
      LOW      →  0–9   people   (safe, normal)
      MEDIUM   → 10–24  people   (monitor)
      HIGH     → 25–49  people   (alert authorities)
      CRITICAL → 50+    people   (immediate action)

    These are tuned for a typical public square view.
    For a narrow street or indoor space, lower the thresholds.
    """

    LEVELS = [
        {
            "name":      "LOW",
            "min":       0,
            "max":       9,
            "color_bgr": COLOR_GREEN,
            "alert":     False,
            "message":   "Normal crowd levels. No action needed.",
        },
        {
            "name":      "MEDIUM",
            "min":       10,
            "max":       24,
            "color_bgr": COLOR_ORANGE,
            "alert":     False,
            "message":   "Moderate crowd. Continue monitoring.",
        },
        {
            "name":      "HIGH",
            "min":       25,
            "max":       49,
            "color_bgr": COLOR_RED,
            "alert":     True,
            "message":   "High crowd density detected! Notify crowd management.",
        },
        {
            "name":      "CRITICAL",
            "min":       50,
            "max":       float("inf"),
            "color_bgr": COLOR_RED,
            "alert":     True,
            "message":   "CRITICAL density! Immediate action required.",
        },
    ]

    def classify(self, count: int) -> DensityResult:
        """Classify a person count and return a DensityResult."""
        for level in self.LEVELS:
            if level["min"] <= count <= level["max"]:
                return DensityResult(
                    level=level["name"],
                    count=count,
                    color_bgr=level["color_bgr"],
                    alert=level["alert"],
                    message=level["message"],
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                )
        # Fallback (shouldn't happen)
        return DensityResult("UNKNOWN", count, COLOR_RED, False, "?", "")

    def history_summary(self, history: list) -> dict:
        """
        Given a list of DensityResult objects, return summary stats.
        Useful for the dashboard trend display.
        """
        if not history:
            return {}
        counts = [r.count for r in history]
        return {
            "min":     min(counts),
            "max":     max(counts),
            "average": round(sum(counts) / len(counts), 1),
            "latest":  counts[-1],
            "alerts":  sum(1 for r in history if r.alert),
        }
