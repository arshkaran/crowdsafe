# CrowdSafe — AI Crowd Density Monitor
**Thomas More Professional Week Project**
Team: Ryan de Boer, Michiel Mertens, [your name]

Real-time crowd density detection using YOLOv8 + OpenCV + Streamlit.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Test with your webcam first (no internet needed)
```bash
python test_detector.py
```
Point your webcam at a group of people (or a photo/video on your screen).
Press **Q** to quit. You should see bounding boxes + density label.

### 3. Launch the full dashboard
```bash
streamlit run dashboard.py
```
Opens in your browser at http://localhost:8501

---

## Using a Live Stream (Times Square)

In the dashboard sidebar:
1. Select **YouTube live stream**
2. Paste: `https://www.youtube.com/watch?v=_Tpo8q0BKEA`
3. Click **▶ Start monitoring**

> Note: yt-dlp resolves the stream URL. This requires an active internet connection and a live stream being broadcast.

---

## Project Structure

```
crowd-safety/
├── stream.py          # Frame grabber (webcam / YouTube / file)
├── detector.py        # YOLOv8 person detection
├── classifier.py      # Density classification + alerting
├── dashboard.py       # Streamlit live dashboard
├── test_detector.py   # Quick OpenCV test (no Streamlit needed)
└── requirements.txt
```

---

## How It Works

```
Live stream → Frame grabber → YOLOv8 → Person count → Density classifier → Dashboard
(YouTube)     (OpenCV/yt-dlp)  (nano)    (bounding boxes) (LOW/MEDIUM/HIGH)   (Streamlit)
```

1. **stream.py** — Uses `yt-dlp` to resolve the YouTube stream URL, then OpenCV reads frames in a background thread at the configured FPS.
2. **detector.py** — Passes each frame to YOLOv8 (class 0 = person only). Returns count + bounding boxes.
3. **classifier.py** — Maps count to LOW / MEDIUM / HIGH / CRITICAL with configurable thresholds and alert flags.
4. **dashboard.py** — Streamlit UI showing the annotated live feed, real-time count, density level, alert banner, and a historical trend chart.

---

## Density Thresholds (adjustable in classifier.py)

| Level    | Count    | Action                          |
|----------|----------|---------------------------------|
| LOW      | 0–9      | Normal, no action               |
| MEDIUM   | 10–24    | Monitor closely                 |
| HIGH     | 25–49    | Notify crowd management         |
| CRITICAL | 50+      | Immediate intervention required |

---

## Tips for the Presentation

- Start with **webcam demo** — reliable, no internet dependency
- Switch to **Times Square** stream for the wow factor
- Show the **trend chart** filling up over time
- Trigger an **alert** by showing a crowd image to the webcam
- Explain: bounding boxes = YOLO detections, color = density level
# crowdsafe
