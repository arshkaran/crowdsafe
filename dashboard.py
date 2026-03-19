"""
dashboard.py — CrowdSafe  |  Professional crowd monitoring dashboard
Run with:  streamlit run dashboard.py

Video glitch fix:
  The previous version called st.rerun() every 100 ms, which tore down and
  rebuilt every widget on every frame — causing the flash/flicker.
  This version writes raw JPEG bytes into a single persistent st.empty()
  container via a tight while-loop.  Only the image bytes change; nothing
  else in the DOM is touched, so the video is perfectly smooth.
"""

import io
import time
import queue
import threading
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from stream import FrameGrabber
from detector import CrowdDetector
from classifier import DensityClassifier



st.set_page_config(
    page_title="CrowdSafe",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Fonts + global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600;700&display=swap');

/* Base */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #07090d !important;
    color: #b8ccd8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #0a0d13 !important;
    border-right: 1px solid #151e2b !important;
}
[data-testid="stSidebar"] * { font-family: 'IBM Plex Sans', sans-serif !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 1rem !important; }

/* ── Sidebar width ── */
[data-testid="stSidebar"] { min-width: 310px !important; max-width: 310px !important; }
[data-testid="stSidebar"] > div:first-child { padding: 0 20px 20px !important; }

/* ── Sidebar labels ── */
.sidebar-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #3a5870;
    padding: 20px 0 8px;
    border-top: 1px solid #151e2b;
    margin-top: 8px;
}
.sidebar-label:first-child { border-top: none; padding-top: 4px; }

/* ── Sidebar radio bigger ── */
[data-testid="stSidebar"] div[data-baseweb="radio"] label {
    font-size: 14px !important;
    color: #7aafc8 !important;
    padding: 4px 0 !important;
}
[data-testid="stSidebar"] div[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p {
    font-size: 14px !important;
}

/* ── Sidebar slider labels bigger ── */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 13px !important;
    color: #4a6a80 !important;
}
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] {
    font-size: 12px !important;
    color: #2a3d52 !important;
}

/* ── File uploader ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #0c1119 !important;
    border: 1px dashed #1e3042 !important;
    border-radius: 8px !important;
    padding: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 12px 10px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #0c1119 !important;
    border: 1px solid #1e3042 !important;
    color: #00c896 !important;
    font-size: 12px !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    border-color: #00c896 !important;
    background: #031510 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span {
    font-size: 11px !important;
    color: #2a3d52 !important;
}

/* ── Threshold table ── */
.thresh-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 4px;
}
.thresh-table td {
    padding: 7px 0;
    font-size: 13px;
    border-bottom: 1px solid #0f1a26;
    vertical-align: middle;
}
.thresh-table tr:last-child td { border-bottom: none; }
.thresh-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Sidebar buttons bigger ── */
[data-testid="stSidebar"] .stButton > button {
    padding: 11px 18px !important;
    font-size: 13px !important;
}

/* ── Header bar ── */
.header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 16px;
    border-bottom: 1px solid #151e2b;
    margin-bottom: 22px;
}
.header-logo {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    font-size: 20px;
    letter-spacing: -0.3px;
    color: #e2edf5;
}
.header-logo em { font-style: normal; color: #00c896; }
.header-meta {
    display: flex;
    align-items: center;
    gap: 18px;
    font-size: 11px;
    color: #2a3d52;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
}
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #051a12;
    border: 1px solid #00c89640;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 600;
    color: #00c896;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00c896;
    animation: blink 1.8s ease-in-out infinite;
}
.standby-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #111820;
    border: 1px solid #1e2d3d;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 600;
    color: #3a5268;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── KPI row ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 18px;
}
.kpi {
    background: #0c1119;
    border: 1px solid #151e2b;
    border-radius: 8px;
    padding: 16px 18px 14px;
    position: relative;
    overflow: hidden;
}
.kpi::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--kpi-line, #151e2b), transparent);
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 32px;
    font-weight: 500;
    line-height: 1;
    color: var(--kpi-color, #e2edf5);
    margin: 0 0 7px;
    letter-spacing: -1px;
}
.kpi-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2a3d52;
    margin: 0;
}
.kpi-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #1e3042;
    margin: 5px 0 0;
}

/* ── Status banner ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 16px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 18px;
    border-left: 3px solid transparent;
}
.status-ok       { background:#051510; border-color:#00c896; color:#00c896; }
.status-medium   { background:#120e04; border-color:#e8a020; color:#e8a020; }
.status-high     { background:#130606; border-color:#e84040; color:#e84040; }
.status-critical { background:#1a0404; border-color:#ff1e1e; color:#ff8080; }
.status-idle     { background:#0c1119; border-color:#1e2d3d; color:#2a3d52; }
.status-icon { font-size: 14px; }
.status-time {
    margin-left: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    opacity: 0.5;
    letter-spacing: 0.06em;
}

/* ── Video panel ── */
.video-shell {
    background: #07090d;
    border: 1px solid #151e2b;
    border-radius: 8px;
    overflow: hidden;
}
.video-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 14px;
    border-bottom: 1px solid #151e2b;
    background: #0a0d13;
}
.video-topbar-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2a3d52;
    font-family: 'IBM Plex Mono', monospace;
}
.video-topbar-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #1e3042;
    letter-spacing: 0.06em;
}

/* ── Right panel cards ── */
.panel-card {
    background: #0c1119;
    border: 1px solid #151e2b;
    border-radius: 8px;
    padding: 15px 18px;
    margin-bottom: 10px;
}
.panel-card-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2a3d52;
    margin-bottom: 12px;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 5px 0;
    border-bottom: 1px solid #0f1a26;
    font-size: 12px;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: #2a3d52; font-weight: 500; }
.stat-val {
    font-family: 'IBM Plex Mono', monospace;
    color: #8aafc4;
    font-size: 13px;
}

/* ── Density level badge ── */
.level-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 4px;
    text-transform: uppercase;
}
.level-LOW      { background:#031510; color:#00c896; border:1px solid #00c89630; }
.level-MEDIUM   { background:#120e04; color:#e8a020; border:1px solid #e8a02030; }
.level-HIGH     { background:#130606; color:#e84040; border:1px solid #e8404030; }
.level-CRITICAL { background:#1a0404; color:#ff4444; border:1px solid #ff444460; }

/* ── Chart override ── */
[data-testid="stArrowVegaLiteChart"] canvas { border-radius: 4px; }

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: #00c896 !important;
    color: #07090d !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    padding: 9px 18px !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #00e6ad !important; }
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: #111820 !important;
    color: #8aafc4 !important;
    border: 1px solid #151e2b !important;
}
.stRadio > label,
.stSlider > label,
.stSelectSlider > label,
[data-testid="stWidgetLabel"] {
    color: #4a6a80 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
}
.stTextInput input {
    background: #0c1119 !important;
    border: 1px solid #151e2b !important;
    color: #8aafc4 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}
.stTextInput input:focus { border-color: #00c89660 !important; outline: none !important; }
div[data-baseweb="radio"] label { color: #4a6a80 !important; }
.stSlider [data-baseweb="slider"] [role="slider"] { background: #00c896 !important; }
.stAlert { border-radius: 7px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _s(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_s("running",       False)
_s("grabber",       None)
_s("stop_event",    threading.Event())
_s("result_queue",  queue.Queue(maxsize=3))
_s("history",       deque(maxlen=120))
_s("total_alerts",  0)
_s("peak_count",    0)
_s("frame_count",   0)
_s("last_result",   None)


# ── Inference worker (background thread) ──────────────────────────────────────
def inference_worker(grabber, detector, classifier, rq, stop_evt):
    """
    Dedicated thread: grab → detect → classify → push JPEG bytes + metadata.
    Pushes results as fast as inference allows; drops frames if consumer
    is slow (queue full).  This means the display loop only ever shows the
    LATEST frame — no queue backup, no rubber-band catch-up lag.
    """
    while not stop_evt.is_set():
        frame = grabber.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        detection = detector.detect(frame)
        result    = classifier.classify(detection["count"])
        annotated = detector.draw_overlay(
            frame, detection, result.level, result.color_bgr
        )

        # Encode to JPEG once here — much cheaper than PIL conversion in render
        ok, buf = cv2.imencode(
            ".jpg", annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        if not ok:
            continue

        payload = {
            "jpeg":      buf.tobytes(),
            "result":    result,
            "boxes":     detection["boxes"],
        }

        # Non-blocking — drop if slow consumer hasn't caught up
        try:
            rq.put_nowait(payload)
        except queue.Full:
            try:               # replace stale item with fresh one
                rq.get_nowait()
                rq.put_nowait(payload)
            except queue.Empty:
                pass


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 0 18px">
        <div style="font-family:'IBM Plex Sans',sans-serif;font-weight:700;
                    font-size:28px;color:#e2edf5;letter-spacing:-0.5px;line-height:1">
            Crowd<em style="color:#00c896;font-style:normal">Safe</em>
        </div>
        <div style="font-size:11px;color:#2a3d52;letter-spacing:0.14em;
                    text-transform:uppercase;margin-top:6px;font-weight:600">
            Crowd Intelligence System
        </div>
        <div style="margin-top:12px;height:1px;background:linear-gradient(90deg,#00c89630,transparent)"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Video Source</div>', unsafe_allow_html=True)
    source_type = st.radio(
        "", ["Webcam", "YouTube stream", "Local file"],
        label_visibility="collapsed",
    )

    uploaded_file = None
    if source_type == "Webcam":
        stream_source = "webcam"
    elif source_type == "YouTube stream":
        stream_source = st.text_input(
            "Stream URL", value="https://www.youtube.com/watch?v=_Tpo8q0BKEA"
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Select a video file from your computer",
        )
        stream_source = None   # resolved below after sidebar

    st.markdown('<div class="sidebar-label">Model Settings</div>', unsafe_allow_html=True)
    model_size = st.select_slider(
        "Speed  ←  →  Accuracy",
        options=["nano", "small", "medium"],
        value="nano",
    )
    confidence = st.slider("Detection confidence", 0.20, 0.85, 0.40, 0.05,
                           format="%.2f")

    st.markdown('<div class="sidebar-label">Density Levels</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="thresh-table">
        <tr>
            <td><span class="thresh-dot" style="background:#00c896"></span>
                <span style="color:#00c896;font-weight:600">Low</span></td>
            <td style="color:#3a5870;font-family:'IBM Plex Mono',monospace">&lt; 10 people</td>
        </tr>
        <tr>
            <td><span class="thresh-dot" style="background:#e8a020"></span>
                <span style="color:#e8a020;font-weight:600">Medium</span></td>
            <td style="color:#3a5870;font-family:'IBM Plex Mono',monospace">10 – 24</td>
        </tr>
        <tr>
            <td><span class="thresh-dot" style="background:#e84040"></span>
                <span style="color:#e84040;font-weight:600">High</span></td>
            <td style="color:#3a5870;font-family:'IBM Plex Mono',monospace">25 – 49</td>
        </tr>
        <tr>
            <td><span class="thresh-dot" style="background:#ff4444"></span>
                <span style="color:#ff4444;font-weight:600">Critical</span></td>
            <td style="color:#3a5870;font-family:'IBM Plex Mono',monospace">50+</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Controls</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: start_btn = st.button("▶  Start", use_container_width=True)
    with c2: stop_btn  = st.button("■  Stop",  use_container_width=True)

    if st.button("↺  Reset session", use_container_width=True):
        st.session_state.history      = deque(maxlen=120)
        st.session_state.total_alerts = 0
        st.session_state.peak_count   = 0
        st.session_state.frame_count  = 0
        st.session_state.last_result  = None

# ── Resolve uploaded file to a temp path ─────────────────────────────────────
import tempfile, os
if uploaded_file is not None:
    if ("uploaded_tmp" not in st.session_state or
            st.session_state.get("uploaded_name") != uploaded_file.name):
        suffix = os.path.splitext(uploaded_file.name)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.flush()
        st.session_state.uploaded_tmp  = tmp.name
        st.session_state.uploaded_name = uploaded_file.name
    stream_source = st.session_state.uploaded_tmp
elif source_type == "Local file" and uploaded_file is None:
    stream_source = None  # nothing uploaded yet


# ── Start / stop ──────────────────────────────────────────────────────────────
if start_btn and not st.session_state.running:
    if not stream_source:
        st.sidebar.error("Please upload or select a video source first.")
        start_btn = False

if start_btn and not st.session_state.running:
    st.session_state.running      = True
    st.session_state.stop_event   = threading.Event()
    st.session_state.result_queue = queue.Queue(maxsize=3)

    g = FrameGrabber(source=stream_source, fps_limit=15)
    g.start()
    st.session_state.grabber = g

    det = CrowdDetector(model_size=model_size, confidence=confidence)
    cls = DensityClassifier()

    threading.Thread(
        target=inference_worker,
        args=(g, det, cls,
              st.session_state.result_queue,
              st.session_state.stop_event),
        daemon=True,
    ).start()

if stop_btn and st.session_state.running:
    st.session_state.running = False
    st.session_state.stop_event.set()
    if st.session_state.grabber:
        st.session_state.grabber.stop()
        st.session_state.grabber = None


# ── Header ────────────────────────────────────────────────────────────────────
status_pill = (
    '<span class="live-pill"><span class="live-dot"></span>Live</span>'
    if st.session_state.running else
    '<span class="standby-pill">Standby</span>'
)
src_label = (
    stream_source.split("?")[0][-38:] + "…"
    if (stream_source := locals().get("stream_source", "webcam")) and len(stream_source) > 40
    else stream_source
)

st.markdown(f"""
<div class="header-bar">
    <div class="header-logo">Crowd<em>Safe</em></div>
    <div class="header-meta">
        <span style="font-size:10px">{src_label}</span>
        {status_pill}
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPI placeholders ──────────────────────────────────────────────────────────
kpi_ph    = st.empty()
status_ph = st.empty()


# ── Main layout ───────────────────────────────────────────────────────────────
col_vid, col_panel = st.columns([11, 6], gap="medium")

with col_vid:
    st.markdown("""
    <div class="video-shell">
        <div class="video-topbar">
            <span class="video-topbar-label">Live feed</span>
            <span class="video-topbar-badge" id="frame-badge">—</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # ← This single image placeholder never gets destroyed — only bytes change
    video_ph = st.empty()

with col_panel:
    chart_ph  = st.empty()
    detail_ph = st.empty()


# ── KPI renderer ─────────────────────────────────────────────────────────────
def render_kpis(count=None, level=None, peak=None, alerts=None, avg=None, frames=None):
    count_color = {
        "LOW": "#00c896", "MEDIUM": "#e8a020",
        "HIGH": "#e84040", "CRITICAL": "#ff4444",
    }.get(level, "#2a3d52") if level else "#2a3d52"
    accent = {
        "LOW": "#00c896", "MEDIUM": "#e8a020",
        "HIGH": "#e84040", "CRITICAL": "#ff4444",
    }.get(level, "#151e2b") if level else "#151e2b"

    kpi_ph.markdown(f"""
    <div class="kpi-row">
        <div class="kpi" style="--kpi-line:{accent}">
            <div class="kpi-val" style="--kpi-color:{count_color}">{count if count is not None else "—"}</div>
            <div class="kpi-label">People detected</div>
            <div class="kpi-sub">avg {avg:.1f} / session</div>
        </div>
        <div class="kpi" style="--kpi-line:{accent}">
            <div class="kpi-val" style="--kpi-color:#2a3d52;font-size:13px;padding-top:6px">
                {"<span class='level-badge level-" + level + "'>" + level + "</span>" if level else "—"}
            </div>
            <div class="kpi-label" style="margin-top:10px">Density level</div>
        </div>
        <div class="kpi" style="--kpi-line:#e8a02040">
            <div class="kpi-val" style="--kpi-color:#e8a020">{peak if peak is not None else "—"}</div>
            <div class="kpi-label">Session peak</div>
        </div>
        <div class="kpi" style="--kpi-line:#e8404040">
            <div class="kpi-val" style="--kpi-color:#e84040">{alerts if alerts is not None else "—"}</div>
            <div class="kpi-label">Alerts fired</div>
            <div class="kpi-sub">{frames or 0} frames processed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status(result=None):
    if result is None:
        status_ph.markdown("""
        <div class="status-bar status-idle">
            <span class="status-icon">⬡</span>
            <span>System ready — press Start to begin monitoring</span>
        </div>
        """, unsafe_allow_html=True)
        return
    cls_map = {
        "LOW":      ("status-ok",       "✓"),
        "MEDIUM":   ("status-medium",   "◈"),
        "HIGH":     ("status-high",     "⚠"),
        "CRITICAL": ("status-critical", "⚡"),
    }
    css, icon = cls_map.get(result.level, ("status-idle", "●"))
    status_ph.markdown(f"""
    <div class="status-bar {css}">
        <span class="status-icon">{icon}</span>
        <span>{result.message}</span>
        <span class="status-time">{result.timestamp}</span>
    </div>
    """, unsafe_allow_html=True)


def render_detail(boxes, frames, peak, alerts, avg):
    conf_avg = round(sum(b[4] for b in boxes) / max(len(boxes), 1) * 100) if boxes else 0
    detail_ph.markdown(f"""
    <div class="panel-card">
        <div class="panel-card-title">Detection stats</div>
        <div class="stat-row">
            <span class="stat-key">Confidence avg</span>
            <span class="stat-val">{conf_avg}%</span>
        </div>
        <div class="stat-row">
            <span class="stat-key">Bounding boxes</span>
            <span class="stat-val">{len(boxes)}</span>
        </div>
        <div class="stat-row">
            <span class="stat-key">Session peak</span>
            <span class="stat-val">{peak}</span>
        </div>
        <div class="stat-row">
            <span class="stat-key">Alerts total</span>
            <span class="stat-val">{alerts}</span>
        </div>
        <div class="stat-row">
            <span class="stat-key">Frames processed</span>
            <span class="stat-val">{frames}</span>
        </div>
        <div class="stat-row">
            <span class="stat-key">Session avg</span>
            <span class="stat-val">{avg:.1f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Idle render ───────────────────────────────────────────────────────────────
def render_idle():
    render_kpis(avg=0.0, frames=0, peak=0, alerts=0)
    render_status(None)
    video_ph.markdown("""
    <div style="background:#07090d;border:1px solid #151e2b;border-radius:0 0 8px 8px;
                padding:110px 0;text-align:center;">
        <div style="font-size:28px;color:#151e2b;margin-bottom:10px">⬡</div>
        <div style="font-size:10px;color:#1a2a38;letter-spacing:0.14em;
                    text-transform:uppercase;font-weight:600">
            No feed active
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main loop — SMOOTH VIDEO ──────────────────────────────────────────────────
#
#  Key insight: instead of st.rerun() (which rebuilds the whole page),
#  we sit in a tight while-loop and update ONLY the image placeholder.
#  The KPIs and chart update every N frames to avoid thrashing the DOM.
#
if not st.session_state.running:
    render_idle()

else:
    render_status(st.session_state.last_result)

    hist    = st.session_state.history
    N_CHART = 8   # redraw chart every 8 frames (chart is expensive)
    i       = 0

    while st.session_state.running:
        try:
            payload = st.session_state.result_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        result = payload["result"]
        boxes  = payload["boxes"]
        fc     = st.session_state.frame_count + 1

        # Update session state
        st.session_state.frame_count = fc
        if result.count > st.session_state.peak_count:
            st.session_state.peak_count = result.count
        if result.alert:
            st.session_state.total_alerts += 1
        hist.append({"f": fc, "count": result.count})
        st.session_state.last_result = result

        avg = sum(h["count"] for h in hist) / max(len(hist), 1)

        # ── Video — update every frame, smooth ──
        video_ph.image(
            payload["jpeg"],
            use_column_width=True,
            output_format="JPEG",
        )

        # ── KPIs + status — update every frame (cheap HTML swap) ──
        render_kpis(
            count=result.count,
            level=result.level,
            peak=st.session_state.peak_count,
            alerts=st.session_state.total_alerts,
            avg=avg,
            frames=fc,
        )
        render_status(result)

        # ── Chart + detail — throttled ──
        i += 1
        if i % N_CHART == 0 and len(hist) > 2:
            df = pd.DataFrame(list(hist))
            with chart_ph.container():
                st.markdown('<div class="panel-card-title" style="font-size:10px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#2a3d52;margin-bottom:8px">Count trend</div>', unsafe_allow_html=True)
                st.line_chart(
                    df.set_index("f")["count"],
                    height=150,
                    use_container_width=True,
                    color="#00c896",
                )
            render_detail(
                boxes,
                fc,
                st.session_state.peak_count,
                st.session_state.total_alerts,
                avg,
            )