import os
import json
import time
import cv2
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import websocket  # websocket-client (sync)
from PIL import Image

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

st.set_page_config(page_title="NEXTGEN VISION AI", page_icon="N", layout="wide", initial_sidebar_state="expanded")

# ── Backend WebSocket endpoint ──────────────────────────────────────────────
WS_URL = "ws://172.20.207.169:8080/ws/dehaze"

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root{--bg:#020617;--panel:#0f172a;--border:rgba(148,163,184,.18);--cyan:#22d3ee;--blue:#38bdf8;--green:#34d399;--text:#e5f4ff;--muted:#94a3b8;--warning:#fbbf24}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}.stApp{background:radial-gradient(circle at top left,rgba(34,211,238,.14),transparent 34%),radial-gradient(circle at top right,rgba(52,211,153,.10),transparent 31%),linear-gradient(135deg,#020617 0%,#050b18 55%,#020617 100%)!important;color:var(--text)!important}.block-container{max-width:1350px!important;padding-top:1.8rem!important;padding-bottom:3rem!important}header[data-testid="stHeader"]{background:transparent!important}[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(15,23,42,.98),rgba(2,6,23,.98))!important;border-right:1px solid var(--border)!important}h1{font-size:3.2rem!important;line-height:1.02!important;font-weight:900!important;letter-spacing:-.07em!important;color:var(--text)!important;margin-bottom:.2rem!important}h2,h3{color:var(--text)!important;font-weight:800!important;letter-spacing:-.04em!important}.hero-subtitle{color:var(--muted);font-size:1.05rem;margin-top:-.4rem;margin-bottom:1.4rem;max-width:850px;line-height:1.65}.hero-badge{display:inline-flex;align-items:center;gap:.45rem;background:rgba(34,211,238,.10);border:1px solid rgba(34,211,238,.25);color:#cffafe;padding:.38rem .7rem;border-radius:999px;font-size:.78rem;font-weight:700;margin-right:.45rem;margin-bottom:.6rem}.metric-card{position:relative;overflow:hidden;background:linear-gradient(180deg,rgba(15,23,42,.96),rgba(15,23,42,.72))!important;border:1px solid var(--border)!important;border-radius:22px!important;padding:1.1rem 1rem!important;text-align:left!important;min-height:105px;box-shadow:0 18px 55px rgba(0,0,0,.25)}.metric-card:after{content:"";position:absolute;height:3px;left:18px;right:18px;top:0;background:linear-gradient(90deg,var(--cyan),var(--green));border-radius:999px}.metric-val{position:relative;color:var(--text)!important;font-size:1.22rem!important;font-weight:900!important;letter-spacing:-.03em!important;margin-top:.25rem}.metric-lbl{position:relative;color:var(--muted)!important;font-size:.72rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;margin-top:.45rem}.small-note{color:var(--muted);font-size:.86rem;line-height:1.7;background:rgba(15,23,42,.56);border:1px solid var(--border);border-radius:16px;padding:.85rem}.warning-box{background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.28);color:#fde68a;padding:1rem 1.1rem;border-radius:18px;margin-bottom:1rem}.info-box{background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.25);color:#cffafe;padding:1rem 1.1rem;border-radius:18px;margin-bottom:1rem}[data-testid="stImage"] img,video{border-radius:22px!important;border:1px solid var(--border)!important;box-shadow:0 20px 70px rgba(0,0,0,.35)}[data-testid="stFileUploader"]{background:rgba(15,23,42,.72)!important;border:1px dashed rgba(34,211,238,.40)!important;border-radius:22px!important;padding:1rem!important}.stButton>button{background:linear-gradient(135deg,var(--cyan),var(--blue))!important;color:#020617!important;border:0!important;border-radius:14px!important;font-weight:900!important;box-shadow:0 12px 30px rgba(34,211,238,.22)!important}.stSelectbox div[data-baseweb="select"]>div,.stTextInput input,.stNumberInput input{background:rgba(15,23,42,.92)!important;border:1px solid var(--border)!important;border-radius:14px!important;color:var(--text)!important}.stAlert{background:rgba(15,23,42,.84)!important;border:1px solid var(--border)!important;border-radius:18px!important}.stDeployButton{display:none!important}
[data-testid="stVerticalBlock"] video{max-width:760px!important;width:100%!important;height:auto!important;object-fit:contain!important}
</style>
''', unsafe_allow_html=True)

# ── WebSocket helpers ────────────────────────────────────────────────────────

def _encode_frame(bgr, quality=90):
    """Encode a BGR frame to JPEG bytes."""
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def _decode_frame(data):
    """Decode JPEG bytes to a BGR frame."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _build_params(strength, dcp_only, inference_size, enable_detection,
                  conf_threshold, only_driving_classes, draw_ar_style, mode):
    return json.dumps({
        "strength": strength,
        "dcp_only": dcp_only,
        "inference_size": inference_size,
        "enable_detection": enable_detection,
        "conf_threshold": conf_threshold,
        "only_driving_classes": only_driving_classes,
        "draw_ar_style": draw_ar_style,
        "mode": mode,
    })

def process_frame_remote(bgr, strength, dcp_only, inference_size,
                         enable_detection, conf_threshold,
                         only_driving_classes, draw_ar_style, mode,
                         ws=None, owns_ws=True):
    """
    Send one frame to the backend and receive:
      message 1 – JSON metadata  {"dets": [...], "dehaze_time": float, "yolo_time": float}
      message 2 – JPEG bytes of the dehazed frame
      message 3 – JPEG bytes of the final annotated frame

    Pass an already-open websocket.WebSocket() as `ws` for video/live loops
    to avoid reconnecting on every frame.
    """
    close_after = False
    if ws is None:
        ws = websocket.WebSocket()
        ws.connect(WS_URL)
        close_after = True

    try:
        params = _build_params(strength, dcp_only, inference_size,
                               enable_detection, conf_threshold,
                               only_driving_classes, draw_ar_style, mode)
        ws.send(params)                        # message 1 → params (text)
        ws.send_binary(_encode_frame(bgr))     # message 2 → raw frame

        meta   = json.loads(ws.recv())         # ← JSON metadata
        deh    = _decode_frame(ws.recv())      # ← dehazed JPEG
        final  = _decode_frame(ws.recv())      # ← annotated JPEG

        dets           = meta.get("dets", [])
        dehaze_time    = meta.get("dehaze_time", 0.0)
        yolo_time      = meta.get("yolo_time", 0.0)
        return deh, final, dets, dehaze_time, yolo_time
    finally:
        if close_after:
            ws.close()

def check_backend():
    try:
        ws = websocket.WebSocket()
        ws.connect(WS_URL, timeout=3)
        ws.close()
        return True
    except Exception:
        return False

def visibility_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    sharp    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(100, max(0, int(contrast * 1.4 + (sharp ** .5) * 2))), round(contrast, 2), round(sharp, 2)

def draw_system_overlay(img_bgr, mode='IMAGE', fps=None, inference_time=None, detection_count=0):
    out = img_bgr.copy()
    cv2.putText(out, f'NEXTGEN VISION AI | {mode}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .72, (0, 255, 180), 2, cv2.LINE_AA)
    line = f'Objects: {detection_count}'
    if fps is not None:            line += f' | FPS: {fps:.1f}'
    if inference_time is not None: line += f' | Time: {inference_time:.2f}s'
    cv2.putText(out, line, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 212, 255), 2, cv2.LINE_AA)
    return out

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('### System Control')
    backend_ok = check_backend()
    if backend_ok:
        st.success('✅ NVIDIA backend connected')
    else:
        st.error(f'❌ Backend unreachable at\n`{WS_URL}`')
    st.markdown('---')
    app_mode = st.radio('Mode', ['Image Upload', 'Video Upload', 'Live Camera'], index=0)
    st.markdown('---'); st.markdown('### Enhancement')
    strength       = st.slider('Enhancement strength', 0.0, 1.0, 1.0, .05)
    dcp_only       = st.toggle('DCP only mode', value=False)
    inference_size = st.selectbox('Dehazing inference size', [128, 192, 256], index=1)
    st.markdown('---'); st.markdown('### Object Detection')
    enable_detection     = st.toggle('Enable YOLO detection', value=True)
    conf_threshold       = st.slider('YOLO confidence', .10, .90, .35, .05)
    only_driving_classes = st.toggle('Driving classes only', value=True)
    draw_ar_style        = st.toggle('AR-style overlay', value=True)
    st.markdown('---'); st.markdown('### Video Settings')
    video_max_frames = st.slider('Max video frames to process', 10, 180, 60, 10)
    video_frame_skip = st.slider('Process every Nth frame', 1, 10, 3, 1)
    st.markdown('---')
    st.markdown('<div class="small-note">All inference runs on the NVIDIA backend. Frontend only handles display.</div>', unsafe_allow_html=True)

if not backend_ok:
    st.error(f'Cannot reach backend at `{WS_URL}`. Start the server on the NVIDIA machine and refresh.')
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('# NEXTGEN VISION AI')
st.markdown(
    '<div class="hero-subtitle">Real-Time AR Vision Enhancement System for adverse weather driving conditions. '
    'Inference runs on an NVIDIA GPU backend — this frontend streams frames and renders results.</div>'
    '<span class="hero-badge">DCP + ResNet Dehazing</span>'
    '<span class="hero-badge">YOLOv8 Object Detection</span>'
    '<span class="hero-badge">Image · Video · Live Camera</span>',
    unsafe_allow_html=True
)
c1, c2, c3, c4 = st.columns(4)
c1.markdown('<div class="metric-card"><div class="metric-val">NVIDIA</div><div class="metric-lbl">Backend</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="metric-val">{"ON" if enable_detection else "OFF"}</div><div class="metric-lbl">YOLO Detection</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="metric-val">{inference_size}px</div><div class="metric-lbl">Inference Size</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card"><div class="metric-val">{"DCP" if dcp_only else "DCP+ResNet"}</div><div class="metric-lbl">Dehaze Mode</div></div>', unsafe_allow_html=True)

# ── Image Upload ─────────────────────────────────────────────────────────────
if app_mode == 'Image Upload':
    st.markdown('## Image Enhancement')
    st.markdown('<div class="info-box">Upload a hazy road image — the frame is sent to the NVIDIA backend and results returned instantly.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader('Upload a hazy/foggy road image', type=['jpg', 'jpeg', 'png'])
    if uploaded:
        rgb = np.array(Image.open(uploaded).convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with st.spinner('Sending to NVIDIA backend…'):
            t0 = time.time()
            deh, final, dets, dt, yt = process_frame_remote(
                bgr, strength, dcp_only, inference_size,
                enable_detection, conf_threshold,
                only_driving_classes, draw_ar_style, 'IMAGE'
            )
            total_time = time.time() - t0

        col1, col2, col3 = st.columns(3)
        col1.image(rgb,                              caption='Original Input',         use_container_width=True)
        col2.image(cv2.cvtColor(deh,   cv2.COLOR_BGR2RGB), caption='Dehazed Output',          use_container_width=True)
        col3.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), caption='Final AR Detection Output', use_container_width=True)

        oscore, _, _ = visibility_score(bgr)
        escore, _, _ = visibility_score(deh)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Original Visibility',  f'{oscore}%')
        m2.metric('Enhanced Visibility',  f'{escore}%')
        m3.metric('Round-trip Time',      f'{total_time:.2f}s')
        m4.metric('Objects Detected',     len(dets))

        if dets:
            st.dataframe(pd.DataFrame(dets), use_container_width=True)
        else:
            st.info('No driving-related objects detected.')
    else:
        st.info('Upload an image to start.')

# ── Video Upload ─────────────────────────────────────────────────────────────
elif app_mode == 'Video Upload':
    st.markdown('## Video Processing')
    st.markdown('<div class="info-box">Frames are streamed to the NVIDIA backend over a persistent WebSocket — no reconnection overhead per frame.</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader('Upload a hazy/foggy road video', type=['mp4', 'avi', 'mov', 'mkv'])
    if uploaded_video:
        inp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        inp.write(uploaded_video.read()); inp.close()
        st.video(inp.name)

        if st.button('Process Video'):
            cap = cv2.VideoCapture(inp.name)
            fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps > 0 else 10
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                  max(1, fps / video_frame_skip), (w, h))

            prog    = st.progress(0); status = st.empty(); preview = st.empty()
            idx = processed = total_det = 0
            start = time.time()

            # open one persistent WS for the whole video
            ws = websocket.WebSocket()
            ws.connect(WS_URL)
            try:
                while cap.isOpened() and processed < video_max_frames:
                    ret, frame = cap.read()
                    if not ret: break
                    idx += 1
                    if idx % video_frame_skip != 0: continue

                    deh, final, dets, dt, yt = process_frame_remote(
                        frame, strength, dcp_only, inference_size,
                        enable_detection, conf_threshold,
                        only_driving_classes, draw_ar_style, 'VIDEO',
                        ws=ws, owns_ws=False
                    )
                    total_det += len(dets)
                    out.write(final)
                    processed += 1

                    if processed % 3 == 0:
                        preview.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB),
                                      caption=f'Processing frame {processed}',
                                      use_container_width=True)
                    prog.progress(min(processed / video_max_frames, 1.0))
                    status.write(f'Processed {processed}/{video_max_frames} frames · Latest detections: {len(dets)}')
            finally:
                ws.close()
                cap.release(); out.release()

            total = time.time() - start
            st.success('Video processing completed.')
            st.video(out_path)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Frames Processed',    processed)
            m2.metric('Total Time',          f'{total:.1f}s')
            m3.metric('Avg Time / Frame',    f'{total / max(processed, 1):.2f}s')
            m4.metric('Total Detections',    total_det)
            with open(out_path, 'rb') as f:
                st.download_button('Download Processed Video', data=f,
                                   file_name='nextgen_vision_processed_video.mp4',
                                   mime='video/mp4')
    else:
        st.info('Upload a road video to process.')

# ── Live Camera ───────────────────────────────────────────────────────────────
else:
    st.markdown('## Camera Preview')
    st.markdown('<div class="warning-box">Live frames are sent to the NVIDIA backend per frame. Use Video Upload for the most stable full-pipeline demo.</div>', unsafe_allow_html=True)

    if not WEBRTC_AVAILABLE:
        st.error('streamlit-webrtc is not installed. Add it to requirements.txt.')
        st.stop()

    rtc_config = RTCConfiguration({'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]})

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_time   = time.time()
            self.fps         = 0.
            self.frame_count = 0
            self.cached_final = None
            self.cached_dets  = []
            # persistent WS per processor instance
            self._ws = None
            self._connect()

        def _connect(self):
            try:
                self._ws = websocket.WebSocket()
                self._ws.connect(WS_URL)
            except Exception:
                self._ws = None

        def recv(self, frame):
            img   = frame.to_ndarray(format='bgr24')
            start = time.time()
            try:
                if self._ws is None:
                    self._connect()
                self.frame_count += 1
                # run detection every 5th frame to keep latency low
                run_det = enable_detection and (self.frame_count % 5 == 0)
                deh, final, dets, _, _ = process_frame_remote(
                    img, strength, dcp_only, 128,
                    run_det, conf_threshold,
                    only_driving_classes, draw_ar_style, 'LIVE',
                    ws=self._ws, owns_ws=False
                )
                if run_det:
                    self.cached_final = final.copy()
                    self.cached_dets  = dets
                elif self.cached_final is not None:
                    final = self.cached_final.copy()
                    dets  = self.cached_dets
                else:
                    final = deh; dets = []

                now = time.time(); dt = now - self.last_time
                self.last_time = now
                self.fps       = 1 / dt if dt > 0 else self.fps
                final = draw_system_overlay(final, 'LIVE', self.fps, time.time() - start, len(dets))
            except Exception:
                final = img
                self._connect()   # attempt reconnect on error
            return av.VideoFrame.from_ndarray(final, format='bgr24')

    webrtc_streamer(
        key='nextgen-live-camera',
        video_processor_factory=LiveProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            'video': {'width': {'ideal': 240}, 'height': {'ideal': 180},
                      'frameRate': {'ideal': 8, 'max': 10}},
            'audio': False
        },
        async_processing=True
    )

# ── Diagnostics ───────────────────────────────────────────────────────────────
with st.expander('System Diagnostics'):
    st.write('Backend URL:', WS_URL)
    st.write('Backend reachable:', backend_ok)
    st.write('WebRTC available:', WEBRTC_AVAILABLE)
