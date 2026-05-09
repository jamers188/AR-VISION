import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import io
import os
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NEXTGEN VISION AI",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  – dark HUD aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

/* ── Root palette ── */
:root {
  --bg:        #050a12;
  --bg2:       #0a1525;
  --panel:     #0d1e35;
  --border:    #1a3a5c;
  --accent:    #00d4ff;
  --accent2:   #00ff88;
  --danger:    #ff4060;
  --warn:      #ffb800;
  --text:      #c8dff0;
  --dim:       #4a6a88;
  --glow:      0 0 20px rgba(0,212,255,0.4);
  --glow2:     0 0 30px rgba(0,255,136,0.3);
}

/* ── Base reset ── */
html, body, [class*="css"], .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Rajdhani', sans-serif !important;
}

/* ── Streamlit scaffolding ── */
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }
section[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
.stButton>button {
  background: transparent !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.12em !important;
  padding: 0.55rem 1.3rem !important;
  border-radius: 3px !important;
  transition: all 0.2s !important;
  text-transform: uppercase !important;
}
.stButton>button:hover {
  background: var(--accent) !important;
  color: var(--bg) !important;
  box-shadow: var(--glow) !important;
}
.stSlider > div > div > div > div { background: var(--accent) !important; }
.stSelectbox > div, .stFileUploader > div {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}
.stProgress > div > div { background: var(--accent) !important; }
hr { border-color: var(--border) !important; }
label, .stSelectbox label, .stSlider label { color: var(--dim) !important; font-size: 0.8rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }

/* ── Custom components ── */
.hud-header {
  display: flex; align-items: center; gap: 1.2rem;
  padding: 1.2rem 0 1.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}
.hud-logo {
  font-family: 'Orbitron', monospace;
  font-size: 1.6rem; font-weight: 900;
  color: var(--accent);
  text-shadow: var(--glow);
  letter-spacing: 0.05em;
  line-height: 1;
}
.hud-logo span { color: var(--accent2); }
.hud-tagline {
  font-size: 0.78rem; color: var(--dim);
  letter-spacing: 0.18em; text-transform: uppercase;
  margin-top: 0.2rem;
}
.badge {
  display: inline-block;
  background: var(--panel);
  border: 1px solid var(--accent);
  color: var(--accent);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem;
  padding: 0.2rem 0.6rem;
  border-radius: 2px;
  letter-spacing: 0.12em;
}
.badge-green { border-color: var(--accent2); color: var(--accent2); }
.badge-warn  { border-color: var(--warn);    color: var(--warn); }
.badge-red   { border-color: var(--danger);  color: var(--danger); }

.metric-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-top: 2px solid var(--accent);
  border-radius: 4px;
  padding: 1rem 1.2rem;
  text-align: center;
}
.metric-val {
  font-family: 'Orbitron', monospace;
  font-size: 1.5rem; font-weight: 700;
  color: var(--accent);
  text-shadow: var(--glow);
}
.metric-lbl {
  font-size: 0.7rem; color: var(--dim);
  letter-spacing: 0.12em; text-transform: uppercase;
  margin-top: 0.3rem;
}
.metric-card.green { border-top-color: var(--accent2); }
.metric-card.green .metric-val { color: var(--accent2); text-shadow: var(--glow2); }
.metric-card.warn { border-top-color: var(--warn); }
.metric-card.warn .metric-val { color: var(--warn); }

.panel-box {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 1rem;
}
.panel-title {
  font-family: 'Orbitron', monospace;
  font-size: 0.7rem; color: var(--accent);
  letter-spacing: 0.15em; text-transform: uppercase;
  margin-bottom: 0.9rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.panel-title::before {
  content: ''; display: inline-block;
  width: 6px; height: 6px;
  background: var(--accent);
  border-radius: 50%;
  box-shadow: var(--glow);
}
.scanline {
  position: relative;
  background: linear-gradient(
    to bottom,
    transparent 50%,
    rgba(0,212,255,0.03) 50%
  );
  background-size: 100% 4px;
}
.vis-bar-wrap { background: var(--bg); border-radius: 3px; height: 8px; overflow: hidden; margin: 0.4rem 0; }
.vis-bar { height: 8px; border-radius: 3px; transition: width 0.6s; }
.info-row { display: flex; justify-content: space-between; align-items: center; padding: 0.35rem 0; border-bottom: 1px solid rgba(26,58,92,0.5); font-size: 0.85rem; }
.info-row:last-child { border-bottom: none; }
.info-key { color: var(--dim); font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
.info-val { font-family: 'Share Tech Mono', monospace; color: var(--text); font-size: 0.85rem; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 0.4rem; }
.dot-green { background: var(--accent2); box-shadow: 0 0 6px var(--accent2); animation: pulse 2s infinite; }
.dot-red   { background: var(--danger); }
.dot-warn  { background: var(--warn); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

.compare-label {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem; letter-spacing: 0.15em;
  text-transform: uppercase; color: var(--dim);
  text-align: center; padding: 0.4rem;
}
.compare-label.enhanced { color: var(--accent2); }

.footer-bar {
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.7rem; color: var(--dim); letter-spacing: 0.1em;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FFA-NET MODEL DEFINITION
# (Must match the architecture used during training)
# ─────────────────────────────────────────────

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1  = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super().__init__()
        kernel = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        kernel.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*kernel)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class FFA(nn.Module):
    def __init__(self, gps=3, blocks=19, conv=default_conv):
        super().__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.palayer = PALayer(self.dim)
        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size),
        ]
        self.pre  = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load the FFA-Net generator weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFA(gps=3, blocks=19)
    try:
        state = torch.load(model_path, map_location=device)
        # handle various checkpoint formats
        if isinstance(state, dict):
            if "params" in state:
                model.load_state_dict(state["params"], strict=False)
            elif "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            elif "model" in state:
                model.load_state_dict(state["model"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
        else:
            model = state
        model.eval().to(device)
        return model, device, None
    except Exception as e:
        return None, device, str(e)


# ─────────────────────────────────────────────
# INFERENCE  (single image)
# ─────────────────────────────────────────────
def dehaze_image(model, device, img_np: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Run the FFA-Net on a BGR numpy array, return BGR numpy array."""
    h, w = img_np.shape[:2]
    # resize to multiple of 16 for stability
    H = (h // 16) * 16 or 16
    W = (w // 16) * 16 or 16
    inp = cv2.resize(img_np, (W, H))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip(out, 0, 1)
    # blend with original based on strength
    if strength < 1.0:
        orig_norm = inp
        out = orig_norm * (1 - strength) + out * strength
    out = (out * 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    out = cv2.resize(out, (w, h))
    return out


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(orig: np.ndarray, enhanced: np.ndarray) -> dict:
    orig_f = orig.astype(np.float64)
    enh_f  = enhanced.astype(np.float64)
    mse = np.mean((orig_f - enh_f) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else 100.0

    orig_g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float64)
    enh_g  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # visibility: std of Laplacian (sharpness proxy)
    sharpness_orig = cv2.Laplacian(orig_g.astype(np.uint8), cv2.CV_64F).var()
    sharpness_enh  = cv2.Laplacian(enh_g.astype(np.uint8), cv2.CV_64F).var()
    sharpness_gain = (sharpness_enh - sharpness_orig) / (sharpness_orig + 1e-8) * 100

    # contrast
    contrast_orig = orig_g.std()
    contrast_enh  = enh_g.std()

    return {
        "psnr": round(psnr, 2),
        "sharpness_gain": round(sharpness_gain, 1),
        "contrast_orig": round(contrast_orig, 1),
        "contrast_enh": round(contrast_enh, 1),
    }


# ─────────────────────────────────────────────
# OBJECT DETECTION (lightweight, no extra model)
# ─────────────────────────────────────────────
def detect_edges_contours(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img_bgr.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 150), 1)
    return out


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="panel-title">⚙ System Configuration</div>', unsafe_allow_html=True)

    model_path = st.text_input(
        "Model path (.pth)",
        value="net_g_latest.pth",
        help="Path to your FFA-Net generator checkpoint"
    )

    st.markdown("---")
    st.markdown('<div class="panel-title">Enhancement Controls</div>', unsafe_allow_html=True)

    strength = st.slider("Enhancement Strength", 0.0, 1.0, 1.0, 0.05,
                         help="Blend factor between original and dehazed")
    obj_detect = st.toggle("Object / Hazard Overlay", value=False)
    show_compare = st.toggle("Side-by-Side Comparison", value=True)
    show_metrics = st.toggle("Show Metrics", value=True)

    st.markdown("---")
    st.markdown('<div class="panel-title">Input Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["📷  Image", "🎞  Video"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.68rem;color:#4a6a88;line-height:1.8;letter-spacing:0.05em">
    <b style="color:#00d4ff">NEXTGEN VISION AI</b><br>
    BCS 410 · Canadian University Dubai<br>
    Mahdi Parvaz · Thejaswini Sunil<br>
    Supervisor: Dr. Mehak Khurana
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hud-header">
  <div>
    <div class="hud-logo">NEXTGEN <span>VISION</span> AI</div>
    <div class="hud-tagline">Real-Time AR Vision Enhancement System · FFA-Net Dehazing</div>
  </div>
  <div style="margin-left:auto;display:flex;gap:0.5rem;align-items:center">
    <span class="badge">BCS 410</span>
    <span class="badge badge-green">FFA-Net</span>
    <span class="badge">CUD 2024</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL STATUS ROW
# ─────────────────────────────────────────────
model_col1, model_col2, model_col3, model_col4 = st.columns(4)

model_exists = os.path.exists(model_path)
device_label = "CUDA GPU" if torch.cuda.is_available() else "CPU"

with model_col1:
    dot = "dot-green" if model_exists else "dot-red"
    label = "MODEL LOADED" if model_exists else "MODEL NOT FOUND"
    st.markdown(f"""
    <div class="metric-card {'green' if model_exists else ''}">
      <div class="metric-val" style="font-size:0.9rem">
        <span class="status-dot {dot}"></span>{label}
      </div>
      <div class="metric-lbl">{model_path}</div>
    </div>""", unsafe_allow_html=True)

with model_col2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val" style="font-size:1rem">{device_label}</div>
      <div class="metric-lbl">Compute Device</div>
    </div>""", unsafe_allow_html=True)

with model_col3:
    st.markdown(f"""
    <div class="metric-card green">
      <div class="metric-val" style="font-size:1rem">FFA-Net</div>
      <div class="metric-lbl">Architecture · gps=3 blocks=19</div>
    </div>""", unsafe_allow_html=True)

with model_col4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val" style="font-size:1rem">{int(strength*100)}%</div>
      <div class="metric-lbl">Enhancement Strength</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
if model_exists:
    with st.spinner("🔄 Loading model weights..."):
        model, device, err = load_model(model_path)
    if err:
        st.error(f"⚠ Model load error: {err}")
        model = None
else:
    model = None
    st.warning("⚠ Model file not found at the specified path. Upload processing will use a demonstration pass-through. Place `net_g_latest.pth` in the same folder as this app.")


# ─────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────
if "📷" in mode:
    st.markdown('<div class="panel-title">📷 Image Enhancement</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a hazy image here",
        type=["jpg", "jpeg", "png", "bmp", "tif"],
        label_visibility="collapsed"
    )

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        orig_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if orig_bgr is None:
            st.error("Could not decode image.")
        else:
            t0 = time.time()

            if model is not None:
                enh_bgr = dehaze_image(model, device, orig_bgr, strength)
            else:
                # demo fallback: CLAHE contrast enhancement
                lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enh_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

            if obj_detect:
                enh_bgr = detect_edges_contours(enh_bgr)

            elapsed = time.time() - t0
            h, w = orig_bgr.shape[:2]
            metrics = compute_metrics(orig_bgr, enh_bgr)

            # ── Display ──
            if show_compare:
                col_orig, col_enh = st.columns(2)
                with col_orig:
                    st.markdown('<div class="compare-label">◀ ORIGINAL · HAZY INPUT</div>', unsafe_allow_html=True)
                    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                    st.image(orig_rgb, use_container_width=True)
                with col_enh:
                    st.markdown('<div class="compare-label enhanced">▶ ENHANCED · AI DEHAZED</div>', unsafe_allow_html=True)
                    enh_rgb = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)
                    st.image(enh_rgb, use_container_width=True)
            else:
                enh_rgb = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)
                st.image(enh_rgb, caption="AI Enhanced Output", use_container_width=True)

            # ── Download ──
            enh_pil = Image.fromarray(cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            enh_pil.save(buf, format="PNG")
            st.download_button(
                "⬇ Download Enhanced Image",
                data=buf.getvalue(),
                file_name="enhanced_output.png",
                mime="image/png"
            )

            # ── Metrics ──
            if show_metrics:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="panel-title">📊 Enhancement Metrics</div>', unsafe_allow_html=True)
                m1, m2, m3, m4, m5 = st.columns(5)
                cards = [
                    (m1, f"{elapsed*1000:.0f} ms", "Processing Time", ""),
                    (m2, f"{metrics['psnr']} dB", "PSNR", "green"),
                    (m3, f"+{metrics['sharpness_gain']}%", "Sharpness Gain", "green" if metrics['sharpness_gain'] > 0 else "warn"),
                    (m4, f"{metrics['contrast_enh']:.0f}", "Contrast (Enhanced)", "green"),
                    (m5, f"{w}×{h}", "Resolution", ""),
                ]
                for col, val, lbl, cls in cards:
                    with col:
                        st.markdown(f"""
                        <div class="metric-card {cls}">
                          <div class="metric-val" style="font-size:1.1rem">{val}</div>
                          <div class="metric-lbl">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

                # contrast bar
                st.markdown("<br>", unsafe_allow_html=True)
                orig_pct = min(int(metrics['contrast_orig'] / 128 * 100), 100)
                enh_pct  = min(int(metrics['contrast_enh']  / 128 * 100), 100)
                st.markdown(f"""
                <div class="panel-box">
                  <div class="panel-title">Contrast Comparison</div>
                  <div class="info-row">
                    <span class="info-key">Original</span>
                    <span class="info-val">{metrics['contrast_orig']:.1f}</span>
                  </div>
                  <div class="vis-bar-wrap"><div class="vis-bar" style="width:{orig_pct}%;background:#4a6a88"></div></div>
                  <div class="info-row">
                    <span class="info-key">Enhanced</span>
                    <span class="info-val">{metrics['contrast_enh']:.1f}</span>
                  </div>
                  <div class="vis-bar-wrap"><div class="vis-bar" style="width:{enh_pct}%;background:var(--accent2)"></div></div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="panel-box" style="text-align:center;padding:3rem;border-style:dashed">
          <div style="font-family:'Orbitron',monospace;font-size:2rem;color:#1a3a5c;margin-bottom:1rem">👁</div>
          <div style="color:#4a6a88;font-size:0.9rem;letter-spacing:0.1em">
            UPLOAD A HAZY IMAGE TO BEGIN ENHANCEMENT<br>
            <span style="font-size:0.75rem;color:#2a4a6a">Supports: JPG · PNG · BMP · TIFF</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VIDEO MODE
# ─────────────────────────────────────────────
else:
    st.markdown('<div class="panel-title">🎞 Video Enhancement</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel-box" style="border-color:var(--warn)">
      <div class="info-row">
        <span class="info-key">⚡ GPU Recommended</span>
        <span class="info-val">CPU may be slow for large videos</span>
      </div>
      <div class="info-row">
        <span class="info-key">Max Resolution</span>
        <span class="info-val">Frames resized to nearest 16× for stability</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    vid_file = st.file_uploader("Upload a hazy video", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")
    max_frames = st.slider("Max frames to process", 10, 300, 60, 10)

    if vid_file and st.button("🚀 Process Video"):
        if model is None:
            st.error("Model not loaded. Cannot process video.")
        else:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(vid_file.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path = tmp_path.replace(".mp4", "_enhanced.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

            progress = st.progress(0)
            status   = st.empty()
            frame_i  = 0
            t_start  = time.time()

            while cap.isOpened() and frame_i < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                enh = dehaze_image(model, device, frame, strength)
                if obj_detect:
                    enh = detect_edges_contours(enh)
                writer.write(enh)
                frame_i += 1
                pct = frame_i / min(max_frames, total)
                progress.progress(pct)
                elapsed = time.time() - t_start
                fps_proc = frame_i / (elapsed + 1e-8)
                status.markdown(f"""
                <div class="panel-box">
                  <div class="info-row"><span class="info-key">Frames processed</span><span class="info-val">{frame_i} / {min(max_frames,total)}</span></div>
                  <div class="info-row"><span class="info-key">Processing speed</span><span class="info-val">{fps_proc:.1f} FPS</span></div>
                  <div class="info-row"><span class="info-key">Elapsed</span><span class="info-val">{elapsed:.1f}s</span></div>
                </div>""", unsafe_allow_html=True)

            cap.release()
            writer.release()
            os.unlink(tmp_path)

            progress.progress(1.0)
            st.success(f"✅ Enhancement complete — {frame_i} frames processed in {time.time()-t_start:.1f}s")

            with open(out_path, "rb") as f:
                st.download_button("⬇ Download Enhanced Video", f.read(), "enhanced_video.mp4", "video/mp4")
            os.unlink(out_path)

    elif not vid_file:
        st.markdown("""
        <div class="panel-box" style="text-align:center;padding:3rem;border-style:dashed">
          <div style="font-family:'Orbitron',monospace;font-size:2rem;color:#1a3a5c;margin-bottom:1rem">🎞</div>
          <div style="color:#4a6a88;font-size:0.9rem;letter-spacing:0.1em">
            UPLOAD A HAZY VIDEO TO BEGIN PROCESSING<br>
            <span style="font-size:0.75rem;color:#2a4a6a">Supports: MP4 · AVI · MOV · MKV</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
  <span>NEXTGEN VISION AI · BCS 410 · Canadian University Dubai</span>
  <span>FFA-Net · PyTorch · OpenCV · Streamlit</span>
  <span>Mahdi Parvaz &amp; Thejaswini Sunil</span>
</div>
""", unsafe_allow_html=True)
