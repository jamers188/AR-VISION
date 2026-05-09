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

# ─────────────────────────────────────────────
# AUTO-DOWNLOAD MODEL FROM GOOGLE DRIVE
# ─────────────────────────────────────────────
MODEL_PATH = "net_g_latest.pth"
GDRIVE_ID  = "1l9FNhi0gec-qBqd3M55Tpw16fkSNDgWT"

def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return True, None
    try:
        try:
            import gdown
        except ImportError:
            os.system("pip install gdown -q")
            import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        if os.path.exists(MODEL_PATH):
            return True, None
        return False, "Download completed but file not found."
    except Exception as e:
        return False, str(e)

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
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');
:root {
  --bg:#050a12; --bg2:#0a1525; --panel:#0d1e35; --border:#1a3a5c;
  --accent:#00d4ff; --accent2:#00ff88; --danger:#ff4060; --warn:#ffb800;
  --text:#c8dff0; --dim:#4a6a88;
  --glow:0 0 20px rgba(0,212,255,0.4); --glow2:0 0 30px rgba(0,255,136,0.3);
}
html,body,[class*="css"],.stApp{background-color:var(--bg)!important;color:var(--text)!important;font-family:'Rajdhani',sans-serif!important;}
.block-container{padding:1.5rem 2rem!important;max-width:1400px;}
section[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border);}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
.stButton>button{background:transparent!important;border:1px solid var(--accent)!important;color:var(--accent)!important;font-family:'Orbitron',monospace!important;font-size:0.72rem!important;letter-spacing:0.12em!important;padding:0.55rem 1.3rem!important;border-radius:3px!important;transition:all 0.2s!important;text-transform:uppercase!important;}
.stButton>button:hover{background:var(--accent)!important;color:var(--bg)!important;box-shadow:var(--glow)!important;}
.stSlider>div>div>div>div{background:var(--accent)!important;}
.stSelectbox>div,.stFileUploader>div{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:4px!important;}
hr{border-color:var(--border)!important;}
label,.stSelectbox label,.stSlider label{color:var(--dim)!important;font-size:0.8rem!important;letter-spacing:0.08em!important;text-transform:uppercase!important;}
.hud-header{display:flex;align-items:center;gap:1.2rem;padding:1.2rem 0 1.5rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem;}
.hud-logo{font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:900;color:var(--accent);text-shadow:var(--glow);letter-spacing:0.05em;line-height:1;}
.hud-logo span{color:var(--accent2);}
.hud-tagline{font-size:0.78rem;color:var(--dim);letter-spacing:0.18em;text-transform:uppercase;margin-top:0.2rem;}
.badge{display:inline-block;background:var(--panel);border:1px solid var(--accent);color:var(--accent);font-family:'Share Tech Mono',monospace;font-size:0.65rem;padding:0.2rem 0.6rem;border-radius:2px;letter-spacing:0.12em;}
.badge-green{border-color:var(--accent2);color:var(--accent2);}
.badge-warn{border-color:var(--warn);color:var(--warn);}
.metric-card{background:var(--panel);border:1px solid var(--border);border-top:2px solid var(--accent);border-radius:4px;padding:1rem 1.2rem;text-align:center;}
.metric-val{font-family:'Orbitron',monospace;font-size:1.5rem;font-weight:700;color:var(--accent);text-shadow:var(--glow);}
.metric-lbl{font-size:0.7rem;color:var(--dim);letter-spacing:0.12em;text-transform:uppercase;margin-top:0.3rem;}
.metric-card.green{border-top-color:var(--accent2);}
.metric-card.green .metric-val{color:var(--accent2);text-shadow:var(--glow2);}
.metric-card.warn{border-top-color:var(--warn);}
.metric-card.warn .metric-val{color:var(--warn);}
.panel-box{background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:1.2rem 1.4rem;margin-bottom:1rem;}
.panel-title{font-family:'Orbitron',monospace;font-size:0.7rem;color:var(--accent);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.9rem;display:flex;align-items:center;gap:0.5rem;}
.panel-title::before{content:'';display:inline-block;width:6px;height:6px;background:var(--accent);border-radius:50%;box-shadow:var(--glow);}
.vis-bar-wrap{background:var(--bg);border-radius:3px;height:8px;overflow:hidden;margin:0.4rem 0;}
.vis-bar{height:8px;border-radius:3px;transition:width 0.6s;}
.info-row{display:flex;justify-content:space-between;align-items:center;padding:0.35rem 0;border-bottom:1px solid rgba(26,58,92,0.5);font-size:0.85rem;}
.info-row:last-child{border-bottom:none;}
.info-key{color:var(--dim);font-size:0.75rem;letter-spacing:0.08em;text-transform:uppercase;}
.info-val{font-family:'Share Tech Mono',monospace;color:var(--text);font-size:0.85rem;}
.status-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:0.4rem;}
.dot-green{background:var(--accent2);box-shadow:0 0 6px var(--accent2);animation:pulse 2s infinite;}
.dot-red{background:var(--danger);}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.compare-label{font-family:'Orbitron',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--dim);text-align:center;padding:0.4rem;}
.compare-label.enhanced{color:var(--accent2);}
.footer-bar{margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;font-size:0.7rem;color:var(--dim);letter-spacing:0.1em;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FFA-NET MODEL
# ─────────────────────────────────────────────
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel//8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//8, 1, 1, padding=0, bias=True),
            nn.Sigmoid())
    def forward(self, x):
        return x * self.pa(x)

class CALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel//8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//8, channel, 1, padding=0, bias=True),
            nn.Sigmoid())
    def forward(self, x):
        return x * self.ca(self.avg_pool(x))

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super().__init__()
        self.conv1   = conv(dim, dim, kernel_size, bias=True)
        self.act1    = nn.ReLU(inplace=True)
        self.conv2   = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):
        res = self.act1(self.conv1(x)) + x
        res = self.palayer(self.calayer(self.conv2(res))) + x
        return res

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super().__init__()
        kernel = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        kernel.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*kernel)
    def forward(self, x):
        return self.gp(x) + x

class FFA(nn.Module):
    def __init__(self, gps=3, blocks=19, conv=default_conv):
        super().__init__()
        self.gps = gps
        self.dim = 64
        ks = 3
        self.pre  = nn.Sequential(conv(3, self.dim, ks))
        self.g1   = Group(conv, self.dim, ks, blocks)
        self.g2   = Group(conv, self.dim, ks, blocks)
        self.g3   = Group(conv, self.dim, ks, blocks)
        self.ca   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps, self.dim//16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid())
        self.palayer = PALayer(self.dim)
        self.post = nn.Sequential(conv(self.dim, self.dim, ks), conv(self.dim, 3, ks))

    def forward(self, x1):
        x   = self.pre(x1)
        r1, r2, r3 = self.g1(x), self.g2(self.g1(x)), self.g3(self.g2(self.g1(x)))
        w   = self.ca(torch.cat([r1, r2, r3], dim=1))
        w   = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:,0,::]*r1 + w[:,1,::]*r2 + w[:,2,::]*r3
        return self.post(self.palayer(out)) + x1


# ─────────────────────────────────────────────
# MODEL LOADER (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FFA(gps=3, blocks=19)
    try:
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            # Try common checkpoint key names
            key = next((k for k in ["params", "state_dict", "model", "net_g", "generator"] if k in state), None)
            sd  = state[key] if key else state
            # Strip "module." prefix if saved with DataParallel
            sd  = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        model.eval().to(device)
        return model, device, None
    except Exception as e:
        return None, device, str(e)


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def dehaze_image(model, device, img_bgr, strength=1.0):
    h, w = img_bgr.shape[:2]
    H, W = max((h//16)*16, 16), max((w//16)*16, 16)
    resized = cv2.resize(img_bgr, (W, H))
    # Convert BGR->RGB for the model (trained on RGB)
    inp_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(inp_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip(out, 0, 1)
    # Blend with original if strength < 1
    if strength < 1.0:
        out = inp_rgb * (1 - strength) + out * strength
    # Convert RGB->BGR back for OpenCV
    out_bgr = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(out_bgr, (w, h))

def compute_metrics(orig, enh):
    mse  = np.mean((orig.astype(np.float64)-enh.astype(np.float64))**2)
    psnr = 10*np.log10(255**2/mse) if mse > 0 else 100.0
    og   = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    eg   = cv2.cvtColor(enh,  cv2.COLOR_BGR2GRAY).astype(np.uint8)
    sg   = cv2.Laplacian(og, cv2.CV_64F).var()
    se   = cv2.Laplacian(eg, cv2.CV_64F).var()
    return {"psnr":round(psnr,2), "sharpness_gain":round((se-sg)/(sg+1e-8)*100,1),
            "contrast_orig":round(og.astype(np.float64).std(),1),
            "contrast_enh":round(eg.astype(np.float64).std(),1)}

def detect_hazards(img_bgr):
    gray = cv2.GaussianBlur(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY),(5,5),0)
    edges = cv2.Canny(gray,30,90)
    out   = img_bgr.copy()
    for c in cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) > 1500:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,150),1)
    return out


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="panel-title">⚙ System Configuration</div>', unsafe_allow_html=True)

    model_ready = os.path.exists(MODEL_PATH)
    status_color = "#00ff88" if model_ready else "#ff4060"
    status_text  = "Ready" if model_ready else "Not found"
    st.markdown(f"""
    <div class="panel-box" style="padding:0.8rem 1rem;margin-bottom:0.5rem">
      <div class="info-row">
        <span class="info-key">Model</span>
        <span class="info-val" style="font-size:0.72rem">net_g_latest.pth</span>
      </div>
      <div class="info-row">
        <span class="info-key">Source</span>
        <span class="info-val" style="font-size:0.72rem">Google Drive</span>
      </div>
      <div class="info-row">
        <span class="info-key">Status</span>
        <span class="info-val" style="font-size:0.72rem;color:{status_color}">● {status_text}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ready:
        if st.button("⬇ Download Model"):
            with st.spinner("Downloading from Google Drive..."):
                ok, err = download_model_if_needed()
            if ok:
                st.success("Model downloaded! Refresh the page.")
            else:
                st.error(f"Failed: {err}")
    
    st.markdown("---")
    st.markdown('<div class="panel-title">Enhancement Controls</div>', unsafe_allow_html=True)
    strength     = st.slider("Enhancement Strength", 0.0, 1.0, 1.0, 0.05)
    obj_detect   = st.toggle("Object / Hazard Overlay", value=False)
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
    <div class="hud-tagline">Real-Time AR Vision Enhancement System &middot; FFA-Net Dehazing</div>
  </div>
  <div style="margin-left:auto;display:flex;gap:0.5rem;align-items:center">
    <span class="badge">BCS 410</span>
    <span class="badge badge-green">FFA-Net</span>
    <span class="badge">CUD 2024</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STATUS ROW
# ─────────────────────────────────────────────
model_exists = os.path.exists(MODEL_PATH)
device_label = "CUDA GPU" if torch.cuda.is_available() else "CPU"

c1,c2,c3,c4 = st.columns(4)
with c1:
    dot = "dot-green" if model_exists else "dot-red"
    lbl = "MODEL READY" if model_exists else "DOWNLOADING..."
    cls = "green" if model_exists else ""
    st.markdown(f'<div class="metric-card {cls}"><div class="metric-val" style="font-size:0.85rem"><span class="status-dot {dot}"></span>{lbl}</div><div class="metric-lbl">net_g_latest.pth</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1rem">{device_label}</div><div class="metric-lbl">Compute Device</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card green"><div class="metric-val" style="font-size:1rem">FFA-Net</div><div class="metric-lbl">Architecture · gps=3 blocks=19</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1rem">{int(strength*100)}%</div><div class="metric-lbl">Enhancement Strength</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Checkpoint Inspector (debug) ──
with st.expander("🔍 Checkpoint Inspector (click to debug model keys)"):
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                st.markdown(f"**Top-level keys:** `{list(ckpt.keys())}`")
                key = next((k for k in ["params","state_dict","model","net_g","generator"] if k in ckpt), None)
                if key:
                    inner = ckpt[key]
                    st.markdown(f"**Using key:** `{key}` — {len(inner)} weight tensors")
                    st.markdown(f"**First 5 weight names:** `{list(inner.keys())[:5]}`")
                else:
                    st.markdown(f"**No standard key found — treating as raw state dict**")
                    st.markdown(f"**First 5 weight names:** `{list(ckpt.keys())[:5]}`")
            else:
                st.markdown(f"**Checkpoint type:** `{type(ckpt)}`")
        except Exception as e:
            st.error(f"Could not inspect checkpoint: {e}")
    else:
        st.warning("Model file not downloaded yet.")

# Auto-download on first load (silent)
if not model_exists:
    with st.spinner("⬇ First-time setup: downloading model from Google Drive..."):
        ok, err = download_model_if_needed()
    if ok:
        st.success("✅ Model ready! You can now upload images.")
        st.rerun()
    else:
        st.error(f"Could not download model automatically. Error: {err}\n\nMake sure the Google Drive file is shared publicly.")

# Load model
model, device, load_err = (None, None, None)
if model_exists:
    with st.spinner("Loading model weights..."):
        model, device, load_err = load_model(MODEL_PATH)
    if load_err:
        st.error(f"Model load error: {load_err}")
        model = None


# ─────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────
if "📷" in mode:
    st.markdown('<div class="panel-title">📷 Image Enhancement</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop a hazy image here", type=["jpg","jpeg","png","bmp","tif"], label_visibility="collapsed")

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        orig_bgr   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if orig_bgr is None:
            st.error("Could not decode image.")
        else:
            t0 = time.time()
            if model is not None:
                enh_bgr = dehaze_image(model, device, orig_bgr, strength)
            else:
                # CLAHE fallback when model not loaded
                lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB)
                l,a,b = cv2.split(lab)
                l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
                enh_bgr = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

            if obj_detect:
                enh_bgr = detect_hazards(enh_bgr)

            elapsed = time.time() - t0
            h, w    = orig_bgr.shape[:2]
            metrics = compute_metrics(orig_bgr, enh_bgr)

            if show_compare:
                col_o, col_e = st.columns(2)
                with col_o:
                    st.markdown('<div class="compare-label">◀ ORIGINAL &middot; HAZY INPUT</div>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col_e:
                    st.markdown('<div class="compare-label enhanced">▶ ENHANCED &middot; AI DEHAZED</div>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.image(cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB), caption="AI Enhanced Output", use_container_width=True)

            buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
            st.download_button("⬇ Download Enhanced Image", buf.getvalue(), "enhanced_output.png", "image/png")

            if show_metrics:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="panel-title">📊 Enhancement Metrics</div>', unsafe_allow_html=True)
                m1,m2,m3,m4,m5 = st.columns(5)
                cards = [
                    (m1, f"{elapsed*1000:.0f} ms", "Processing Time", ""),
                    (m2, f"{metrics['psnr']} dB", "PSNR", "green"),
                    (m3, f"+{metrics['sharpness_gain']}%", "Sharpness Gain", "green" if metrics['sharpness_gain']>0 else "warn"),
                    (m4, f"{metrics['contrast_enh']:.0f}", "Contrast (Enhanced)", "green"),
                    (m5, f"{w}x{h}", "Resolution", ""),
                ]
                for col, val, lbl, cls in cards:
                    with col:
                        st.markdown(f'<div class="metric-card {cls}"><div class="metric-val" style="font-size:1.1rem">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                op = min(int(metrics['contrast_orig']/128*100),100)
                ep = min(int(metrics['contrast_enh']/128*100),100)
                st.markdown(f"""
                <div class="panel-box">
                  <div class="panel-title">Contrast Comparison</div>
                  <div class="info-row"><span class="info-key">Original</span><span class="info-val">{metrics['contrast_orig']:.1f}</span></div>
                  <div class="vis-bar-wrap"><div class="vis-bar" style="width:{op}%;background:#4a6a88"></div></div>
                  <div class="info-row"><span class="info-key">Enhanced</span><span class="info-val">{metrics['contrast_enh']:.1f}</span></div>
                  <div class="vis-bar-wrap"><div class="vis-bar" style="width:{ep}%;background:var(--accent2)"></div></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="panel-box" style="text-align:center;padding:3rem;border-style:dashed">
          <div style="font-family:'Orbitron',monospace;font-size:2rem;color:#1a3a5c;margin-bottom:1rem">👁</div>
          <div style="color:#4a6a88;font-size:0.9rem;letter-spacing:0.1em">
            UPLOAD A HAZY IMAGE TO BEGIN ENHANCEMENT<br>
            <span style="font-size:0.75rem;color:#2a4a6a">Supports: JPG &middot; PNG &middot; BMP &middot; TIFF</span>
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VIDEO MODE
# ─────────────────────────────────────────────
else:
    st.markdown('<div class="panel-title">🎞 Video Enhancement</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel-box" style="border-color:var(--warn)">
      <div class="info-row"><span class="info-key">GPU Recommended</span><span class="info-val">CPU may be slow for large videos</span></div>
      <div class="info-row"><span class="info-key">Max Resolution</span><span class="info-val">Frames auto-resized to nearest 16px boundary</span></div>
    </div>""", unsafe_allow_html=True)

    vid_file   = st.file_uploader("Upload a hazy video", type=["mp4","avi","mov","mkv"], label_visibility="collapsed")
    max_frames = st.slider("Max frames to process", 10, 300, 60, 10)

    if vid_file and st.button("🚀 Process Video"):
        if model is None:
            st.error("Model not loaded. Cannot process video.")
        else:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(vid_file.read())
                tmp_path = tmp.name

            cap  = cv2.VideoCapture(tmp_path)
            fps  = cap.get(cv2.CAP_PROP_FPS) or 25
            tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = tmp_path.replace(".mp4","_enhanced.mp4")
            writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))

            prog = st.progress(0)
            stat = st.empty()
            fi   = 0
            t0   = time.time()

            while cap.isOpened() and fi < max_frames:
                ret, frame = cap.read()
                if not ret: break
                enh = dehaze_image(model, device, frame, strength)
                if obj_detect: enh = detect_hazards(enh)
                writer.write(enh)
                fi += 1
                el = time.time()-t0
                prog.progress(fi/min(max_frames,tot))
                stat.markdown(f"""
                <div class="panel-box">
                  <div class="info-row"><span class="info-key">Frames processed</span><span class="info-val">{fi} / {min(max_frames,tot)}</span></div>
                  <div class="info-row"><span class="info-key">Speed</span><span class="info-val">{fi/(el+1e-8):.1f} FPS</span></div>
                  <div class="info-row"><span class="info-key">Elapsed</span><span class="info-val">{el:.1f}s</span></div>
                </div>""", unsafe_allow_html=True)

            cap.release(); writer.release(); os.unlink(tmp_path)
            prog.progress(1.0)
            st.success(f"Done — {fi} frames in {time.time()-t0:.1f}s")
            with open(out_path,"rb") as f:
                st.download_button("⬇ Download Enhanced Video", f.read(), "enhanced_video.mp4", "video/mp4")
            os.unlink(out_path)
    elif not vid_file:
        st.markdown("""
        <div class="panel-box" style="text-align:center;padding:3rem;border-style:dashed">
          <div style="font-family:'Orbitron',monospace;font-size:2rem;color:#1a3a5c;margin-bottom:1rem">🎞</div>
          <div style="color:#4a6a88;font-size:0.9rem;letter-spacing:0.1em">
            UPLOAD A HAZY VIDEO TO BEGIN PROCESSING<br>
            <span style="font-size:0.75rem;color:#2a4a6a">Supports: MP4 &middot; AVI &middot; MOV &middot; MKV</span>
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
  <span>NEXTGEN VISION AI &middot; BCS 410 &middot; Canadian University Dubai</span>
  <span>FFA-Net &middot; PyTorch &middot; OpenCV &middot; Streamlit</span>
  <span>Mahdi Parvaz &amp; Thejaswini Sunil</span>
</div>""", unsafe_allow_html=True)
