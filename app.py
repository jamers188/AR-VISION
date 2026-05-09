import os
import time
import cv2
import gdown
import torch
import functools
import numpy as np
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ============================================================
# ClearDrive / NEXTGEN VISION AI
# One-file Streamlit prototype: DCP + ResnetGenerator only
# This avoids DVD/NSDNGAN because DVD needs custom DCNv2 CUDA ops.
# ============================================================

st.set_page_config(
    page_title="NEXTGEN VISION AI",
    page_icon="👁️",
    layout="wide",
)

# -----------------------------
# Model download config
# -----------------------------
MODEL_PATH = "remove_hazy_model_256x256.pth"
GDRIVE_ID = "1ji3x-KO19X2yGpT7oaUIpJ5DiCgQg8xS"


def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return True, None

    try:
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

        if os.path.exists(MODEL_PATH):
            return True, None

        return False, "Download finished, but model file was not found."
    except Exception as e:
        return False, str(e)


# -----------------------------
# UI CSS
# -----------------------------
st.markdown(
    """
<style>
.stApp {
    background: #050a12;
    color: #d8ecff;
}
.block-container {
    padding-top: 1.4rem;
}
h1, h2, h3 {
    color: #00d4ff !important;
}
[data-testid="stSidebar"] {
    background: #07111f;
}
.metric-card {
    background: #0d1e35;
    border: 1px solid #1a3a5c;
    border-top: 2px solid #00d4ff;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}
.metric-val {
    color: #00ff88;
    font-size: 1.25rem;
    font-weight: 800;
}
.metric-lbl {
    color: #7fa1bd;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.small-note {
    color: #7fa1bd;
    font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DCP + ResnetGenerator architecture
# from pre_dehazing/network/dehaze_net.py
# ============================================================

class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3):
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = nn.AvgPool2d(kernel_size=2 * self.r + 1, stride=1, padding=self.r)

    def forward(self, I, p):
        N = self.boxfilter(torch.ones(p.size(), device=p.device, dtype=p.dtype))

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I * p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I * I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b


class DCPDehazeGenerator(nn.Module):
    def __init__(self, win_size=15, r=40, eps=1e-3):
        super(DCPDehazeGenerator, self).__init__()
        self.guided_filter = GuidedFilter(r=r, eps=eps)
        self.neighborhood_size = win_size
        self.omega = 0.95

    def get_dark_channel(self, img, w):
        if len(img.shape) == 4:
            img, _ = torch.min(img, dim=1)
            img = torch.unsqueeze(img, dim=1)
            pad_size = int(np.floor(w / 2))

            if w % 2 == 0:
                pads = [pad_size, pad_size - 1, pad_size, pad_size - 1]
            else:
                pads = [pad_size, pad_size, pad_size, pad_size]

            img_min = F.pad(img, pads, mode="replicate")
            dark_img = -F.max_pool2d(-img_min, kernel_size=w, stride=1)
            return dark_img

        raise NotImplementedError("Dark channel only supports 4D tensor [N,C,H,W].")

    def atmospheric_light(self, img, dark_img):
        num, chl, height, width = img.shape
        top_num = max(int(0.001 * height * width), 1)

        A = torch.zeros(num, chl, 1, 1, device=img.device, dtype=img.dtype)

        for num_id in range(num):
            cur_img = img[num_id, ...]
            cur_dark_img = dark_img[num_id, 0, ...]

            _, indices = cur_dark_img.reshape(height * width).sort(descending=True)

            for chl_id in range(chl):
                img_slice = cur_img[chl_id, ...].reshape(height * width)
                A[num_id, chl_id, 0, 0] = torch.mean(img_slice[indices[0:top_num]])

        return A

    def forward(self, x):
        # x is expected in [-1, 1]
        if x.shape[1] > 1:
            guidance = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
        else:
            guidance = x[:, 0, :, :]

        guidance = (guidance + 1) / 2
        guidance = torch.unsqueeze(guidance, dim=1)

        img_patch = (x + 1) / 2
        num, chl, height, width = img_patch.shape

        dark_img = self.get_dark_channel(img_patch, self.neighborhood_size)
        A = self.atmospheric_light(img_patch, dark_img)

        map_A = A.repeat(1, 1, height, width).clamp(min=1e-6)
        trans_raw = 1 - self.omega * self.get_dark_channel(img_patch / map_A, self.neighborhood_size)
        trans_raw = trans_raw.clamp(min=0.05, max=1.0)

        T_DCP = self.guided_filter(guidance, trans_raw).clamp(min=0.05, max=1.0)
        J_DCP = (img_patch - map_A) / T_DCP.repeat(1, 3, 1, 1) + map_A

        return J_DCP.clamp(0, 1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []

        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
            p = 0
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
            p = 0
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
            p = 0
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
            p = 0
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=9,
        padding_type="reflect",
    ):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, bias=use_bias)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return torch.clamp(out, min=-1, max=1)


# ============================================================
# Loading + inference
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcp = DCPDehazeGenerator().to(device).eval()

    resnet = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        norm_layer=nn.InstanceNorm2d,
    ).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)

    if isinstance(ckpt, dict):
        key = next((k for k in ["params", "state_dict", "model", "net_g", "generator"] if k in ckpt), None)
        state_dict = ckpt[key] if key else ckpt
    else:
        state_dict = ckpt

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = resnet.load_state_dict(state_dict, strict=False)

    resnet.eval()
    return dcp, resnet, device, missing, unexpected


def bgr_to_tensor_minus1_to_1(img_bgr, size=256):
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor


def tensor_0_1_to_bgr(tensor, original_hw):
    out = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    out = out.transpose(1, 2, 0)
    out_bgr = cv2.cvtColor((out * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    h, w = original_hw
    out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    return out_bgr


def dehaze_image(img_bgr, strength=1.0, dcp_only=False):
    h, w = img_bgr.shape[:2]
    dcp, resnet, device, _, _ = load_models()

    x = bgr_to_tensor_minus1_to_1(img_bgr, size=256).to(device)

    with torch.no_grad():
        dcp_out_01 = dcp(x)

        if dcp_only:
            refined_01 = dcp_out_01
        else:
            # Resnet expects input roughly in [0,1] from DCP and outputs [-1,1]
            refined_minus1_1 = resnet(dcp_out_01)
            refined_01 = (refined_minus1_1 + 1.0) / 2.0

        result_bgr = tensor_0_1_to_bgr(refined_01, (h, w))

    if strength < 1.0:
        result_bgr = cv2.addWeighted(img_bgr, 1.0 - strength, result_bgr, strength, 0)

    return result_bgr


def auto_visibility_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = min(100, max(0, int((contrast * 1.4) + (sharpness ** 0.5) * 2.0)))
    return score, round(float(contrast), 2), round(float(sharpness), 2)


def add_hazard_overlay(img_bgr):
    out = img_bgr.copy()
    gray = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 35, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1800:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 150), 2)
            count += 1

    return out, count


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### ⚙ System Configuration")

    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found.")
        if st.button("Download Model"):
            with st.spinner("Downloading model from Google Drive..."):
                ok, err = download_model_if_needed()
            if ok:
                st.success("Model downloaded. Refreshing...")
                st.rerun()
            else:
                st.error(err)
    else:
        st.success("Model ready")

    st.markdown("---")

    strength = st.slider("Enhancement strength", 0.0, 1.0, 1.0, 0.05)
    dcp_only = st.toggle("Use DCP only", value=False)
    show_overlay = st.toggle("Hazard overlay", value=False)
    show_metrics = st.toggle("Show metrics", value=True)

    st.markdown("---")
    st.markdown(
        """
        <div class="small-note">
        Prototype mode: DCP + ResnetGenerator.<br>
        DVD temporal model is excluded because it requires custom DCNv2 CUDA extensions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# Auto-download on first load
if not os.path.exists(MODEL_PATH):
    with st.spinner("First run: downloading model from Google Drive..."):
        ok, err = download_model_if_needed()
    if ok:
        st.rerun()
    else:
        st.error(f"Could not download model: {err}")
        st.stop()


# Load model
try:
    with st.spinner("Loading dehazing model..."):
        dcp_model, resnet_model, device, missing_keys, unexpected_keys = load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ============================================================
# Header
# ============================================================

st.markdown("# 👁️ NEXTGEN VISION AI")
st.markdown("### Real-Time AR Vision Enhancement System — Streamlit Prototype")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-val">READY</div><div class="metric-lbl">Model Status</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{str(device).upper()}</div><div class="metric-lbl">Compute Device</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-val">DCP + RESNET</div><div class="metric-lbl">Pipeline</div></div>', unsafe_allow_html=True)

st.markdown("")

with st.expander("Debug model loading"):
    st.write("Missing keys:", len(missing_keys))
    st.write("Unexpected keys:", len(unexpected_keys))
    if missing_keys:
        st.write(missing_keys[:10])
    if unexpected_keys:
        st.write(unexpected_keys[:10])


# ============================================================
# Main upload
# ============================================================

uploaded = st.file_uploader("Upload a hazy/foggy road image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to start dehazing.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
img_rgb = np.array(image)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

start = time.time()
result_bgr = dehaze_image(img_bgr, strength=strength, dcp_only=dcp_only)
elapsed = time.time() - start

display_bgr = result_bgr
hazard_count = 0
if show_overlay:
    display_bgr, hazard_count = add_hazard_overlay(result_bgr)

result_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)

col1, col2 = st.columns(2)

with col1:
    st.image(img_rgb, caption="Original Input", use_container_width=True)

with col2:
    st.image(result_rgb, caption="Enhanced Output", use_container_width=True)

if show_metrics:
    orig_score, orig_contrast, orig_sharp = auto_visibility_score(img_bgr)
    enh_score, enh_contrast, enh_sharp = auto_visibility_score(result_bgr)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original Visibility", f"{orig_score}%")
    m2.metric("Enhanced Visibility", f"{enh_score}%")
    m3.metric("Processing Time", f"{elapsed:.2f}s")
    m4.metric("Hazards Highlighted", hazard_count if show_overlay else "OFF")

    st.markdown("#### Technical Metrics")
    st.write(
        {
            "original_contrast": orig_contrast,
            "enhanced_contrast": enh_contrast,
            "original_sharpness": orig_sharp,
            "enhanced_sharpness": enh_sharp,
        }
    )
