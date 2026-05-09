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

# YOLO object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Real-time webcam
try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


# ============================================================
# NEXTGEN VISION AI
# One-file Streamlit prototype:
# Image Upload + Real-Time Webcam + Dehazing + Object Detection
#
# Pipeline:
# Input Frame -> DCP + Resnet Dehazing -> YOLO Object Detection -> AR-style Overlay
#
# Note:
# Full DVD/NSDNGAN is not used here because it depends on custom DCNv2 CUDA extensions,
# which usually break on Streamlit Cloud.
# ============================================================

st.set_page_config(
    page_title="NEXTGEN VISION AI",
    page_icon="👁️",
    layout="wide",
)


# ============================================================
# MODEL DOWNLOAD CONFIG
# ============================================================

DEHAZE_MODEL_PATH = "remove_hazy_model_256x256.pth"
DEHAZE_GDRIVE_ID = "1ji3x-KO19X2yGpT7oaUIpJ5DiCgQg8xS"

YOLO_MODEL_NAME = "yolov8n.pt"


def download_dehaze_model_if_needed():
    if os.path.exists(DEHAZE_MODEL_PATH):
        return True, None

    try:
        url = f"https://drive.google.com/uc?id={DEHAZE_GDRIVE_ID}"
        gdown.download(url, DEHAZE_MODEL_PATH, quiet=False)

        if os.path.exists(DEHAZE_MODEL_PATH):
            return True, None

        return False, "Download finished, but dehazing model file was not found."
    except Exception as e:
        return False, str(e)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #020617;
    --panel: #0f172a;
    --panel2: #111827;
    --card: rgba(15, 23, 42, 0.88);
    --border: rgba(148, 163, 184, 0.18);
    --cyan: #22d3ee;
    --blue: #38bdf8;
    --green: #34d399;
    --text: #e5f4ff;
    --muted: #94a3b8;
    --danger: #fb7185;
    --warning: #fbbf24;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(34, 211, 238, 0.13), transparent 33%),
        radial-gradient(circle at top right, rgba(52, 211, 153, 0.10), transparent 32%),
        linear-gradient(135deg, #020617 0%, #050b18 50%, #020617 100%) !important;
    color: var(--text) !important;
}

/* Main page spacing */
.block-container {
    max-width: 1280px !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* Hide ugly Streamlit chrome spacing */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(2, 6, 23, 0.98)) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem !important;
}

[data-testid="stSidebar"] h3 {
    color: var(--cyan) !important;
    font-size: 0.95rem !important;
    letter-spacing: -0.02em !important;
}

/* Headings */
h1 {
    font-size: 3rem !important;
    line-height: 1.05 !important;
    font-weight: 900 !important;
    letter-spacing: -0.06em !important;
    color: var(--text) !important;
    margin-bottom: 0.3rem !important;
}

h1::first-letter {
    color: var(--cyan);
}

h2, h3 {
    color: var(--text) !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
}

p, label, span, div {
    color: inherit;
}

/* Subtitle under title */
h3 {
    color: var(--muted) !important;
    font-size: 1.18rem !important;
}

/* Metric cards */
.metric-card {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.72)) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 1.1rem 1rem !important;
    text-align: left !important;
    min-height: 105px;
    box-shadow: 0 18px 55px rgba(0, 0, 0, 0.25);
}

.metric-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.16), transparent 45%);
    pointer-events: none;
}

.metric-card::after {
    content: "";
    position: absolute;
    height: 3px;
    left: 18px;
    right: 18px;
    top: 0;
    background: linear-gradient(90deg, var(--cyan), var(--green));
    border-radius: 999px;
}

.metric-val {
    position: relative;
    color: var(--text) !important;
    font-size: 1.25rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.03em !important;
    margin-top: 0.25rem;
}

.metric-lbl {
    position: relative;
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin-top: 0.45rem;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(15, 23, 42, 0.78) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text) !important;
}

[data-testid="stExpander"] {
    border: 0 !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.74) !important;
    border: 1px dashed rgba(34, 211, 238, 0.45) !important;
    border-radius: 22px !important;
    padding: 1.2rem !important;
}

[data-testid="stFileUploader"] section {
    background: rgba(2, 6, 23, 0.35) !important;
    border: 0 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.72) !important;
    border: 1px dashed rgba(34, 211, 238, 0.35) !important;
    border-radius: 18px !important;
}

/* Buttons */
.stButton > button,
[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg, var(--cyan), var(--blue)) !important;
    color: #020617 !important;
    border: 0 !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    box-shadow: 0 12px 30px rgba(34, 211, 238, 0.22) !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.05);
}

/* Inputs */
.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {
    background: rgba(15, 23, 42, 0.92) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] > div {
    color: var(--cyan) !important;
}

.stSlider [role="slider"] {
    background: var(--cyan) !important;
    box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.18) !important;
}

/* Radio / toggles labels */
.stRadio label,
.stCheckbox label {
    color: var(--muted) !important;
    font-weight: 600 !important;
}

/* Alerts */
.stAlert {
    background: rgba(15, 23, 42, 0.84) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
}

/* Images */
[data-testid="stImage"] img {
    border-radius: 20px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 18px 60px rgba(0,0,0,0.35);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 18px !important;
    overflow: hidden !important;
}

/* Custom boxes */
.small-note {
    color: var(--muted);
    font-size: 0.86rem;
    line-height: 1.7;
    background: rgba(15, 23, 42, 0.56);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.85rem;
}

.warning-box {
    background: rgba(251, 191, 36, 0.10);
    border: 1px solid rgba(251, 191, 36, 0.28);
    color: #fde68a;
    padding: 1rem 1.1rem;
    border-radius: 18px;
    margin-bottom: 1rem;
}

.info-box {
    background: rgba(34, 211, 238, 0.08);
    border: 1px solid rgba(34, 211, 238, 0.25);
    color: #cffafe;
    padding: 1rem 1.1rem;
    border-radius: 18px;
    margin-bottom: 1rem;
}

/* Make success boxes prettier */
[data-testid="stSidebar"] .stAlert {
    background: rgba(16, 185, 129, 0.10) !important;
    border: 1px solid rgba(16, 185, 129, 0.22) !important;
}

/* Top action icons area less ugly */
.viewerBadge_container__1QSob,
.stDeployButton {
    display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DEHAZING ARCHITECTURE
# DCPDehazeGenerator + ResnetGenerator
# ============================================================

class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3):
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = nn.AvgPool2d(
            kernel_size=2 * self.r + 1,
            stride=1,
            padding=self.r
        )

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
        if len(img.shape) != 4:
            raise NotImplementedError("Dark channel only supports 4D tensor [N,C,H,W].")

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
        # x expected range: [-1, 1]
        if x.shape[1] > 1:
            guidance = (
                0.2989 * x[:, 0, :, :] +
                0.5870 * x[:, 1, :, :] +
                0.1140 * x[:, 2, :, :]
            )
        else:
            guidance = x[:, 0, :, :]

        guidance = (guidance + 1) / 2
        guidance = torch.unsqueeze(guidance, dim=1)

        img_patch = (x + 1) / 2
        num, chl, height, width = img_patch.shape

        dark_img = self.get_dark_channel(img_patch, self.neighborhood_size)
        A = self.atmospheric_light(img_patch, dark_img)

        map_A = A.repeat(1, 1, height, width).clamp(min=1e-6)

        trans_raw = 1 - self.omega * self.get_dark_channel(
            img_patch / map_A,
            self.neighborhood_size
        )
        trans_raw = trans_raw.clamp(min=0.05, max=1.0)

        T_DCP = self.guided_filter(guidance, trans_raw).clamp(min=0.05, max=1.0)

        J_DCP = (img_patch - map_A) / T_DCP.repeat(1, 3, 1, 1) + map_A

        return J_DCP.clamp(0, 1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim,
            padding_type,
            norm_layer,
            use_dropout,
            use_bias
        )

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
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
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
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_dehaze_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcp = DCPDehazeGenerator().to(device).eval()

    resnet = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        norm_layer=nn.InstanceNorm2d,
    ).to(device)

    ckpt = torch.load(DEHAZE_MODEL_PATH, map_location=device)

    if isinstance(ckpt, dict):
        key = next(
            (k for k in ["params", "state_dict", "model", "net_g", "generator"] if k in ckpt),
            None
        )
        state_dict = ckpt[key] if key else ckpt
    else:
        state_dict = ckpt

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = resnet.load_state_dict(state_dict, strict=False)

    resnet.eval()

    return dcp, resnet, device, missing, unexpected


@st.cache_resource(show_spinner=False)
def load_yolo_model():
    if not YOLO_AVAILABLE:
        return None
    return YOLO(YOLO_MODEL_NAME)


# ============================================================
# INFERENCE HELPERS
# ============================================================

def bgr_to_tensor_minus1_to_1(img_bgr, size=256):
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor


def tensor_0_1_to_bgr(tensor, original_hw):
    out = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    out = out.transpose(1, 2, 0)

    out_bgr = cv2.cvtColor(
        (out * 255.0).round().astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    h, w = original_hw
    out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    return out_bgr


def dehaze_image(img_bgr, strength=1.0, dcp_only=False, inference_size=256):
    h, w = img_bgr.shape[:2]

    dcp, resnet, device, _, _ = load_dehaze_models()

    x = bgr_to_tensor_minus1_to_1(img_bgr, size=inference_size).to(device)

    with torch.no_grad():
        dcp_out_01 = dcp(x)

        if dcp_only:
            refined_01 = dcp_out_01
        else:
            refined_minus1_1 = resnet(dcp_out_01)
            refined_01 = (refined_minus1_1 + 1.0) / 2.0

        result_bgr = tensor_0_1_to_bgr(refined_01, (h, w))

    if strength < 1.0:
        result_bgr = cv2.addWeighted(
            img_bgr,
            1.0 - strength,
            result_bgr,
            strength,
            0
        )

    return result_bgr


def auto_visibility_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    score = min(
        100,
        max(0, int((contrast * 1.4) + (sharpness ** 0.5) * 2.0))
    )

    return score, round(float(contrast), 2), round(float(sharpness), 2)


# ============================================================
# YOLO OBJECT DETECTION
# ============================================================

DRIVING_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}


def detect_objects_yolo(
    img_bgr,
    conf_threshold=0.35,
    only_driving_classes=True,
    draw_ar_style=True,
):
    """
    Runs YOLO on BGR image and returns:
    annotated image, list of detection dictionaries
    """

    yolo_model = load_yolo_model()

    if yolo_model is None:
        return img_bgr, []

    # YOLO accepts BGR numpy image too.
    results = yolo_model(
        img_bgr,
        conf=conf_threshold,
        verbose=False,
    )

    result = results[0]
    annotated = img_bgr.copy()
    detections = []

    if result.boxes is None:
        return annotated, detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = yolo_model.names[cls_id]

        if only_driving_classes and name not in DRIVING_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        detections.append(
            {
                "class": name,
                "confidence": round(conf, 2),
                "box": [int(x1), int(y1), int(x2), int(y2)],
            }
        )

        if draw_ar_style:
            color = (0, 255, 150)

            if name in ["person", "motorcycle", "bicycle"]:
                color = (0, 80, 255)
            elif name in ["car", "bus", "truck"]:
                color = (0, 255, 150)
            elif name in ["traffic light", "stop sign"]:
                color = (0, 212, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{name.upper()} {conf:.2f}"

            label_y = max(y1 - 10, 25)

            cv2.rectangle(
                annotated,
                (x1, label_y - 22),
                (x1 + max(120, len(label) * 11), label_y + 4),
                color,
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 5, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (5, 10, 18),
                2,
                cv2.LINE_AA,
            )

            # AR-style center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(annotated, (cx, cy), 4, color, -1)

        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 150), 2)
            cv2.putText(
                annotated,
                f"{name} {conf:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 150),
                2,
                cv2.LINE_AA,
            )

    return annotated, detections


def draw_system_overlay(img_bgr, fps=None, inference_time=None, detection_count=0, mode="LIVE"):
    out = img_bgr.copy()

    cv2.putText(
        out,
        f"NEXTGEN VISION AI | {mode}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 180),
        2,
        cv2.LINE_AA,
    )

    second_line = f"Objects: {detection_count}"

    if fps is not None:
        second_line += f" | FPS: {fps:.1f}"

    if inference_time is not None:
        second_line += f" | Inference: {inference_time:.2f}s"

    cv2.putText(
        out,
        second_line,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 212, 255),
        2,
        cv2.LINE_AA,
    )

    return out


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚙ System Configuration")

    if not os.path.exists(DEHAZE_MODEL_PATH):
        st.warning("Dehazing model not found.")
        if st.button("Download Dehazing Model"):
            with st.spinner("Downloading dehazing model from Google Drive..."):
                ok, err = download_dehaze_model_if_needed()

            if ok:
                st.success("Model downloaded.")
                st.rerun()
            else:
                st.error(err)
    else:
        st.success("Dehazing model ready")

    if YOLO_AVAILABLE:
        st.success("YOLO available")
    else:
        st.error("YOLO not installed")

    st.markdown("---")

    app_mode = st.radio(
        "Mode",
        ["Image Upload", "Real-Time Camera"],
        index=0
    )

    st.markdown("---")
    st.markdown("### Dehazing Controls")

    strength = st.slider("Enhancement strength", 0.0, 1.0, 1.0, 0.05)
    dcp_only = st.toggle("Use DCP only", value=False)

    inference_size = st.selectbox(
        "Dehazing inference size",
        [128, 192, 256],
        index=0,
        help="Lower is faster. 192 is recommended for real-time mode."
    )

    st.markdown("---")
    st.markdown("### Object Detection Controls")

    enable_detection = st.toggle("Enable YOLO object detection", value=True)

    conf_threshold = st.slider(
        "YOLO confidence threshold",
        0.10,
        0.90,
        0.35,
        0.05,
    )

    only_driving_classes = st.toggle(
        "Only driving-related classes",
        value=True
    )

    draw_ar_style = st.toggle(
        "AR-style detection overlay",
        value=True
    )

    st.markdown("---")

    show_metrics = st.toggle("Show metrics", value=True)

    st.markdown("---")

    st.markdown(
        """
        <div class="small-note">
        Pipeline: input frame → dehazing → YOLO object detection → AR-style overlay.<br><br>
        Real-time mode uses 320×240 webcam resolution and runs YOLO every 3 frames to reduce delay.<br><br>
        Full DVD/NSDNGAN is excluded here because it requires custom DCNv2 CUDA extensions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DOWNLOAD + LOAD MODELS
# ============================================================

if not os.path.exists(DEHAZE_MODEL_PATH):
    with st.spinner("First run: downloading dehazing model from Google Drive..."):
        ok, err = download_dehaze_model_if_needed()

    if ok:
        st.rerun()
    else:
        st.error(f"Could not download dehazing model: {err}")
        st.stop()


try:
    with st.spinner("Loading dehazing model..."):
        dcp_model, resnet_model, device, missing_keys, unexpected_keys = load_dehaze_models()
except Exception as e:
    st.error(f"Dehazing model loading failed: {e}")
    st.stop()


if enable_detection:
    if not YOLO_AVAILABLE:
        st.error("YOLO is not installed. Add ultralytics to requirements.txt.")
        st.stop()

    try:
        with st.spinner("Loading YOLO model..."):
            yolo_model = load_yolo_model()
    except Exception as e:
        st.error(f"YOLO loading failed: {e}")
        st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("# 👁️ NEXTGEN VISION AI")
st.markdown("### AI-powered visibility enhancement with real-time road object detection")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        '<div class="metric-card"><div class="metric-val">READY</div><div class="metric-lbl">Dehazing Status</div></div>',
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{str(device).upper()}</div><div class="metric-lbl">Compute Device</div></div>',
        unsafe_allow_html=True
    )

with c3:
    yolo_status = "ON" if enable_detection else "OFF"
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{yolo_status}</div><div class="metric-lbl">YOLO Detection</div></div>',
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        '<div class="metric-card"><div class="metric-val">DCP + RESNET + YOLO</div><div class="metric-lbl">Pipeline</div></div>',
        unsafe_allow_html=True
    )

st.markdown("")

with st.expander("Debug model loading"):
    st.write("Missing dehazing keys:", len(missing_keys))
    st.write("Unexpected dehazing keys:", len(unexpected_keys))

    if missing_keys:
        st.write("First missing keys:", missing_keys[:10])

    if unexpected_keys:
        st.write("First unexpected keys:", unexpected_keys[:10])

    st.write("YOLO available:", YOLO_AVAILABLE)
    st.write("YOLO model:", YOLO_MODEL_NAME)


# ============================================================
# IMAGE UPLOAD MODE
# ============================================================

if app_mode == "Image Upload":
    uploaded = st.file_uploader(
        "Upload a hazy/foggy road image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is None:
        st.info("Upload an image to start dehazing and object detection.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    start = time.time()

    dehazed_bgr = dehaze_image(
        img_bgr,
        strength=strength,
        dcp_only=dcp_only,
        inference_size=inference_size,
    )

    dehaze_time = time.time() - start

    detection_start = time.time()
    detections = []

    if enable_detection:
        final_bgr, detections = detect_objects_yolo(
            dehazed_bgr,
            conf_threshold=conf_threshold,
            only_driving_classes=only_driving_classes,
            draw_ar_style=draw_ar_style,
        )
    else:
        final_bgr = dehazed_bgr

    detection_time = time.time() - detection_start

    final_bgr = draw_system_overlay(
        final_bgr,
        fps=None,
        inference_time=dehaze_time + detection_time,
        detection_count=len(detections),
        mode="IMAGE",
    )

    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    dehazed_rgb = cv2.cvtColor(dehazed_bgr, cv2.COLOR_BGR2RGB)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_rgb, caption="Original Input", use_container_width=True)

    with col2:
        st.image(dehazed_rgb, caption="Dehazed Output", use_container_width=True)

    with col3:
        st.image(final_rgb, caption="Final Output with Detection", use_container_width=True)

    if show_metrics:
        orig_score, orig_contrast, orig_sharp = auto_visibility_score(img_bgr)
        enh_score, enh_contrast, enh_sharp = auto_visibility_score(dehazed_bgr)

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Original Visibility", f"{orig_score}%")
        m2.metric("Enhanced Visibility", f"{enh_score}%")
        m3.metric("Dehazing Time", f"{dehaze_time:.2f}s")
        m4.metric("Objects Detected", len(detections))

        st.markdown("#### Detection Results")

        if enable_detection:
            if len(detections) == 0:
                st.info("No driving-related objects detected.")
            else:
                st.dataframe(detections, use_container_width=True)
        else:
            st.info("YOLO detection is disabled.")

        st.markdown("#### Technical Metrics")
        st.write(
            {
                "original_contrast": orig_contrast,
                "enhanced_contrast": enh_contrast,
                "original_sharpness": orig_sharp,
                "enhanced_sharpness": enh_sharp,
                "dehazing_time_seconds": round(dehaze_time, 3),
                "detection_time_seconds": round(detection_time, 3),
                "total_time_seconds": round(dehaze_time + detection_time, 3),
                "inference_size": inference_size,
                "yolo_confidence_threshold": conf_threshold,
            }
        )


# ============================================================
# REAL-TIME CAMERA MODE
# ============================================================

else:
    st.markdown("## 📹 Real-Time Camera: Dehazing + Object Detection")

    st.markdown(
        """
        <div class="warning-box">
        Real-time mode is heavy because every camera frame goes through dehazing and YOLO.
        For smoother FPS on Streamlit Cloud, use inference size 128, 320×240 webcam resolution, and YOLO frame skipping.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not WEBRTC_AVAILABLE:
        st.error(
            "streamlit-webrtc is not installed. Add streamlit-webrtc and av to requirements.txt."
        )
        st.stop()

    rtc_config = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        }
    )

    class RealTimeVisionProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_time = time.time()
            self.fps = 0.0

            # Frame skipping for YOLO.
            # YOLO will run every 3 frames only, which reduces delay.
            self.frame_count = 0

            # Cache the last detection output so skipped frames still show boxes.
            self.cached_detection_frame = None
            self.cached_detections = []

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            start = time.time()

            try:
                # 1) Dehaze every frame
                dehazed = dehaze_image(
                    img,
                    strength=strength,
                    dcp_only=dcp_only,
                    inference_size=inference_size,
                )

                detections = []

                # 2) Run YOLO only every 3 frames to reduce delay
                self.frame_count += 1

                if enable_detection:
                    if self.frame_count % 3 == 0:
                        final, detections = detect_objects_yolo(
                            dehazed,
                            conf_threshold=conf_threshold,
                            only_driving_classes=only_driving_classes,
                            draw_ar_style=draw_ar_style,
                        )

                        self.cached_detection_frame = final.copy()
                        self.cached_detections = detections

                    else:
                        # Reuse previous YOLO output instead of running YOLO again
                        if self.cached_detection_frame is not None:
                            final = self.cached_detection_frame.copy()
                            detections = self.cached_detections
                        else:
                            final = dehazed
                            detections = []
                else:
                    final = dehazed
                    detections = []

                processing_time = time.time() - start

                now = time.time()
                dt = now - self.last_time
                self.last_time = now

                if dt > 0:
                    self.fps = 1.0 / dt

                final = draw_system_overlay(
                    final,
                    fps=self.fps,
                    inference_time=processing_time,
                    detection_count=len(detections),
                    mode="LIVE",
                )

            except Exception:
                final = img.copy()

                cv2.putText(
                    final,
                    "Processing error - showing original frame",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            return av.VideoFrame.from_ndarray(final, format="bgr24")

    webrtc_streamer(
        key="nextgen-realtime-dehaze-detect",
        video_processor_factory=RealTimeVisionProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 320},
                "height": {"ideal": 240},
                "frameRate": {"ideal": 8, "max": 10},
            },
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown(
        """
        <div class="info-box">
        Tip for demo: point the camera at a phone/laptop screen showing a road scene with cars or people.
        YOLO will detect objects after the frame is enhanced.
        </div>
        """,
        unsafe_allow_html=True,
    )
