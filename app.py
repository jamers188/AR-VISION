import streamlit as st

st.set_page_config(page_title="Diagnostic", layout="wide")
st.title("Step 1: Streamlit OK")

import sys
st.success(f"Python {sys.version}")

st.write("Testing numpy...")
import numpy as np
st.success(f"numpy {np.__version__}")

st.write("Testing pandas...")
import pandas as pd
st.success(f"pandas {pd.__version__}")

st.write("Testing Pillow...")
from PIL import Image
import PIL
st.success(f"Pillow {PIL.__version__}")

st.write("Testing requests...")
import requests
st.success("requests OK")

st.write("Testing cv2...")
try:
    import cv2
    st.success(f"cv2 {cv2.__version__}")
except Exception as e:
    st.error(f"cv2 FAILED: {e}")

st.write("Testing torch...")
try:
    import torch
    st.success(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
except Exception as e:
    st.error(f"torch FAILED: {e}")

st.write("Testing gdown...")
try:
    import gdown
    st.success("gdown OK")
except Exception as e:
    st.error(f"gdown FAILED: {e}")

st.write("Testing ultralytics...")
try:
    from ultralytics import YOLO
    st.success("ultralytics OK")
except Exception as e:
    st.error(f"ultralytics FAILED: {e}")

st.write("Testing av...")
try:
    import av
    st.success(f"av {av.__version__}")
except Exception as e:
    st.error(f"av FAILED: {e}")

st.write("Testing streamlit_webrtc...")
try:
    from streamlit_webrtc import webrtc_streamer
    st.success("streamlit_webrtc OK")
except Exception as e:
    st.error(f"streamlit_webrtc FAILED: {e}")

st.balloons()
st.success("ALL DONE — copy the errors above and share them")
