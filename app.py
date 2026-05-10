import os
import time
import cv2
import gdown
import torch
import functools
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

st.set_page_config(page_title="NEXTGEN VISION AI", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

DEHAZE_MODEL_PATH = "remove_hazy_model_256x256.pth"
DEHAZE_GDRIVE_ID = "1ji3x-KO19X2yGpT7oaUIpJ5DiCgQg8xS"
YOLO_MODEL_NAME = "yolov8n.pt"

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root{--bg:#020617;--panel:#0f172a;--border:rgba(148,163,184,.18);--cyan:#22d3ee;--blue:#38bdf8;--green:#34d399;--text:#e5f4ff;--muted:#94a3b8;--warning:#fbbf24}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}.stApp{background:radial-gradient(circle at top left,rgba(34,211,238,.14),transparent 34%),radial-gradient(circle at top right,rgba(52,211,153,.10),transparent 31%),linear-gradient(135deg,#020617 0%,#050b18 55%,#020617 100%)!important;color:var(--text)!important}.block-container{max-width:1350px!important;padding-top:1.8rem!important;padding-bottom:3rem!important}header[data-testid="stHeader"]{background:transparent!important}[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(15,23,42,.98),rgba(2,6,23,.98))!important;border-right:1px solid var(--border)!important}h1{font-size:3.2rem!important;line-height:1.02!important;font-weight:900!important;letter-spacing:-.07em!important;color:var(--text)!important;margin-bottom:.2rem!important}h2,h3{color:var(--text)!important;font-weight:800!important;letter-spacing:-.04em!important}.hero-subtitle{color:var(--muted);font-size:1.05rem;margin-top:-.4rem;margin-bottom:1.4rem;max-width:850px;line-height:1.65}.hero-badge{display:inline-flex;align-items:center;gap:.45rem;background:rgba(34,211,238,.10);border:1px solid rgba(34,211,238,.25);color:#cffafe;padding:.38rem .7rem;border-radius:999px;font-size:.78rem;font-weight:700;margin-right:.45rem;margin-bottom:.6rem}.metric-card{position:relative;overflow:hidden;background:linear-gradient(180deg,rgba(15,23,42,.96),rgba(15,23,42,.72))!important;border:1px solid var(--border)!important;border-radius:22px!important;padding:1.1rem 1rem!important;text-align:left!important;min-height:105px;box-shadow:0 18px 55px rgba(0,0,0,.25)}.metric-card:after{content:"";position:absolute;height:3px;left:18px;right:18px;top:0;background:linear-gradient(90deg,var(--cyan),var(--green));border-radius:999px}.metric-val{position:relative;color:var(--text)!important;font-size:1.22rem!important;font-weight:900!important;letter-spacing:-.03em!important;margin-top:.25rem}.metric-lbl{position:relative;color:var(--muted)!important;font-size:.72rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;margin-top:.45rem}.small-note{color:var(--muted);font-size:.86rem;line-height:1.7;background:rgba(15,23,42,.56);border:1px solid var(--border);border-radius:16px;padding:.85rem}.warning-box{background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.28);color:#fde68a;padding:1rem 1.1rem;border-radius:18px;margin-bottom:1rem}.info-box{background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.25);color:#cffafe;padding:1rem 1.1rem;border-radius:18px;margin-bottom:1rem}[data-testid="stImage"] img,video{border-radius:22px!important;border:1px solid var(--border)!important;box-shadow:0 20px 70px rgba(0,0,0,.35)}[data-testid="stFileUploader"]{background:rgba(15,23,42,.72)!important;border:1px dashed rgba(34,211,238,.40)!important;border-radius:22px!important;padding:1rem!important}.stButton>button{background:linear-gradient(135deg,var(--cyan),var(--blue))!important;color:#020617!important;border:0!important;border-radius:14px!important;font-weight:900!important;box-shadow:0 12px 30px rgba(34,211,238,.22)!important}.stSelectbox div[data-baseweb="select"]>div,.stTextInput input,.stNumberInput input{background:rgba(15,23,42,.92)!important;border:1px solid var(--border)!important;border-radius:14px!important;color:var(--text)!important}.stAlert{background:rgba(15,23,42,.84)!important;border:1px solid var(--border)!important;border-radius:18px!important}.stDeployButton{display:none!important}
</style>
''', unsafe_allow_html=True)

def download_dehaze_model_if_needed():
    if os.path.exists(DEHAZE_MODEL_PATH):
        return True, None
    try:
        gdown.download(f"https://drive.google.com/uc?id={DEHAZE_GDRIVE_ID}", DEHAZE_MODEL_PATH, quiet=False)
        return (os.path.exists(DEHAZE_MODEL_PATH), None if os.path.exists(DEHAZE_MODEL_PATH) else "Downloaded but file not found")
    except Exception as e:
        return False, str(e)

class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3):
        super().__init__(); self.r=r; self.eps=eps; self.boxfilter=nn.AvgPool2d(2*r+1,1,r)
    def forward(self,I,p):
        N=self.boxfilter(torch.ones(p.size(),device=p.device,dtype=p.dtype)); mean_I=self.boxfilter(I)/N; mean_p=self.boxfilter(p)/N; mean_Ip=self.boxfilter(I*p)/N; cov_Ip=mean_Ip-mean_I*mean_p; mean_II=self.boxfilter(I*I)/N; var_I=mean_II-mean_I*mean_I; a=cov_Ip/(var_I+self.eps); b=mean_p-a*mean_I; return (self.boxfilter(a)/N)*I+self.boxfilter(b)/N
class DCPDehazeGenerator(nn.Module):
    def __init__(self, win_size=15, r=40, eps=1e-3):
        super().__init__(); self.guided_filter=GuidedFilter(r,eps); self.neighborhood_size=win_size; self.omega=.95
    def get_dark_channel(self,img,w):
        img,_=torch.min(img,dim=1); img=torch.unsqueeze(img,dim=1); p=int(np.floor(w/2)); pads=[p,p,p,p] if w%2 else [p,p-1,p,p-1]; return -F.max_pool2d(-F.pad(img,pads,mode='replicate'),kernel_size=w,stride=1)
    def atmospheric_light(self,img,dark_img):
        num,chl,h,w=img.shape; top=max(int(.001*h*w),1); A=torch.zeros(num,chl,1,1,device=img.device,dtype=img.dtype)
        for n in range(num):
            _,idx=dark_img[n,0].reshape(h*w).sort(descending=True)
            for c in range(chl): A[n,c,0,0]=torch.mean(img[n,c].reshape(h*w)[idx[:top]])
        return A
    def forward(self,x):
        guidance=(.2989*x[:,0]+.5870*x[:,1]+.1140*x[:,2]) if x.shape[1]>1 else x[:,0]; guidance=torch.unsqueeze((guidance+1)/2,1); img=(x+1)/2; _,_,h,w=img.shape; dark=self.get_dark_channel(img,self.neighborhood_size); A=self.atmospheric_light(img,dark); map_A=A.repeat(1,1,h,w).clamp(min=1e-6); trans=1-self.omega*self.get_dark_channel(img/map_A,self.neighborhood_size); trans=trans.clamp(.05,1); T=self.guided_filter(guidance,trans).clamp(.05,1); return ((img-map_A)/T.repeat(1,3,1,1)+map_A).clamp(0,1)
class ResnetBlock(nn.Module):
    def __init__(self,dim,padding_type,norm_layer,use_dropout,use_bias):
        super().__init__(); block=[]
        for i in range(2):
            block += [nn.ReflectionPad2d(1)] if padding_type=='reflect' else []; p=0 if padding_type=='reflect' else 1
            block += [nn.Conv2d(dim,dim,3,1,p,bias=use_bias), norm_layer(dim)]
            if i==0: block += [nn.ReLU(True)] + ([nn.Dropout(.5)] if use_dropout else [])
        self.conv_block=nn.Sequential(*block)
    def forward(self,x): return x+self.conv_block(x)
class ResnetGenerator(nn.Module):
    def __init__(self,input_nc,output_nc,ngf=64,norm_layer=nn.BatchNorm2d,use_dropout=False,n_blocks=9,padding_type='reflect'):
        super().__init__(); use_bias=norm_layer==nn.InstanceNorm2d; model=[nn.ReflectionPad2d(3),nn.Conv2d(input_nc,ngf,7,padding=0,bias=use_bias),norm_layer(ngf),nn.ReLU(True)]
        for i in range(2):
            m=2**i; model += [nn.Conv2d(ngf*m,ngf*m*2,3,2,1,bias=use_bias),norm_layer(ngf*m*2),nn.ReLU(True)]
        for _ in range(n_blocks): model += [ResnetBlock(ngf*4,padding_type,norm_layer,use_dropout,use_bias)]
        for i in range(2):
            m=2**(2-i); model += [nn.ConvTranspose2d(ngf*m,int(ngf*m/2),3,2,1,output_padding=1,bias=use_bias),norm_layer(int(ngf*m/2)),nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),nn.Conv2d(ngf,output_nc,7,padding=0,bias=use_bias),nn.Tanh()]; self.model=nn.Sequential(*model)
    def forward(self,x): return torch.clamp(self.model(x),-1,1)

@st.cache_resource(show_spinner=False)
def load_dehaze_models():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); dcp=DCPDehazeGenerator().to(device).eval(); resnet=ResnetGenerator(3,3,norm_layer=nn.InstanceNorm2d).to(device); ckpt=torch.load(DEHAZE_MODEL_PATH,map_location=device); sd=ckpt[next((k for k in ['params','state_dict','model','net_g','generator'] if isinstance(ckpt,dict) and k in ckpt), None)] if isinstance(ckpt,dict) and any(k in ckpt for k in ['params','state_dict','model','net_g','generator']) else ckpt; sd={k.replace('module.',''):v for k,v in sd.items()}; missing,unexpected=resnet.load_state_dict(sd,strict=False); resnet.eval(); return dcp,resnet,device,missing,unexpected
@st.cache_resource(show_spinner=False)
def load_yolo_model(): return YOLO(YOLO_MODEL_NAME) if YOLO_AVAILABLE else None

def bgr_to_tensor(img,size):
    img=cv2.resize(img,(size,size),interpolation=cv2.INTER_CUBIC); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.; return torch.from_numpy(rgb.transpose(2,0,1)).float().unsqueeze(0)*2-1
def tensor_to_bgr(tensor,hw):
    out=tensor.squeeze(0).detach().cpu().clamp(0,1).numpy().transpose(1,2,0); bgr=cv2.cvtColor((out*255).round().astype(np.uint8),cv2.COLOR_RGB2BGR); h,w=hw; return cv2.resize(bgr,(w,h),interpolation=cv2.INTER_CUBIC)
def dehaze_image(img_bgr,strength=1.0,dcp_only=False,inference_size=192):
    h,w=img_bgr.shape[:2]; dcp,resnet,device,_,_=load_dehaze_models(); x=bgr_to_tensor(img_bgr,inference_size).to(device)
    with torch.no_grad():
        dcp_out=dcp(x); refined=dcp_out if dcp_only else (resnet(dcp_out)+1)/2; result=tensor_to_bgr(refined,(h,w))
    return cv2.addWeighted(img_bgr,1-strength,result,strength,0) if strength<1 else result
DRIVING_CLASSES={'person','bicycle','car','motorcycle','bus','truck','traffic light','stop sign'}
def detect_objects_yolo(img_bgr,conf_threshold=.35,only_driving_classes=True,draw_ar_style=True):
    model=load_yolo_model();
    if model is None: return img_bgr,[]
    res=model(img_bgr,conf=conf_threshold,verbose=False)[0]; annotated=img_bgr.copy(); det=[]
    if res.boxes is None: return annotated,det
    for box in res.boxes:
        cls=int(box.cls[0]); conf=float(box.conf[0]); name=model.names[cls]
        if only_driving_classes and name not in DRIVING_CLASSES: continue
        x1,y1,x2,y2=box.xyxy[0].cpu().numpy().astype(int); det.append({'class':name,'confidence':round(conf,2),'box':[int(x1),int(y1),int(x2),int(y2)]})
        color=(0,255,150) if name not in ['person','motorcycle','bicycle'] else (0,90,255)
        if name in ['traffic light','stop sign']: color=(0,212,255)
        cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2); label=f'{name.upper()} {conf:.2f}'; ly=max(y1-10,25); cv2.rectangle(annotated,(x1,ly-24),(x1+max(130,len(label)*12),ly+5),color,-1); cv2.putText(annotated,label,(x1+6,ly-5),cv2.FONT_HERSHEY_SIMPLEX,.55,(2,6,23),2,cv2.LINE_AA); cv2.circle(annotated,(int((x1+x2)/2),int((y1+y2)/2)),4,color,-1)
    return annotated,det
def draw_system_overlay(img_bgr,mode='IMAGE',fps=None,inference_time=None,detection_count=0):
    out=img_bgr.copy(); cv2.putText(out,f'NEXTGEN VISION AI | {mode}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,255,180),2,cv2.LINE_AA); line=f'Objects: {detection_count}'; line += f' | FPS: {fps:.1f}' if fps is not None else ''; line += f' | Time: {inference_time:.2f}s' if inference_time is not None else ''; cv2.putText(out,line,(15,60),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,212,255),2,cv2.LINE_AA); return out
def visibility_score(img_bgr):
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY); contrast=float(gray.std()); sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var()); return min(100,max(0,int(contrast*1.4+(sharp**.5)*2))),round(contrast,2),round(sharp,2)
def process_pipeline(frame,strength,dcp_only,inference_size,enable_detection,conf,only_classes,ar_style,mode):
    t0=time.time(); dehazed=dehaze_image(frame,strength,dcp_only,inference_size); dt=time.time()-t0; t1=time.time(); final,dets=detect_objects_yolo(dehazed,conf,only_classes,ar_style) if enable_detection else (dehazed,[]); yt=time.time()-t1; return dehazed,draw_system_overlay(final,mode=mode,inference_time=dt+yt,detection_count=len(dets)),dets,dt,yt

with st.sidebar:
    st.markdown('### ⚙️ System Control')
    st.success('Dehazing model ready' if os.path.exists(DEHAZE_MODEL_PATH) else 'Dehazing model will download')
    st.success('YOLO available' if YOLO_AVAILABLE else 'YOLO missing')
    st.markdown('---')
    app_mode=st.radio('Demo Mode',['🖼️ Image Upload','🎞️ Video Upload','📹 Live Camera'],index=0)
    st.markdown('---'); st.markdown('### Enhancement')
    strength=st.slider('Enhancement strength',0.0,1.0,1.0,.05); dcp_only=st.toggle('DCP only mode',value=False); inference_size=st.selectbox('Dehazing inference size',[128,192,256],index=1)
    st.markdown('---'); st.markdown('### Object Detection')
    enable_detection=st.toggle('Enable YOLO detection',value=True); conf_threshold=st.slider('YOLO confidence',.10,.90,.35,.05); only_driving_classes=st.toggle('Driving classes only',value=True); draw_ar_style=st.toggle('AR-style overlay',value=True)
    st.markdown('---'); st.markdown('### Video Settings')
    video_max_frames=st.slider('Max video frames to process',10,180,60,10); video_frame_skip=st.slider('Process every Nth frame',1,10,3,1)
    st.markdown('---'); st.markdown('<div class="small-note">Best flow: Image mode for quality, Video mode for full pipeline, Live Camera as prototype.</div>',unsafe_allow_html=True)

if not os.path.exists(DEHAZE_MODEL_PATH):
    with st.spinner('Downloading dehazing model from Google Drive...'):
        ok,err=download_dehaze_model_if_needed()
    if ok: st.rerun()
    else: st.error(f'Could not download model: {err}'); st.stop()
try:
    with st.spinner('Loading AI models...'):
        dcp_model,resnet_model,device,missing_keys,unexpected_keys=load_dehaze_models()
        if enable_detection and YOLO_AVAILABLE: yolo_model=load_yolo_model()
except Exception as e: st.error(f'Model loading failed: {e}'); st.stop()

st.markdown('# 👁️ NEXTGEN VISION AI')
st.markdown('<div class="hero-subtitle">Real-Time AR Vision Enhancement System for adverse weather driving conditions. This prototype demonstrates visibility restoration, hazard detection, AR-style overlays, and performance metrics.</div><span class="hero-badge">DCP + ResNet Dehazing</span><span class="hero-badge">YOLOv8 Object Detection</span><span class="hero-badge">Image · Video · Live Camera</span>',unsafe_allow_html=True)
c1,c2,c3,c4=st.columns(4)
c1.markdown('<div class="metric-card"><div class="metric-val">READY</div><div class="metric-lbl">Dehazing Model</div></div>',unsafe_allow_html=True); c2.markdown(f'<div class="metric-card"><div class="metric-val">{str(device).upper()}</div><div class="metric-lbl">Compute Device</div></div>',unsafe_allow_html=True); c3.markdown(f'<div class="metric-card"><div class="metric-val">{"ON" if enable_detection else "OFF"}</div><div class="metric-lbl">YOLO Detection</div></div>',unsafe_allow_html=True); c4.markdown(f'<div class="metric-card"><div class="metric-val">{inference_size}px</div><div class="metric-lbl">Inference Size</div></div>',unsafe_allow_html=True)

if app_mode=='🖼️ Image Upload':
    st.markdown('## 🖼️ Image Upload Demo'); st.markdown('<div class="info-box">Use this mode for the cleanest before/after result during presentation.</div>',unsafe_allow_html=True)
    uploaded=st.file_uploader('Upload a hazy/foggy road image',type=['jpg','jpeg','png'])
    if uploaded:
        rgb=np.array(Image.open(uploaded).convert('RGB')); bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR); deh,final,dets,dt,yt=process_pipeline(bgr,strength,dcp_only,inference_size,enable_detection,conf_threshold,only_driving_classes,draw_ar_style,'IMAGE')
        col1,col2,col3=st.columns(3); col1.image(rgb,caption='Original Input',use_container_width=True); col2.image(cv2.cvtColor(deh,cv2.COLOR_BGR2RGB),caption='Dehazed Output',use_container_width=True); col3.image(cv2.cvtColor(final,cv2.COLOR_BGR2RGB),caption='Final AR Detection Output',use_container_width=True)
        oscore,_,_=visibility_score(bgr); escore,_,_=visibility_score(deh); m1,m2,m3,m4=st.columns(4); m1.metric('Original Visibility',f'{oscore}%'); m2.metric('Enhanced Visibility',f'{escore}%'); m3.metric('Processing Time',f'{dt+yt:.2f}s'); m4.metric('Objects Detected',len(dets))
        st.dataframe(pd.DataFrame(dets),use_container_width=True) if dets else st.info('No driving-related objects detected.')
    else: st.info('Upload an image to start.')
elif app_mode=='🎞️ Video Upload':
    st.markdown('## 🎞️ Video Upload Full Pipeline Demo'); st.markdown('<div class="info-box">Best mode for presentation: it shows dehazing + object detection together without webcam lag.</div>',unsafe_allow_html=True)
    uploaded_video=st.file_uploader('Upload a hazy/foggy road video',type=['mp4','avi','mov','mkv'])
    if uploaded_video:
        inp=tempfile.NamedTemporaryFile(delete=False,suffix='.mp4'); inp.write(uploaded_video.read()); inp.close(); st.video(inp.name)
        if st.button('▶ Process Video'):
            cap=cv2.VideoCapture(inp.name); fps=cap.get(cv2.CAP_PROP_FPS); fps=fps if fps and fps>0 else 10; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); out_path=tempfile.NamedTemporaryFile(delete=False,suffix='.mp4').name; out=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'mp4v'),max(1,fps/video_frame_skip),(w,h)); prog=st.progress(0); status=st.empty(); preview=st.empty(); idx=0; processed=0; total_det=0; start=time.time()
            while cap.isOpened() and processed<video_max_frames:
                ret,frame=cap.read();
                if not ret: break
                idx+=1
                if idx%video_frame_skip!=0: continue
                deh,final,dets,dt,yt=process_pipeline(frame,strength,dcp_only,inference_size,enable_detection,conf_threshold,only_driving_classes,draw_ar_style,'VIDEO'); total_det+=len(dets); out.write(final); processed+=1
                if processed%3==0: preview.image(cv2.cvtColor(final,cv2.COLOR_BGR2RGB),caption=f'Processing frame {processed}',use_container_width=True)
                prog.progress(min(processed/video_max_frames,1.0)); status.write(f'Processed {processed}/{video_max_frames} frames · Latest detections: {len(dets)}')
            cap.release(); out.release(); total=time.time()-start; st.success('Video processing completed.'); st.video(out_path); m1,m2,m3,m4=st.columns(4); m1.metric('Frames Processed',processed); m2.metric('Total Time',f'{total:.1f}s'); m3.metric('Avg Time / Frame',f'{total/max(processed,1):.2f}s'); m4.metric('Total Detections',total_det)
            with open(out_path,'rb') as f: st.download_button('⬇ Download Processed Video',data=f,file_name='nextgen_vision_processed_video.mp4',mime='video/mp4')
    else: st.info('Upload a road video to process.')
else:
    st.markdown('## 📹 Live Camera Prototype'); st.markdown('<div class="warning-box">Live camera is experimental on Streamlit Cloud. Use Video Upload for the stable full pipeline presentation.</div>',unsafe_allow_html=True)
    if not WEBRTC_AVAILABLE: st.error('streamlit-webrtc is not installed. Make sure requirements.txt includes streamlit-webrtc.'); st.stop()
    rtc_config=RTCConfiguration({'iceServers':[{'urls':['stun:stun.l.google.com:19302']}]})
    class LiveProcessor(VideoProcessorBase):
        def __init__(self): self.last_time=time.time(); self.fps=0.; self.frame_count=0; self.cached_frame=None; self.cached_dets=[]
        def recv(self,frame):
            img=frame.to_ndarray(format='bgr24'); start=time.time()
            try:
                deh=dehaze_image(img,strength,dcp_only,128); self.frame_count+=1
                if enable_detection and self.frame_count%5==0: final,dets=detect_objects_yolo(deh,conf_threshold,only_driving_classes,draw_ar_style); self.cached_frame=final.copy(); self.cached_dets=dets
                elif self.cached_frame is not None: final=self.cached_frame.copy(); dets=self.cached_dets
                else: final=deh; dets=[]
                now=time.time(); dt=now-self.last_time; self.last_time=now; self.fps=1/dt if dt>0 else self.fps; final=draw_system_overlay(final,'LIVE',self.fps,time.time()-start,len(dets))
            except Exception: final=img
            return av.VideoFrame.from_ndarray(final,format='bgr24')
    webrtc_streamer(key='nextgen-live-camera',video_processor_factory=LiveProcessor,rtc_configuration=rtc_config,media_stream_constraints={'video':{'width':{'ideal':240},'height':{'ideal':180},'frameRate':{'ideal':8,'max':10}},'audio':False},async_processing=True)
with st.expander('Developer Debug'):
    st.write('Dehazing missing keys:',len(missing_keys)); st.write('Dehazing unexpected keys:',len(unexpected_keys)); st.write('YOLO available:',YOLO_AVAILABLE); st.write('WebRTC available:',WEBRTC_AVAILABLE)
