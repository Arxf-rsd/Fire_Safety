import streamlit as st
import cv2
import PIL
import numpy as np
import time
import threading
from camera_input_live import camera_input_live
import utils  # your fire/smoke detection utilities

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(
    page_title="Fire/Smoke Detection",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🔥 Fire/Smoke Detection App")

# ----------------------------
# Sidebar options
# ----------------------------
source_radio = st.sidebar.radio("Select Input Source:", ["IMAGE", "VIDEO", "WEBCAM"])

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=10, max_value=100, value=40, step=5
) / 100

# ----------------------------
# Cache the model so it loads only once
# ----------------------------
@st.cache_resource
def load_model():
    return utils.load_model() if hasattr(utils, "load_model") else None

model = load_model()

# ----------------------------
# Helper class for threaded video capture
# ----------------------------
class VideoStream:
    def __init__(self, source):
        self.stream = cv2.VideoCapture(source)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stop()
                return
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ----------------------------
# Function: play video with FPS control
# ----------------------------
def play_video(video_source):
    vs = VideoStream(video_source)
    st_frame = st.empty()

    prev_time = 0
    fps_limit = 15  # target FPS
    resize_dim = (640, 480)  # lower size for speed

    while True:
        ret, frame = vs.read()
        if not ret or frame is None:
            break

        current_time = time.time()
        if (current_time - prev_time) > 2.0 / fps_limit:
            prev_time = current_time

            frame_resized = cv2.resize(frame, resize_dim)
            visualized_image, _ = utils.predict_image(frame_resized, conf_threshold)
            st_frame.image(visualized_image, channels="BGR", use_container_width=True)

    vs.stop()

# ----------------------------
# Function: play live camera
# ----------------------------
def play_live_camera():
    image = camera_input_live()
    if image is not None:
        uploaded_image = PIL.Image.open(image)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        uploaded_image_cv = cv2.resize(uploaded_image_cv, (640, 480))
        visualized_image, _ = utils.predict_image(uploaded_image_cv, conf_threshold)
        st.image(visualized_image, channels="BGR")

# ----------------------------
# IMAGE INPUT
# ----------------------------
if source_radio == "IMAGE":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file:
        uploaded_image = PIL.Image.open(uploaded_file)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        uploaded_image_cv = cv2.resize(uploaded_image_cv, (640, 480))
        visualized_image, _ = utils.predict_image(uploaded_image_cv, conf_threshold)
        st.image(visualized_image, channels="BGR")
    else:
        st.image("assets/sample.jpg")
        st.write("Upload an image from the sidebar to run detection.")

# ----------------------------
# VIDEO INPUT
# ----------------------------
elif source_radio == "VIDEO":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4"])
    temp_video_path = None

    if uploaded_file:
        temp_video_path = "temp_upload.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

    if temp_video_path:
        play_video(temp_video_path)
    else:
        st.video("assets/sample.mp4")
        st.write("Upload a video from the sidebar to run detection.")

# ----------------------------
# WEBCAM INPUT
# ----------------------------
elif source_radio == "WEBCAM":
    play_live_camera()
