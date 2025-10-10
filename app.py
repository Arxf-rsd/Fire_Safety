import streamlit as st
import PIL
import cv2
import numpy as np
import utils
import time
from streamlit_camera_input_live import camera_input_live

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(
    page_title="🔥 Fire/Smoke Detection",
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
# Helper: Process Image
# ----------------------------
def process_image(uploaded_image_cv):
    visualized_image, _ = utils.predict_image(uploaded_image_cv, conf_threshold)
    return visualized_image

# ----------------------------
# Helper: Play video file
# ----------------------------
def play_video(video_source):
    camera = cv2.VideoCapture(video_source)
    st_frame = st.empty()

    if not camera.isOpened():
        st.error("Unable to open video source.")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        visualized_image = process_image(frame)
        st_frame.image(visualized_image, channels="BGR")

    camera.release()

# ----------------------------
# Helper: Live webcam (hybrid version)
# ----------------------------
def play_live_camera():
    st_frame = st.empty()
    st.info("🎥 Live camera active. Press **Stop Webcam** to end.")
    stop = st.button("Stop Webcam")

    while not stop:
        image = camera_input_live()
        if image is not None:
            uploaded_image = PIL.Image.open(image).resize((640, 480))
            uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)

            # Run detection
            visualized_image = process_image(uploaded_image_cv)
            st_frame.image(visualized_image, channels="BGR")

        else:
            st.warning("Waiting for camera feed...")
            time.sleep(0.1)

        # Check stop condition
        stop = st.session_state.get("stop_webcam", False)

# ----------------------------
# IMAGE INPUT
# ----------------------------
if source_radio == "IMAGE":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file:
        uploaded_image = PIL.Image.open(uploaded_file)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = process_image(uploaded_image_cv)
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
