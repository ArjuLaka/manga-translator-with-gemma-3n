@st.cache_resource
def load_model():
    return YOLO(Path("models/bubble_detector.pt"))