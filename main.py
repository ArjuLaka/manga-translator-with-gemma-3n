import streamlit as st
from PIL import Image
import numpy as np
import tempfile
from pathlib import Path
from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
##from wordsegment import load, segment
##load()

@st.cache_resource
def load_model():
    return YOLO(Path("models/bubble_detector.pt"))
    
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_doc_orientation_classify=False,  use_doc_unwarping=False,  use_textline_orientation=False, lang='en')  # gunakan flag baru

model = load_model()
ocr = load_ocr()

st.title("Manga Translator: YOLOv8 + PaddleOCR")
uploaded_file = st.file_uploader("Unggah gambar manga", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar asli", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        path = Path(tmp.name)
        image.save(path)
        results = model(path)
        img_boxes = results[0].plot()
        st.image(img_boxes, caption="Deteksi Bubble", use_container_width=True)

        st.subheader("OCR setiap bubble:")
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crop = image.crop((x1, y1, x2, y2))
            crop_np = np.array(crop)
            bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)
            ocr_img = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            result = ocr.predict(ocr_img)

            final = ""

            for res in result:  
                res.print()
                texts = res["rec_texts"]
                final = " ".join(texts)
                ##final = " ".join([" ".join(ws.segment(text)) for text in texts])

            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(crop, caption=f"Bubble #{i+1}", width=150)
            with col2:
                st.markdown(f"**Teks OCR:** {final if final else '*Tidak terbaca*'}")
