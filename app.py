import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Handwritten OCR", layout="wide")

@st.cache_resource
def load_reader():
    return easyocr.Reader(["en"], gpu=False)

def preprocess_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )
    return thresh

def recognize_text(image):
    image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    processed = preprocess_image(image_bgr)

    reader = load_reader()
    results = reader.readtext(processed)

    output = image_bgr.copy()
    detected_text = []

    for box, text, confidence in results:
        points = np.array(box, dtype=np.int32)
        cv2.polylines(output, [points], True, (0, 255, 0), 2)

        x = int(min(point[0] for point in box))
        y = int(min(point[1] for point in box))

        cv2.putText(
            output,
            text,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        detected_text.append({
            "Text": text,
            "Confidence": round(float(confidence), 3)
        })

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output_rgb, processed, detected_text

st.title("Handwritten Text OCR")
st.write("Upload an image or use the webcam to recognize handwritten English text.")

option = st.sidebar.radio("Choose input method", ["Upload Image", "Webcam"])

image_file = None

if option == "Upload Image":
    image_file = st.file_uploader(
        "Upload handwritten text image",
        type=["jpg", "jpeg", "png", "bmp"]
    )
else:
    image_file = st.camera_input("Show handwritten text in front of webcam and capture")

if image_file:
    image = Image.open(image_file)

    with st.spinner("Recognizing text..."):
        output_image, processed_image, detected_text = recognize_text(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detected Text Image")
        st.image(output_image, use_container_width=True)

    st.subheader("Preprocessed Image")
    st.image(processed_image, channels="GRAY", use_container_width=True)

    st.subheader("Recognized Text")

    if detected_text:
        st.dataframe(detected_text, use_container_width=True)
        final_text = "\n".join(item["Text"] for item in detected_text)
        st.text_area("Final OCR Output", final_text, height=150)
    else:
        st.warning("No text detected. Try better lighting and clearer handwriting.")
