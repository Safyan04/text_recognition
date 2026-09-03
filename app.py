import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Multi-Image Text Recognition", page_icon="📝", layout="centered"
)

st.title("📝 Multi-Image Text Recognition & Extraction")
st.write(
    "Upload multiple images containing blurry or small text, and the app will"
    " extract text from all of them!"
)


# Caching reader so it loads only once
@st.cache_resource
def load_reader():
  return easyocr.Reader(["en"], gpu=False)


with st.spinner("Loading OCR Model..."):
  reader = load_reader()

# File Uploader with multiple files support
uploaded_files = st.file_uploader(
    "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
  if st.button("Extract Text from All Images"):
    with st.spinner("Processing images and recognizing text..."):
      for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"📁 Processing: {uploaded_file.name}")

        # Read image with OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display original uploaded image
        st.image(
            cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB),
            caption=f"Original: {uploaded_file.name}",
            channels="RGB",
            use_container_width=True,
        )

        # 2x Resize for blurry/small text handling
        height, width = img_cv.shape[:2]
        img_resized = cv2.resize(
            img_cv, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC
        )

        # OCR Run
        results = reader.readtext(img_resized)

        extracted_texts = []
        for bbox, text, prob in results:
          if prob > 0.4:
            extracted_texts.append(text)
            # Map coordinates back to original size
            top_left = (int(bbox[0][0] / 2), int(bbox[0][1] / 2))
            bottom_right = (int(bbox[2][0] / 2), int(bbox[2][1] / 2))
            cv2.rectangle(img_cv, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                img_cv,
                text,
                top_left,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        # Display Processed Image with Bounding Boxes
        st.image(
            cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB),
            caption=f"Detected Output: {uploaded_file.name}",
            channels="RGB",
            use_container_width=True,
        )

        # Display Extracted Text
        final_text = " ".join(extracted_texts)
        st.success(
            f"**Extracted Text:**\n\n{final_text if final_text else 'No text detected.'}"
        )