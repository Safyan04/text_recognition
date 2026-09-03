# Multi-Image Text Recognition App

A Streamlit-powered web application designed to extract text from multiple images simultaneously using OpenCV and EasyOCR.

## Features
- **Batch Image Processing:** Upload and extract text from multiple images in one go.
- **Visual Bounding Boxes:** Automatically highlights recognized text with clear visual bounding boxes on output images.
- **Interactive Web UI:** Simple and clean browser interface built with Streamlit.

## Tech Stack
- Python
- Streamlit
- EasyOCR
- OpenCV
- NumPy

## Setup & Run
```bash
git clone
pip install opencv-python easyocr streamlit numpy pillow torch
streamlit run app.py
