import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="IQA Analyzer", layout="centered")

st.title("Image Quality Assessment (IQA)")
st.write("Detects Blur, Overexposure, Underexposure, and Noise")

# -------------------------
# 1. BLUR DETECTION
# -------------------------
def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance, variance < 100

# -------------------------
# 2. OVEREXPOSURE
# -------------------------
def detect_overexposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    bright_pixels = hist[250:].sum()
    total_pixels = gray.size
    ratio = bright_pixels / total_pixels
    return ratio, ratio > 0.2

# -------------------------
# 3. UNDEREXPOSURE
# -------------------------
def detect_underexposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness, brightness < 50

# -------------------------
# 4. NOISE DETECTION
# -------------------------
def detect_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = np.std(gray)
    return noise, noise > 50

# -------------------------
# UI
# -------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    blur_score, is_blurry = detect_blur(img)
    glare_score, is_glare = detect_overexposure(img)
    dark_score, is_dark = detect_underexposure(img)
    noise_score, is_noisy = detect_noise(img)

    st.subheader("Results")

    st.write(f"**Blur Score:** {blur_score:.2f} → {'❌ Blurry' if is_blurry else '✅ Sharp'}")
    st.write(f"**Overexposure:** {glare_score:.2f} → {'❌ Too Bright' if is_glare else '✅ OK'}")
    st.write(f"**Brightness:** {dark_score:.2f} → {'❌ Too Dark' if is_dark else '✅ OK'}")
    st.write(f"**Noise Score:** {noise_score:.2f} → {'❌ Noisy' if is_noisy else '✅ Clean'}")