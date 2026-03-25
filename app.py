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
def detect_blur(gray_image):
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()
    return variance, variance < 100

# -------------------------
# 2. OVEREXPOSURE
# -------------------------
def detect_overexposure(gray_image):
    hist = cv2.calcHist([gray_image], [0], None, [256], [0,256])
    bright_pixels = hist[250:].sum()
    total_pixels = gray_image.size
    ratio = bright_pixels / total_pixels
    return ratio, ratio > 0.2

# -------------------------
# 3. UNDEREXPOSURE
# -------------------------
def detect_underexposure(gray_image):
    brightness = np.mean(gray_image)
    return brightness, brightness < 50

# -------------------------
# 4. NOISE DETECTION
# -------------------------
def detect_noise(gray_image):
    noise = np.std(gray_image)
    return noise, noise > 50

# -------------------------
# UI
# -------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        img = np.array(image)

        # Ensure image is in RGB format for processing (PIL opens as RGB)
        # However, some images might be grayscale or RGBA
        if img.ndim == 2:  # Grayscale
            gray = img
        elif img.shape[2] == 4:  # RGBA
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        else:  # RGB
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        blur_score, is_blurry = detect_blur(gray)
        glare_score, is_glare = detect_overexposure(gray)
        dark_score, is_dark = detect_underexposure(gray)
        noise_score, is_noisy = detect_noise(gray)

        st.subheader("Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Blur Score", f"{blur_score:.2f}", delta="Blury" if is_blurry else "Sharp", delta_color="inverse")
            st.metric("Overexposure", f"{glare_score:.2f}", delta="Too Bright" if is_glare else "OK", delta_color="inverse")
        
        with col2:
            st.metric("Brightness", f"{dark_score:.2f}", delta="Too Dark" if is_dark else "OK", delta_color="inverse")
            st.metric("Noise Score", f"{noise_score:.2f}", delta="Noisy" if is_noisy else "Clean", delta_color="inverse")

        if any([is_blurry, is_glare, is_dark, is_noisy]):
            st.warning("⚠️ High issues detected in the image.")
        else:
            st.success("✅ Image quality looks good!")

    except Exception as e:
        st.error(f"Error processing image: {e}")