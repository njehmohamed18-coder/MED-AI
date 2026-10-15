import streamlit as st
import requests
from PIL import Image

# Configure the page layout
st.set_page_config(page_title="AI Endoscopy Diagnostics", layout="centered")

st.title("Real-Time Diagnostic Vision Dashboard")
st.write("Upload an endoscopy frame for instant analysis by the ResNet18 model.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an endoscopy image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Endoscopy Frame', use_container_width=True)

    st.write("Analyzing tissue...")

    # 2. Send the image to your running FastAPI backend
    try:
        # Prepare the file exactly how FastAPI expects it (multipart/form-data)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        # 3. Process and display the JSON result
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            confidence = result["confidence"]

            st.subheader("Diagnostic Result")
            
            # Color-code the output based on whether it's an anomaly or healthy
            anomalies = ['polyps', 'dyed-lifted-polyps', 'esophagitis', 'ulcerative-colitis']
            
            if prediction in anomalies:
                st.error(f"⚠️ Anomaly Detected: **{prediction.upper()}** ({confidence}%)")
            else:
                st.success(f"✅ Healthy/Normal: **{prediction.upper()}** ({confidence}%)")
        else:
            st.error("Error: The API backend rejected the request.")
            
    except Exception as e:
        st.error("Failed to connect. Is your FastAPI server running?")