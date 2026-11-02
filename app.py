import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(page_title="AI Endoscopy Diagnostics", layout="wide")

st.title("Real-Time Explainable AI (XAI) Endoscopy Dashboard")
st.write("Upload an endoscopy frame for instant analysis and visual tissue heatmap.")

uploaded_file = st.file_uploader("Choose an endoscopy image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("Analyzing tissue...")

    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            confidence = result["confidence"]
            heatmap_base64 = result.get("heatmap_base64")

            st.subheader("Diagnostic Result")
            anomalies = ['polyps', 'dyed-lifted-polyps', 'esophagitis', 'ulcerative-colitis']
            
            if prediction in anomalies:
                st.error(f"⚠️ Anomaly Detected: **{prediction.upper()}** ({confidence}%)")
            else:
                st.success(f"✅ Healthy/Normal: **{prediction.upper()}** ({confidence}%)")
            
            # Display side-by-side images
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_container_width=True)
                
            with col2:
                st.markdown("### XAI Diagnostic Heatmap")
                if heatmap_base64:
                    heatmap_bytes = base64.b64decode(heatmap_base64)
                    heatmap_image = Image.open(io.BytesIO(heatmap_bytes))
                    st.image(heatmap_image, use_container_width=True)
                else:
                    st.warning("Heatmap generation failed.")
                    
        else:
            st.error("Error: The API backend rejected the request.")
            
    except Exception as e:
        st.error("Failed to connect. Is your FastAPI server running?")