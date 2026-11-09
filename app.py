import streamlit as st
import requests
from PIL import Image
import io
import base64

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="MED-AI Endoscopy", page_icon="🔬", layout="wide")

# 2. Professional Sidebar for Context
with st.sidebar:
    st.title("⚙️ MED-AI System")
    st.markdown("---")
    st.markdown("**Architecture:**")
    st.markdown("- **Backend:** FastAPI (Asynchronous)")
    st.markdown("- **AI Engine:** ResNet18 (PyTorch)")
    st.markdown("- **XAI Method:** Grad-CAM Heatmaps")
    st.markdown("- **Dataset:** Kvasir v2")
    st.markdown("---")
    st.caption("Developed for Macromedia Hackathon 2026")

# 3. Main Dashboard Header
st.title("🔬 Real-Time Explainable AI (XAI) Endoscopy")
st.markdown("Upload a gastrointestinal endoscopy frame for instant analysis and visual tissue mapping.")
st.divider()

uploaded_file = st.file_uploader("Choose an endoscopy image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 4. Dynamic Loading Spinner
    with st.spinner("Analyzing tissue with ResNet18 model..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post("http://127.0.0.1:8000/predict", files=files)

            if response.status_code == 200:
                result = response.json()
                # Clean up the text format (e.g., "dyed-lifted-polyps" -> "Dyed Lifted Polyps")
                prediction_clean = result["prediction"].replace('-', ' ').title()
                confidence = result["confidence"]
                heatmap_base64 = result.get("heatmap_base64")

                # 5. Executive Metrics Display
                st.subheader("Diagnostic Report")
                anomalies = ['polyps', 'dyed-lifted-polyps', 'esophagitis', 'ulcerative-colitis']
                
                col_alert, col_metric = st.columns([3, 1])
                
                with col_alert:
                    if result["prediction"] in anomalies:
                        st.error(f"⚠️ **Anomaly Detected:** {prediction_clean}")
                    else:
                        st.success(f"✅ **Healthy Tissue:** {prediction_clean}")
                        
                with col_metric:
                    st.metric(label="AI Confidence", value=f"{confidence}%")

                st.markdown("---")
                
                # 6. Side-by-Side Visuals
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Original Endoscopy Frame")
                    original_image = Image.open(uploaded_file)
                    st.image(original_image, use_container_width=True)
                    
                with col2:
                    st.markdown("#### XAI Diagnostic Heatmap")
                    if heatmap_base64:
                        heatmap_bytes = base64.b64decode(heatmap_base64)
                        heatmap_image = Image.open(io.BytesIO(heatmap_bytes))
                        st.image(heatmap_image, use_container_width=True)
                    else:
                        st.warning("Heatmap generation failed.")
                        
            else:
                st.error("Error: The API backend rejected the request.")
                
        except Exception as e:
            st.error("Failed to connect. Ensure your FastAPI server is running on port 8000.")