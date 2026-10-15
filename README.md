# Real-Time Endoscopy Diagnostic Vision Dashboard

An end-to-end machine learning application that provides real-time, computer-aided diagnostics for gastrointestinal endoscopy. 

This repository serves as the functional software implementation of academic research into *AI vs. Cancer: Revolutionary Gastric Cancer Detection through Real-Time Endoscopy*. It transitions theoretical computer vision models into a deployable, user-facing web prototype designed to assist in identifying anomalies like polyps and esophagitis instantly.

## Architecture

The system is decoupled into a high-performance inference API and a lightweight frontend:
*   **Backend:** An asynchronous **FastAPI** server that manages model state in memory using lifespan context managers, ensuring rapid inference without reloading weights for each request.
*   **AI Engine:** A **PyTorch** transfer-learning pipeline utilizing a fine-tuned ResNet18 model, optimized to run on CPU, CUDA, or Apple Silicon (MPS).
*   **Frontend:** A **Streamlit** interactive UI allowing users to upload medical imagery and receive color-coded diagnostic confidence scores.

## Tech Stack
*   **Language:** Python 3
*   **Machine Learning:** PyTorch, Torchvision, Scikit-learn
*   **API & Server:** FastAPI, Uvicorn, Pydantic
*   **Frontend UI:** Streamlit
*   **Data Processing:** Pillow (PIL), Requests

## Dataset
This model is trained on the [Kvasir v2 Dataset](https://datasets.simula.no/kvasir/), a multi-class dataset containing 8,000 anatomical landmarks and pathological findings from the GI tract. The model classifies images into 8 distinct categories, including healthy tissue and anomalies like dyed-lifted-polyps and ulcerative-colitis.

## Local Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/Real-Time-Endoscopy-AI.git](https://github.com/YourUsername/Real-Time-Endoscopy-AI.git)
cd Real-Time-Endoscopy-AI

2. Install Dependencies
Bash
pip install torch torchvision fastapi uvicorn python-multipart pillow streamlit requests
3. Setup and Training
Download the Kvasir dataset and initiate the PyTorch training loop to generate the model weights (.pth file).

Bash
python3 training.py
4. Launch the Application
You must run the backend API and the frontend dashboard simultaneously in two separate terminal windows.

Terminal 1: Start the Inference API

Bash
python3 -m uvicorn main:app --reload
The API will run on http://127.0.0.1:8000

Terminal 2: Start the Dashboard

Bash
python3 -m streamlit run app.py
The UI will automatically open in your default web browser.

Future Roadmap
Containerize the application using Docker for easier cloud deployment.

Implement a YOLOv8 object detection model to draw bounding boxes around specific polyps rather than just classifying the entire frame.

Optimize the inference engine with ONNX or TensorRT for lower latency on edge devices.
