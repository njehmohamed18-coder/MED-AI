from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np

# New Explainable AI imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Configuration & Global State
CLASSES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
NUM_CLASSES = len(CLASSES)
MODEL_PATH = "kvasir_resnet18_best.pth"

ml_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() 
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading PyTorch model into memory...")
    ml_models["kvasir_resnet"] = load_model()
    print(f"Model loaded successfully on {device}.")
    yield 
    print("Shutting down and clearing memory...")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="Endoscopy Diagnostic API (Explainable AI)", lifespan=lifespan)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        model = ml_models["kvasir_resnet"]
        
        # 1. Standard Forward Pass (Inference mode removed for Grad-CAM)
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = CLASSES[predicted_idx.item()]
        
        # 2. Grad-CAM XAI Generation
        # Target the final convolutional layer of ResNet18
        target_layers = [model.layer4[-1]]
        
        # Construct the CAM object
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_idx.item())]
        
        # Generate the raw spatial heatmap
        grayscale_cam = cam(input_tensor=input_batch, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # 3. Create the visual overlay
        # Resize original image to match tensor size (224x224), convert to numpy, scale to [0, 1]
        img_resized = image.resize((224, 224))
        rgb_img = np.float32(img_resized) / 255
        
        # Merge heatmap with original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Convert numpy array back to image and encode as base64 string for transmission
        pil_vis = Image.fromarray(visualization)
        buffered = io.BytesIO()
        pil_vis.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": round(confidence.item() * 100, 2),
            "status": "success",
            "heatmap_base64": img_base64  # Pass the encoded image to the frontend
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")