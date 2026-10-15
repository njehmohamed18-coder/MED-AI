from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# 1. Configuration & Global State
# Use the exact same classes from your training script
CLASSES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
NUM_CLASSES = len(CLASSES)
MODEL_PATH = "kvasir_resnet18_best.pth"

# Dictionary to hold our loaded model in memory across requests
ml_models = {}

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_model():
    """Initializes ResNet18 and loads your custom trained weights."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Load the state dictionary you saved during training
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # CRITICAL: Set to evaluation mode (disables dropout, fixes batchnorm)
    return model

# 2. The Lifespan Context Manager (Best Practice)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase: Load the model into memory once
    print("Loading PyTorch model into memory...")
    ml_models["kvasir_resnet"] = load_model()
    print(f"Model loaded successfully on {device}.")
    
    yield # The API is now actively receiving requests
    
    # Shutdown phase: Clean up memory (crucial if running on GPU)
    print("Shutting down and clearing memory...")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialize FastAPI with the lifespan
app = FastAPI(title="Endoscopy Diagnostic API", lifespan=lifespan)

# 3. Define Preprocessing Pipeline
# This MUST match the validation/test transforms from your training script
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. The Inference Endpoint
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        # Read the uploaded image file bytes directly into memory
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess the image and add a batch dimension: [Channels, Height, Width] -> [1, Channels, Height, Width]
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        # Get the model from memory
        model = ml_models["kvasir_resnet"]
        
        # Perform Inference
        # torch.inference_mode() is faster and uses less memory than torch.no_grad()
        with torch.inference_mode():
            output = model(input_batch)
            
            # Convert raw logits to probabilities using Softmax
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get the top prediction
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_class = CLASSES[predicted_idx.item()]
        
        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": round(confidence.item() * 100, 2),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")