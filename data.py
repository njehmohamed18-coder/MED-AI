import os
import urllib.request
import zipfile
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 1. Configuration variables
DATA_URL = "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip"
DATA_DIR = "./kvasir_data"
ZIP_PATH = os.path.join(DATA_DIR, "kvasir-dataset-v2.zip")
BATCH_SIZE = 32

def download_and_extract():
    """Downloads the Kvasir dataset if it doesn't exist and extracts it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(ZIP_PATH):
        print("Downloading Kvasir v2 dataset (~2.3 GB). This may take a while...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Download complete.")
    
    extracted_dir = os.path.join(DATA_DIR, "kvasir-dataset-v2")
    if not os.path.exists(extracted_dir):
        print("Extracting files...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")
        
    return extracted_dir

def get_data_loaders(data_path):
    """Preprocesses the images and splits them into PyTorch DataLoaders."""
    
    # 2. Define the Preprocessing Pipeline (Transforms)
    # ResNet and MobileNet expect 224x224 images and specific normalization
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load all images from the root folder
    print("Loading images into PyTorch...")
    full_dataset = datasets.ImageFolder(root=data_path, transform=data_transforms)
    
    # 3. Calculate split sizes (80% Train, 10% Val, 10% Test)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Split the dataset randomly
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Ensures reproducibility 
    )
    
    # 4. Wrap in DataLoaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Data ready: {len(train_dataset)} Train | {len(val_dataset)} Val | {len(test_dataset)} Test images.")
    print(f"Classes found: {full_dataset.classes}")
    
    return train_loader, val_loader, test_loader, full_dataset.classes

if __name__ == "__main__":
    # Execute the pipeline
    extracted_path = download_and_extract()
    train_loader, val_loader, test_loader, classes = get_data_loaders(extracted_path)
    
    # The data is now ready to be fed into a neural network.