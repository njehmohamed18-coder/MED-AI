import torch
import torch.nn as nn
from data import download_and_extract, get_data_loaders
import torch.optim as optim
from torchvision import models
import time
import copy

# Set device to GPU if available (or Apple Silicon MPS), else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on device: {device}")

def initialize_model(num_classes):
    """Loads pre-trained ResNet18 and modifies the final layer."""
    
    # Load ResNet18 with standard ImageNet weights (PyTorch 2.x standard)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Optional: Freeze the core convolutional layers so they don't update
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # The original ResNet18 outputs 1000 classes. We need to replace the final 
    # fully connected (fc) layer to output our specific number of classes.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """Runs the training and validation loop, saving the best model."""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase, loader in [('train', train_loader), ('val', val_loader)]:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data batches
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # Clear previous gradients

                # Forward pass
                # Track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.float() / len(loader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy we've seen
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the physical file so you can load it in your API later
                torch.save(model.state_dict(), 'kvasir_resnet18_best.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:4f}')

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # 1. Run the data pipeline to get the loaders and classes
    extracted_path = download_and_extract()
    train_loader, val_loader, test_loader, classes = get_data_loaders(extracted_path)
    
    # 2. Initialize Model (Kvasir has 8 classes)
    num_classes = len(classes) 
    model = initialize_model(num_classes)
    
    # 3. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 4. Train the Model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)


