import wandb, torch, tqdm, sys, os, json, math, gc
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union, Any
from dataset import CIFARDataset
from models import ViTForCIFAR100
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision import models

train_ds = CIFARDataset(npz_file="/home/biplab/siddhant_cminds/gnr/IITB-GNR650-ADLCV/Assignment1/data/train_cifar100_image_label.npz")
val_ds = CIFARDataset(npz_file="/home/biplab/siddhant_cminds/gnr/IITB-GNR650-ADLCV/Assignment1/data/test_cifar100_image_label.npz")

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 100)  # Modify final layer to match CIFAR-100's 100 classes
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


num_epochs = 5
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

def plot_metrics(train_values, val_values, ylabel, title, filename):
    epochs = range(1, num_epochs+1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_values, 'b', label='Train')
    plt.plot(epochs, val_values, 'r', label='Validation')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.show()

# Plot and save the Loss and Accuracy curves
plot_metrics(train_losses, val_losses, 'Loss', 'Loss per Epoch', 'loss_plot.png')
plot_metrics(train_accuracies, val_accuracies, 'Accuracy', 'Accuracy per Epoch', 'accuracy_plot.png')

torch.save(model.state_dict(), "resnet18_cifar100_finetuned.pth")