import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage, Compose, Resize, Normalize, ToTensor
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.metrics import accuracy_score

# Dataset class for loading CSV data
class CIFAR100CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image = self.data_frame.iloc[idx, :-1].values.astype(np.uint8).reshape(32, 32, 3)
        label = int(self.data_frame.iloc[idx, -1])
        if self.transform:
            image = self.transform(image)
        return image, label
class CIFAR100CSVTest(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Assume each row contains only the image data (3072 columns)
        image_data = self.data_frame.iloc[idx].values  # Get all pixel data
        image = image_data.astype(np.uint8).reshape(32, 32, 3)  # Reshape into an image
        if self.transform:
            image = self.transform(image)
        return image, idx  # Return the image and its index
# Setup transformations
def get_transform():
    return Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to train the model
def train_model(model, train_loader, device, num_epochs=10):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pkl'
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(model.state_dict(), f)

# Function to evaluate model
def evaluate_model(model, predict_loader, device, checkpoint_dir, actual_labels_path):
    actual_labels_df = pd.read_csv(actual_labels_path)
    num_checkpoints = 10
    for epoch in range(9, num_checkpoints + 1):
        checkpoint_path = f'{checkpoint_dir}checkpoint_epoch_{epoch}.pkl'
        with open(checkpoint_path, 'rb') as f:
            model.load_state_dict(pickle.load(f))
        
        model.eval()
        predictions = []
        for images, ids in predict_loader:
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(zip(ids.numpy(), preds))

        predictions_df = pd.DataFrame(predictions, columns=['ID', 'Target'])
        # df.to_csv(f'/raid/biplab/sarthak/GNR_650/prediction_dir/predictions_epoch_{epoch+1}.csv', index=False)
        predictions_df = predictions_df.sort_values('ID').reset_index(drop=True)
        accuracy = accuracy_score(actual_labels_df['Label'], predictions_df['Target'])
        print(f'Accuracy for checkpoint {epoch}: {accuracy * 100:.2f}%')

# Main function to coordinate the workflow
def main():
    # Initialize the dataset
    train_transform = get_transform()
    train_dataset = CIFAR100CSVDataset('/raid/biplab/sarthak/GNR_650/cifar100_train.csv', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=4)

    # Load model and feature extractor
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=100)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    # Freeze all layers except the last block and the classifier head
    for name, param in model.named_parameters():
        if not name.startswith('vit.encoder.layer.11','vit.encoder.layer.10','vit.encoder.layer.9') and 'classifier' not in name:
            param.requires_grad = False
    model.to(device)
    print(model)
    
    # # Train the model
    # train_model(model, train_loader, device)
    
    # # Predict and evaluate
    # predict_dataset = CIFAR100CSVTest('/raid/biplab/sarthak/GNR_650/cifar100_test_mod.csv', transform=train_transform)
    # predict_loader = DataLoader(predict_dataset, batch_size=128, shuffle=False, num_workers=4)
    # evaluate_model(model, predict_loader, device, '/raid/biplab/sarthak/GNR_650/', '/raid/biplab/sarthak/GNR_650/cifar100_test_labels.csv')

if __name__ == '__main__':
    main()
