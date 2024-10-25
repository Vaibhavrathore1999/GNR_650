import torch, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from torchvision import transforms

class CIFARDataset(Dataset):
    def __init__(self, npz_file: str):
        super(CIFARDataset, self).__init__()
        data = np.load(npz_file)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        label = self.labels[idx]
        image = self.transform(image)
        return image, label
    
    def prepare_dataloader(self, batch_size: int, frac: float):
        total_size = len(self)
        subset_size = int(total_size * frac)
        indices = random.sample(range(total_size), subset_size)
        subset = Subset(self, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
        return dataloader