import torch, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from torchvision import transforms
from transformers import ViTModel
import torchinfo
torch.cuda.manual_seed(42)

class ViTForCLS(nn.Module):
    def __init__(self, num_classes=100, num_hidden_layers_in_cls_head: int = 1):
        super(ViTForCLS, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', add_pooling_layer=False)
        if num_hidden_layers_in_cls_head == 1:
            self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        elif num_hidden_layers_in_cls_head == 2:
            self.classifier = nn.Sequential(nn.Linear(self.vit.config.hidden_size, self.vit.config.hidden_size),
                                            nn.Linear(self.vit.config.hidden_size, num_classes))
        else:
            raise ValueError("Invalid input!")

    def forward(self, x):
        """x.shape = (b, 3, 224, 224)"""
        outputs = self.vit(pixel_values=x) 
        hidden_state = outputs.last_hidden_state[:, 0, :]  # CLS token output
        logits = self.classifier(hidden_state)
        return logits # (b, 100)

if __name__ == "__main__":
    model = ViTForCLS(100,1).to("cuda")
    a = torch.rand(size=(4, 3, 224, 224)).to("cuda")
    model(a)
    print("DONE")