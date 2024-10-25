import numpy as np
import pandas as pd
import torch
from dataset import CIFARDataset
from transformers import ViTForImageClassification
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_test_data():
    csv_file = "/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/gnr-650-assignment-1/cifar100_test_mod.csv"  
    df = pd.read_csv(csv_file)
    reshuffled_images = df.to_numpy()  
    reshuffled_images = reshuffled_images.reshape(-1, 32, 32, 3).astype(np.uint8) 
    labels = np.zeros(shape=(reshuffled_images.shape[0],)) # keep all labels as zeros as these are of no use

    np.savez("/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/outputs/test_image_labels_for_submission.npz", images=reshuffled_images, labels=labels)

def check_acc(pred):
    true_ds = CIFARDataset(npz_file="/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/outputs/test_image_labels.npz")
    true = np.array([int(i[1]) for i in true_ds])
    pred = np.array(pred)
    acc = (true == pred).mean()
    return acc

def main(ckpt_name: str):
    # save_test_data()
    test_ds = CIFARDataset(npz_file="/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/outputs/test_image_labels_for_submission.npz")
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=100).to("cuda")
    checkpoint_path = f"/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/outputs/ckpt/vitB16-finetune-CIFAR100-latest/{ckpt_name}.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state"])
    print(f"[LOAD] ep: {checkpoint['epoch']}, checkpoint loaded from: {checkpoint_path}")
    
    sub_dict = {}
    model.eval()
    for i, data in enumerate(test_ds):
        x = data[0].unsqueeze(0)
        with torch.no_grad():
            logits = model(x.to("cuda")).logits
        pred_label = logits.argmax(dim=-1)
        sub_dict[i] = pred_label.item()
        print(i)
    
    print(ckpt_name, ": ", check_acc(list(sub_dict.values())))
        
    df = pd.DataFrame(list(sub_dict.items()), columns=['ID', 'TARGET'])
    df.to_csv(f"/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/outputs/submission_{'_'.join(ckpt_name.split('/'))}.csv", index=False)

if __name__ == "__main__":
    ckpt_names = ["denoise_all_alt/ckpt_ep_1"]
    main(ckpt_names[0])
