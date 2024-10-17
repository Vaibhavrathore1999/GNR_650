import argparse
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from utils import datasets_to_c, get_cluster_acc
import torchvision
import numpy as np
import os

def _parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to evaluate TURTLE", required=True)
    parser.add_argument('--phis', type=str, default=["clipvitL14", "dinov2"], nargs='+', help="Representation spaces to evaluate TURTLE", 
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2'])
    parser.add_argument('--root_dir', type=str, default='data', help='Root dir to store everything')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the checkpoint to evaluate")
    return parser.parse_args(args)

class Args:
    def __init__(self, phis: list, space: int):
        self.root_dir = "/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/data"
        self.dataset = "cifar100"
        self.phis = phis
        self.device = "cuda"
        if space == 2:
            model = "_".join(self.phis)
        else:
            model = phis[0]
        self.ckpt_path = f"/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/turtle_tasks/{space}space/{model}/cifar100.pt"

def main(args):
    
    # Load pre-computed representations 
    Zs_val = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_train.npy").astype(np.float32) for phi in args.phis]
    y_gt_val = np.load(f"Assignment1/outputs/noised_labels_for_reshuffled_noisy_data.npy")

    print(f'Load dataset {args.dataset}')
    print(f'Representations of {args.phis}: ' + ' '.join(str(Z_val.shape) for Z_val in Zs_val))

    C = datasets_to_c[args.dataset]
    feature_dims = [Z_val.shape[1] for Z_val in Zs_val]
    
    # Task encoder
    task_encoder = [nn.Linear(d, C).to(args.device) for d in feature_dims] 
    ckpt = torch.load(args.ckpt_path)
    for task_phi, ckpt_phi in zip(task_encoder, ckpt.values()):
        task_phi.load_state_dict(ckpt_phi)

    # Denoise
    label_per_space = [F.softmax(task_phi(torch.from_numpy(Z_val).to(args.device)), dim=1) for task_phi, Z_val in zip(task_encoder, Zs_val)] # shape of (N, K, C)
    labels = torch.mean(torch.stack(label_per_space), dim=0) # shape of (N, C)

    y_pred = labels.argmax(dim=-1).detach().cpu().numpy() # pseudolabels
    
    x_dict = {k: [] for k in range(100)}
    y_dict = {k: [] for k in range(100)}
    for idx, value in enumerate(y_pred):
        x_dict[value].append(int(idx))
        y_dict[value].append(int(y_gt_val[idx]))

    histogram_dict = {}
    for key, values in y_dict.items():
        histogram_dict[key] = dict(Counter(values))
        major = max(histogram_dict[key], key=histogram_dict[key].get)
        histogram_dict[key].update({"majority_label": major})
    
    xy_dataset = {i: -1 for i in range(len(y_pred))}
    for key, val in histogram_dict.items():
        for i in x_dict[key]:
            xy_dataset[i] = val["majority_label"]
    
    return xy_dataset

if __name__ == "__main__":

    csv_file = "/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/gnr-650-assignment-1/cifar100_train.csv"
    df = pd.read_csv(csv_file)
    reshuffled_images = df.iloc[:, :-1].to_numpy().reshape(-1, 32, 32, 3)

    args = Args(["clipvitL14", "dinov2"], space=2)
    image_to_denoised_label = main(args)
    denoised_labels = [v for k, v in image_to_denoised_label.items()]
    noised_labels = np.load("/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/outputs/noised_labels_for_reshuffled_noisy_data.npy")
    
    np.savez("Assignment1/outputs/denoised_image_labels_for_reshuffled_noisy_data.npz", images=reshuffled_images, labels=denoised_labels)
    np.savez("Assignment1/outputs/noised_image_labels_for_reshuffled_noisy_data.npz", images=reshuffled_images, labels=noised_labels)
    np.save("Assignment1/outputs/denoised_labels_for_reshuffled_noisy_data.npy", denoised_labels)
    print("DONE")

