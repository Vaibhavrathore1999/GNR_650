import torchvision
import numpy as np
import pandas as pd

# Step 1: Load the original CIFAR-100 dataset and get images and labels
original_train_dataset = torchvision.datasets.CIFAR100(root="data", train=True, download=True)
original_images = original_train_dataset.data
original_labels = np.array(original_train_dataset.targets)

# Step 2: Load your reshuffled images from the CSV
csv_file = "gnr-650-assignment-1/cifar100_train.csv"
df = pd.read_csv(csv_file)
reshuffled_images = df.iloc[:, :-1].to_numpy().reshape(-1, 32, 32, 3)

# Step 3: Retrieve the true labels for each reshuffled image by comparing with original images
true_labels_for_reshuffled = []
for i, reshuffled_image in enumerate(reshuffled_images):
    true_label = None
    for idx, original_image in enumerate(original_images):
        if np.array_equal(reshuffled_image, original_image):  # Compare reshuffled image with original image
            true_label = original_labels[idx]  # Get the true label
            print(i, idx)
            break
    true_labels_for_reshuffled.append(true_label)

# Step 4: Save the true labels for reshuffled images
true_labels_for_reshuffled = np.array(true_labels_for_reshuffled)
np.save("Assignment1/outputs/true_labels_for_reshuffled_noisy_data.npy", true_labels_for_reshuffled)

print("True labels for reshuffled images have been retrieved and saved.")