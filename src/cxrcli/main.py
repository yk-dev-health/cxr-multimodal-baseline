"""
Demonstration script for CXR + tabular dataset loading.

This script:
- Loads CSV metadata and corresponding CXR images
- Confirms dataset size and sample structure
- Prepares tabular features for future model input

For demonstration purposes, the tabular feature here is simply
the patient's age as a numeric tensor.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CXRDataset

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    tabs = torch.tensor([[float(b["age"]) if b["age"] else 0] for b in batch], dtype=torch.float32)
    labels = [b["labels"] for b in batch]
    return images, tabs, labels


def main():
    # Paths to CSV metadata and image directory
    csv_path = "data/nih_cxr/Data_Entry_2017.csv"
    image_dir = "data/nih_cxr/images/images_001"

    # Image preprocessing: resize, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load dataset
    dataset = CXRDataset(csv_path, image_dir, transform=transform)
    print("Dataset size:", len(dataset))

    # Quick check: first 5 entries to confirm CSV-image correspondence
    print("Sample patient data:")
    for i in range(5):
        row = dataset.df.iloc[i]
        print(
            f"Patient ID: {row['Patient ID']}, "
            f"Image: {row['Image Index']}, "
            f"Age: {row['Patient Age']}, "
            f"Gender: {row['Patient Gender']}"
        )

    # Create DataLoader with custom collate function
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Check first batch
    for imgs, tabs, labels in loader:
        print("\nFirst batch shapes and labels:")
        print("Batch image shape:", imgs.shape)
        print("Batch tabular shape:", tabs.shape)
        print("Batch labels:", labels)
        break  # only check one batch for demonstration

if __name__ == "__main__":
    main()