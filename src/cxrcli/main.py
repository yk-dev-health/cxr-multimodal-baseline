"""
Demonstration script for multimodal CXR + tabular data pipeline.

This script:
- Loads CSV metadata and corresponding CXR images
- Verifies CSV image alignment
- Applies image preprocessing
- Encodes tabular features into numeric tensors
- Batches data using a custom collate function
- Runs a forward pass through a baseline fusion model
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CXRDataset
from preprocessing import encode_tabular_features
from models import BaselineFusionModel

def collate_fn(batch):
    """
    Custom collate function to prepare batched model inputs.

    Converts a list of dataset samples into:
    - images: Tensor [B, C, H, W]
    - tabular features: Tensor [B, F]
    - labels: list[str]

    This function defines the exact interface between
    the dataset and the downstream model.
    """
    images = torch.stack([b["image"] for b in batch])
    tabular = torch.stack([encode_tabular_features(b) for b in batch])
    labels = [b["labels"] for b in batch]

    return images, tabular, labels


def main():
    # Paths to CSV metadata and image directory
    csv_path = "data/nih_cxr/Data_Entry_2017.csv"
    image_dir = "data/nih_cxr/images/images_001"

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Initialise dataset
    dataset = CXRDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=transform,
    )

    print("Dataset size:", len(dataset))

    # Quick check: first 5 entries to confirm CSV-image correspondence
    print("\nSample patient data:")
    for i in range(5):
        row = dataset.df.iloc[i]
        print(
            f"Patient ID: {row['Patient ID']}, "
            f"Image: {row['Image Index']}, "
            f"Age: {row['Patient Age']}, "
            f"Gender: {row['Patient Gender']}"
        )

    # DataLoader with custom collate function
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Initialise baseline fusion model
    example_tabular = encode_tabular_features(dataset[0])
    tab_dim = example_tabular.shape[0]

    model = BaselineFusionModel(
        tab_dim=tab_dim,
        num_classes=2,
    )
    model.eval()  # inference mode for demonstration

    # Run a single forward pass for validation
    for images, tabular, labels in loader:
        print("\nFirst batch inspection:")
        print("Batch image shape:", images.shape)
        print("Batch tabular shape:", tabular.shape)
        print("Batch labels:", labels)

        with torch.no_grad():
            outputs = model(images, tabular)

        print("Model output shape:", outputs.shape)
        break  # one batch is sufficient for validation


if __name__ == "__main__":
    main()