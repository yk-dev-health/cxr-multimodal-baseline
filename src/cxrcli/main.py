"""
Demonstration script for multimodal CXR + tabular data pipeline.

This script:
- Loads CSV metadata and corresponding CXR images
- Verifies CSV image alignment
- Applies image preprocessing
- Encodes tabular features into numeric tensors
- Batches data using a custom collate function
- Runs a forward pass through a baseline fusion model
- Saves predictions and evaluation metrics
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CXRDataset
from preprocessing import encode_tabular_features
from models import BaselineFusionModel

def collate_fn(batch):
    """
    Custom collate function to prepare batched model inputs.
    """
    images = torch.stack([b["image"] for b in batch])
    tabular = torch.stack([encode_tabular_features(b) for b in batch])
    labels = [b["labels"] for b in batch]

    return images, tabular, labels


def main(csv_path="data/nih_cxr/Data_Entry_2017.csv",
        image_dir="data/nih_cxr/images/images_001",
        batch_size=16,
        output_dir="outputs/"):
    """
    Main pipeline for CXR multimodal baseline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        batch_size=batch_size,
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
    all_predictions = []
    all_labels = []
    for images, tabular, labels in loader:
        with torch.no_grad():
            outputs = model(images, tabular)
            preds = outputs.argmax(dim=1).tolist()
            all_predictions.extend(preds)
            all_labels.extend(labels)

    # Save predictions
    pred_file = output_dir / "predictions.json"
    with open(pred_file, "w") as f:
        json.dump(all_predictions, f, indent=2)

    # Example metrics (can be extended)
    metrics = {
        "num_samples": len(dataset),
        "num_batches": len(loader)
    }
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPredictions saved → {pred_file}")
    print(f"Metrics saved → {metrics_file}")


if __name__ == "__main__":
    main()