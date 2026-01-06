from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CXRDataset(Dataset):
    """
    Map-style PyTorch Dataset for Chest X-ray images with tabular metadata.

    This class implements:
    - __len__(): required by DataLoader to know dataset size
    - __getitem__(): required by DataLoader to fetch a single sample by index

    Design choices:
    - CSV is the single source of truth for labels and metadata
    - Images are loaded later in __getitem__ (not at initialisation)
    - Suitable for batching, shuffling, multiprocessing, and GPU pipelines
    """

    def __init__(
        self,
        csv_path: Path,
        image_dir: Path,
        transform: Optional[Any] = None,
    ) -> None:
        """
        This method is called once when the Dataset object is created.

        Args:
            csv_path: Path to the CSV file containing metadata and labels
            image_dir: Directory containing image files
            transform: Optional torchvision-style transform applied to images
        """
        # Store paths and configuration as object state
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Load tabular metadata into memory once
        self.df = pd.read_csv(self.csv_path)

        # Resolve image file paths from CSV entries
        # The dataset size is defined by valid image files
        self.df["image_path"] = self.df["Image Index"].apply(
            lambda x: self.image_dir / x
        )

        # Filter out rows where the image file does not exist
        self.df = self.df[self.df["image_path"].apply(lambda p: p.exists())]

        # Reset index so __getitem__(idx) works correctly
        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetch a single sample by index.

        Args:
            idx: Index provided by DataLoader

        Returns:
            A dictionary representing one training sample
        """
        
        row = self.df.iloc[idx] # Retrieve the corresponding row from the metadata table

        # Load image late at access time (not at initialisation)
        img = Image.open(row["image_path"]).convert("RGB")

        # Apply optional preprocessing / augmentation
        if self.transform:
            img = self.transform(img)

        # Return a structured sample
        # DataLoader will later collate multiple samples into a batch
        sample = {
            "image": img,                       # image tensor (after transform)
            "labels": row["Finding Labels"],    # raw label string (multi-label)
            "patient_id": row["Patient ID"],    # patient-level identifier
            "view_position": row.get("View Position", None),
        }

        return sample