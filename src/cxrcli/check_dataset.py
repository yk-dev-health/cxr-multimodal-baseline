from pathlib import Path
from dataset import CXRDataset

dataset = CXRDataset(
    csv_path=Path("data/nih_cxr/Data_Entry_2017.csv"),
    image_dir=Path("data/nih_cxr/images/images_001"),
)

print("Dataset size:", len(dataset))
sample = dataset[0]
print(sample.keys())