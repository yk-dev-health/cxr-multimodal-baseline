# CXR Multimodal Baseline

This project demonstrates a baseline multimodal pipeline for healthcare data, combining chest X-ray images with structured clinical tabular data.

The focus is on data handling, preprocessing, and reproducible pipeline design, rather than achieving state-of-the-art model performance.
It is intended for engineers and data practitioners working with
real-world healthcare data, where multimodal inputs and data quality constraints are common.

## Key Features (Current Status)

* **CSV + CXR image loading** – correctly links patient metadata to corresponding images
* **Tabular feature extraction** – numeric features (e.g., Age) prepared as tensors
* **Batching** – custom `collate_fn` allows images and tabular data to be batched together for downstream models
* **Data inspection** – first batch and sample checks confirm dataset alignment
* **Reproducible pipeline structure** – preprocessing separated from main script, ready for model input

## Example Workflow

1. Load CSV metadata and images
2. Inspect sample data
3. Convert tabular features into numeric tensors
4. Batch images + tabular features for downstream models

```python
for images, tabular, labels in loader:
    print(images.shape)   # [batch_size, channels, height, width]
    print(tabular.shape)  # [batch_size, num_features]
    print(labels)         # list of label strings
    break
```

## Project Positioning

This baseline is designed to demonstrate:

* Clear, reproducible preprocessing and pipeline
* Handling of real-world clinical datasets
* Integration of multimodal data into ML-ready format

It is a foundation for future experimentation with fusion models and ML training, not a research-grade model.
## Dataset

The project uses publicly available chest X-ray and clinical datasets
(for demonstration and educational purposes only).

No private or identifiable patient data is included in this repository.

## Project Structure

```
cxr-multimodal-baseline/
├─ data/            # CSV metadata + CXR images
├─ src/             # Core pipeline and model code
├─ config/          # Configuration files
├─ outputs/         # Model outputs and logs
```

## Disclaimer

This repository is for educational and demonstration purposes only.
It is not intended for clinical use or medical decision-making.
