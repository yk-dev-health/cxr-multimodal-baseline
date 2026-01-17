# src/cxrcli/preprocess.py

import torch
from typing import Dict, Any


def encode_tabular_features(sample: Dict[str, Any]) -> torch.Tensor:
    """
    Convert raw tabular fields into a numeric feature vector.
    This function defines the contract between tabular data and the downstream model.

    Current features:
    - Age (normalized to [0, 1])
    - Gender (M=1.0, F=0.0)
    - View position (PA=1.0, others=0.0)

    Args:
        sample: A single dataset sample (dictionary)

    Returns:
        A 1D float tensor representing tabular features
    """
    # Age normalisation (simple scaling for demonstration)
    age = float(sample["age"]) / 100.0 if sample["age"] is not None else 0.0

    # Binary encoding
    gender = 1.0 if sample["gender"] == "M" else 0.0
    view = 1.0 if sample["view_position"] == "PA" else 0.0

    return torch.tensor([age, gender, view], dtype=torch.float32)