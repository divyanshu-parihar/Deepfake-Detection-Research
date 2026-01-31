"""Data package for deepfake detection."""
from .deepfake_dataset import (
    DeepfakeDataset,
    SyntheticDeepfakeDataset,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    "DeepfakeDataset",
    "SyntheticDeepfakeDataset",
    "get_train_transforms",
    "get_val_transforms",
]
