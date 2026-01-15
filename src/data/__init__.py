"""Data package for deepfake detection."""
from .dataset import DeepfakeDataset, get_train_transforms, get_val_transforms

__all__ = [
    "DeepfakeDataset",
    "get_train_transforms",
    "get_val_transforms",
]
