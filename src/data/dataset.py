"""
Dataset Loaders for Deepfake Detection.

Provides dataset classes for FaceForensics++ and Celeb-DF benchmarks.
"""
import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

try:
    from torchvision import transforms
except ImportError:
    transforms = None

logger = logging.getLogger(__name__)


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> Callable:
    """
    Get training data augmentations.
    
    Args:
        image_size: Target image size (H, W).
        
    Returns:
        Transform function.
    """
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> Callable:
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        image_size: Target image size (H, W).
        
    Returns:
        Transform function.
    """
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection.
    
    Supports both FaceForensics++ and Celeb-DF formats.
    Expects pre-extracted face crops.
    
    Directory structure (FF++):
        root/
            original_sequences/
                youtube/
                    c23/
                        000/
                            0000.png, 0001.png, ...
            manipulated_sequences/
                Deepfakes/
                    c23/
                        000_001/
                            0000.png, ...
    
    Directory structure (Celeb-DF):
        root/
            Celeb-real/
                id0_0000.png, ...
            Celeb-synthesis/
                id0_id1_0000.png, ...
    
    Attributes:
        samples: List of (image_path, label) tuples.
        transform: Data transformation/augmentation function.
    """
    
    def __init__(
        self,
        root: str,
        dataset_type: str = "ff++",
        split: str = "train",
        compression: str = "c23",
        manipulation_types: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        max_frames_per_video: int = 10,
    ) -> None:
        """
        Initialize dataset.
        
        Args:
            root: Root directory of dataset.
            dataset_type: "ff++" or "celeb_df".
            split: "train", "val", or "test".
            compression: Compression level for FF++ ("c23" or "c40").
            manipulation_types: List of manipulation types for FF++.
            transform: Data transformation function.
            max_frames_per_video: Max frames to sample per video.
        """
        self.root = Path(root)
        self.dataset_type = dataset_type.lower()
        self.split = split
        self.compression = compression
        self.transform = transform
        self.max_frames = max_frames_per_video
        
        if manipulation_types is None:
            manipulation_types = [
                "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"
            ]
        self.manipulation_types = manipulation_types
        
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()
        
        logger.info(
            f"DeepfakeDataset initialized: {self.dataset_type}, "
            f"split={split}, samples={len(self.samples)}"
        )
    
    def _load_samples(self) -> None:
        """Load sample paths and labels."""
        if self.dataset_type == "ff++":
            self._load_ff_samples()
        elif self.dataset_type == "celeb_df":
            self._load_celeb_df_samples()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _load_ff_samples(self) -> None:
        """Load FaceForensics++ samples."""
        # Real samples
        real_dir = self.root / "original_sequences" / "youtube" / self.compression
        if real_dir.exists():
            self._add_samples_from_dir(real_dir, label=0)
        
        # Fake samples
        for manip in self.manipulation_types:
            fake_dir = (
                self.root / "manipulated_sequences" / manip / self.compression
            )
            if fake_dir.exists():
                self._add_samples_from_dir(fake_dir, label=1)
    
    def _load_celeb_df_samples(self) -> None:
        """Load Celeb-DF samples."""
        # Real samples
        real_dir = self.root / "Celeb-real"
        if real_dir.exists():
            self._add_samples_from_dir(real_dir, label=0, flat=True)
        
        # YouTube real (higher quality)
        yt_real = self.root / "YouTube-real"
        if yt_real.exists():
            self._add_samples_from_dir(yt_real, label=0, flat=True)
        
        # Fake samples
        fake_dir = self.root / "Celeb-synthesis"
        if fake_dir.exists():
            self._add_samples_from_dir(fake_dir, label=1, flat=True)
    
    def _add_samples_from_dir(
        self, 
        directory: Path, 
        label: int,
        flat: bool = False,
    ) -> None:
        """
        Add samples from a directory.
        
        Args:
            directory: Path to directory.
            label: Label for samples (0=real, 1=fake).
            flat: If True, images are directly in directory.
                  If False, images are in subdirectories (video folders).
        """
        if flat:
            # All images directly in directory
            image_files = list(directory.glob("*.png")) + list(directory.glob("*.jpg"))
            for img_path in image_files[:self.max_frames * 100]:  # Limit total
                self.samples.append((img_path, label))
        else:
            # Images in video subdirectories
            video_dirs = [d for d in directory.iterdir() if d.is_dir()]
            for video_dir in video_dirs:
                image_files = sorted(
                    list(video_dir.glob("*.png")) + list(video_dir.glob("*.jpg"))
                )
                # Sample frames from video
                step = max(1, len(image_files) // self.max_frames)
                sampled = image_files[::step][:self.max_frames]
                for img_path in sampled:
                    self.samples.append((img_path, label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get sample by index.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (image_tensor, label).
        """
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return black image on error
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform is not None:
            if HAS_ALBUMENTATIONS:
                image = np.array(image)
                transformed = self.transform(image=image)
                image = transformed["image"]
            else:
                image = self.transform(image)
        
        return image, label


def create_dataloaders(
    train_root: str,
    val_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    train_dataset_type: str = "ff++",
    val_dataset_type: str = "celeb_df",
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_root: Path to training dataset.
        val_root: Path to validation dataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        image_size: Target image size.
        train_dataset_type: Type of training dataset.
        val_dataset_type: Type of validation dataset.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = DeepfakeDataset(
        root=train_root,
        dataset_type=train_dataset_type,
        split="train",
        transform=get_train_transforms(image_size),
    )
    
    val_dataset = DeepfakeDataset(
        root=val_root,
        dataset_type=val_dataset_type,
        split="val",
        transform=get_val_transforms(image_size),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
