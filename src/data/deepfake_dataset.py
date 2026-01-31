"""
Dataset Module for Deepfake Detection.

Provides dataset classes and transforms for training on FaceForensics++,
Celeb-DF, and other deepfake datasets.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict
import random

import torch
from torch.utils.data import Dataset
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger(__name__)


# =============================================================================
# Transforms
# =============================================================================

def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> Callable:
    """
    Get training transforms with augmentation.
    
    Augmentations (matching paper):
    - Horizontal flip (50%)
    - Brightness/contrast jitter (±20%)
    - Gaussian noise (σ=0.01, 20%)
    - JPEG compression (quality 70-100, 30%)
    """
    def transform(image: np.ndarray) -> torch.Tensor:
        # Resize
        if HAS_CV2:
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
        elif HAS_PIL:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize(image_size, Image.BILINEAR)
            image = np.array(pil_img)
        
        # Convert to float [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Horizontal flip (50%)
        if random.random() < 0.5:
            image = image[:, ::-1, :].copy()
        
        # Brightness/contrast jitter (±20%)
        if random.random() < 0.5:
            factor = 1.0 + (random.random() - 0.5) * 0.4
            image = np.clip(image * factor, 0, 1)
        
        # Gaussian noise (20%)
        if random.random() < 0.2:
            noise = np.random.randn(*image.shape) * 0.01
            image = np.clip(image + noise, 0, 1).astype(np.float32)
        
        # JPEG compression simulation (30%)
        if random.random() < 0.3 and HAS_CV2:
            quality = random.randint(70, 100)
            _, encoded = cv2.imencode('.jpg', (image * 255).astype(np.uint8),
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        
        # Normalize (ImageNet)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = (image - mean) / std
        
        return torch.from_numpy(image).float()
    
    return transform


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> Callable:
    """Get validation transforms (no augmentation)."""
    def transform(image: np.ndarray) -> torch.Tensor:
        # Resize
        if HAS_CV2:
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
        elif HAS_PIL:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize(image_size, Image.BILINEAR)
            image = np.array(pil_img)
        
        # Convert to float [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        
        # Normalize (ImageNet)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = (image - mean) / std
        
        return torch.from_numpy(image).float()
    
    return transform


# =============================================================================
# Dataset Classes
# =============================================================================

class DeepfakeDataset(Dataset):
    """
    General deepfake detection dataset.
    
    Supports:
    - FaceForensics++ (ff++)
    - Celeb-DF
    - DFDC
    - Generic real/fake folder structure
    
    Expected structure for 'ff++':
    root/
    ├── original_sequences/
    │   └── youtube/
    │       └── c23/
    │           └── videos/
    └── manipulated_sequences/
        ├── Deepfakes/
        │   └── c23/
        │       └── videos/
        ├── Face2Face/
        ├── FaceSwap/
        └── NeuralTextures/
    
    Expected structure for 'generic':
    root/
    ├── train/
    │   ├── real/
    │   └── fake/
    └── val/
        ├── real/
        └── fake/
    """
    
    def __init__(
        self,
        root: str,
        dataset_type: str = 'generic',
        split: str = 'train',
        compression: str = 'c23',
        manipulation_types: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            root: Root directory of the dataset.
            dataset_type: One of 'ff++', 'celeb_df', 'dfdc', 'generic'.
            split: 'train', 'val', or 'test'.
            compression: Compression level for FF++ ('c23' or 'c40').
            manipulation_types: List of manipulation types for FF++.
            transform: Transform function to apply to images.
            max_samples_per_class: Max samples per class for balancing.
        """
        self.root = Path(root)
        self.dataset_type = dataset_type
        self.split = split
        self.compression = compression
        self.transform = transform
        self.max_samples = max_samples_per_class
        
        if manipulation_types is None:
            manipulation_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        self.manipulation_types = manipulation_types
        
        # Load samples
        self.samples = self._load_samples()
        
        logger.info(
            f"DeepfakeDataset [{dataset_type}] [{split}]: "
            f"{len(self.samples)} samples loaded"
        )
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load sample paths and labels."""
        if self.dataset_type == 'ff++':
            return self._load_ff_plus_plus()
        elif self.dataset_type == 'celeb_df':
            return self._load_celeb_df()
        elif self.dataset_type == 'dfdc':
            return self._load_dfdc()
        else:
            return self._load_generic()
    
    def _load_generic(self) -> List[Tuple[Path, int]]:
        """Load from generic real/fake folder structure."""
        samples = []
        split_dir = self.root / self.split
        
        # Real (label=0)
        real_dir = split_dir / 'real'
        if real_dir.exists():
            real_samples = self._get_images_from_dir(real_dir)
            samples.extend([(p, 0) for p in real_samples])
        
        # Fake (label=1)
        fake_dir = split_dir / 'fake'
        if fake_dir.exists():
            fake_samples = self._get_images_from_dir(fake_dir)
            samples.extend([(p, 1) for p in fake_samples])
        
        return samples
    
    def _load_ff_plus_plus(self) -> List[Tuple[Path, int]]:
        """Load FaceForensics++ dataset."""
        samples = []
        
        # Real videos
        real_dir = self.root / 'original_sequences' / 'youtube' / self.compression / 'videos'
        if real_dir.exists():
            real_samples = self._get_images_from_dir(real_dir)
            samples.extend([(p, 0) for p in real_samples])
        
        # Fake videos (all manipulation types)
        for manip_type in self.manipulation_types:
            fake_dir = self.root / 'manipulated_sequences' / manip_type / self.compression / 'videos'
            if fake_dir.exists():
                fake_samples = self._get_images_from_dir(fake_dir)
                samples.extend([(p, 1) for p in fake_samples])
        
        return samples
    
    def _load_celeb_df(self) -> List[Tuple[Path, int]]:
        """Load Celeb-DF dataset."""
        samples = []
        
        # Real
        real_patterns = ['Celeb-real', 'YouTube-real']
        for pattern in real_patterns:
            real_dir = self.root / pattern
            if real_dir.exists():
                real_samples = self._get_images_from_dir(real_dir)
                samples.extend([(p, 0) for p in real_samples])
        
        # Fake
        fake_dir = self.root / 'Celeb-synthesis'
        if fake_dir.exists():
            fake_samples = self._get_images_from_dir(fake_dir)
            samples.extend([(p, 1) for p in fake_samples])
        
        return samples
    
    def _load_dfdc(self) -> List[Tuple[Path, int]]:
        """Load DFDC dataset."""
        samples = []
        
        # DFDC uses metadata.json for labels
        # Simplified: assume 'real' and 'fake' subdirs
        return self._load_generic()
    
    def _get_images_from_dir(self, directory: Path) -> List[Path]:
        """Get all image files from directory recursively."""
        images = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        
        for ext in extensions:
            images.extend(directory.rglob(ext))
        
        # Apply max samples limit
        if self.max_samples and len(images) > self.max_samples:
            random.shuffle(images)
            images = images[:self.max_samples]
        
        return images
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        try:
            # Load image
            if HAS_CV2:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Failed to load {path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif HAS_PIL:
                image = np.array(Image.open(path).convert('RGB'))
            else:
                raise ImportError("Neither cv2 nor PIL available")
            
            # Apply transform
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform
                image = image.astype(np.float32) / 255.0
                image = image.transpose(2, 0, 1)
                image = torch.from_numpy(image)
            
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        return image, label


# =============================================================================
# Synthetic Dataset (for testing without real data)
# =============================================================================

class SyntheticDeepfakeDataset(Dataset):
    """
    Synthetic dataset for testing.
    
    Generates fake samples by applying upsampling artifacts to real images.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: Tuple[int, int] = (224, 224),
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info(f"SyntheticDeepfakeDataset: {num_samples} samples")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Generate random image
        image = np.random.rand(*self.image_size, 3).astype(np.float32)
        
        # Alternate between real (0) and fake (1)
        label = idx % 2
        
        if label == 1:
            # Simulate upsampling artifacts for fake
            small = cv2.resize(image, (self.image_size[0]//2, self.image_size[1]//2)) if HAS_CV2 else image[::2, ::2]
            image = cv2.resize(small, self.image_size) if HAS_CV2 else np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        
        # Normalize
        image = image.transpose(2, 0, 1)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = (image - mean) / std
        
        return torch.from_numpy(image).float(), label
