"""
Configuration for Deepfake Detection Training.

This module contains all hyperparameters and path configurations
for training the dual-stream deepfake detection model.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import logging


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # RGB Stream
    rgb_backbone: str = "efficientnet_b4"
    rgb_pretrained: bool = True
    rgb_feature_dim: int = 1792  # EfficientNet-B4 output
    
    # Frequency Stream
    dct_block_size: int = 8
    freq_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    freq_feature_dim: int = 256
    
    # Fusion
    fusion_dim: int = 512
    dropout: float = 0.3
    num_classes: int = 2


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    
    # Contrastive Loss
    temperature: float = 0.07
    contrastive_weight: float = 0.5
    
    # Checkpointing
    save_every: int = 5
    early_stopping_patience: int = 10


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Paths (override these for your setup)
    ff_root: Path = Path("data/FaceForensics++")
    celeb_df_root: Path = Path("data/Celeb-DF-v2")
    
    # Preprocessing
    image_size: Tuple[int, int] = (224, 224)
    
    # FF++ manipulation types
    manipulation_types: List[str] = field(
        default_factory=lambda: [
            "Deepfakes",
            "Face2Face",
            "FaceSwap",
            "NeuralTextures"
        ]
    )
    
    # Compression quality (c23 = light, c40 = heavy)
    compression: str = "c23"
    
    # Train/val split
    train_split: float = 0.8
    val_split: float = 0.2


@dataclass
class Config:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Paths
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    
    # Device
    device: str = "cuda"
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self) -> None:
        """Create directories if they don't exist."""
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Factory function to get default configuration."""
    return Config()
