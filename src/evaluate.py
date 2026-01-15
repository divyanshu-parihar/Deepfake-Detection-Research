"""
Evaluation Script for Deepfake Detection.

Cross-dataset evaluation with AUC metrics and robustness testing.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config, setup_logger
from models import DualStreamDetector
from data import DeepfakeDataset, get_val_transforms

logger = setup_logger(__name__)


def load_model(checkpoint_path: str, config: Config, device: torch.device) -> DualStreamDetector:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        config: Model configuration.
        device: Device to load model on.
        
    Returns:
        Loaded model in eval mode.
    """
    model = DualStreamDetector(
        rgb_feature_dim=config.model.rgb_feature_dim,
        freq_feature_dim=config.model.freq_feature_dim,
        fusion_dim=config.model.fusion_dim,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
        pretrained=False,  # Don't load pretrained, we have our weights
        dct_block_size=config.model.dct_block_size,
        freq_channels=config.model.freq_channels,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")
    
    return model


@torch.no_grad()
def evaluate_dataset(
    model: DualStreamDetector,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Detection model.
        data_loader: Data loader.
        device: Device.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    all_probs = []
    all_labels = []
    correct = 0
    total = 0
    
    for images, labels in tqdm(data_loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits, _ = model(images, return_embeddings=False)
        probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of fake
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # Compute metrics
    try:
        from sklearn.metrics import (
            roc_auc_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        preds = (all_probs >= 0.5).astype(int)
        
        metrics = {
            "accuracy": accuracy_score(all_labels, preds) * 100,
            "auc": roc_auc_score(all_labels, all_probs) * 100,
            "precision": precision_score(all_labels, preds, zero_division=0) * 100,
            "recall": recall_score(all_labels, preds, zero_division=0) * 100,
            "f1": f1_score(all_labels, preds, zero_division=0) * 100,
        }
    except ImportError:
        logger.warning("sklearn not installed, computing basic metrics only")
        metrics = {
            "accuracy": 100 * correct / total,
        }
    
    return metrics


def evaluate_robustness(
    model: DualStreamDetector,
    dataset_root: str,
    dataset_type: str,
    device: torch.device,
    config: Config,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model robustness under various degradations.
    
    Args:
        model: Detection model.
        dataset_root: Path to dataset.
        dataset_type: Type of dataset.
        device: Device.
        config: Configuration.
        
    Returns:
        Dictionary mapping degradation type to metrics.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        logger.error("albumentations required for robustness testing")
        return {}
    
    results = {}
    
    # Define degradation types
    degradations = {
        "clean": [],
        "blur_sigma3": [A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(3, 3), p=1.0)],
        "jpeg_q50": [A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0)],
        "blur_jpeg": [
            A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(3, 3), p=1.0),
            A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0),
        ],
    }
    
    for deg_name, deg_transforms in degradations.items():
        logger.info(f"\nEvaluating with degradation: {deg_name}")
        
        transform = A.Compose([
            A.Resize(config.data.image_size[0], config.data.image_size[1]),
            *deg_transforms,
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        dataset = DeepfakeDataset(
            root=dataset_root,
            dataset_type=dataset_type,
            split="test",
            transform=transform,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )
        
        metrics = evaluate_dataset(model, loader, device)
        results[deg_name] = metrics
        
        logger.info(f"{deg_name}: AUC={metrics.get('auc', 0):.2f}%, Acc={metrics['accuracy']:.2f}%")
    
    return results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detector")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="celeb_df",
        choices=["ff++", "celeb_df"],
        help="Dataset type",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run robustness evaluation",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    config = get_config()
    config.training.batch_size = args.batch_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    if args.robustness:
        # Robustness evaluation
        results = evaluate_robustness(
            model, args.dataset_root, args.dataset_type, device, config
        )
        
        print("\n" + "=" * 60)
        print("ROBUSTNESS EVALUATION RESULTS")
        print("=" * 60)
        for deg_name, metrics in results.items():
            print(f"\n{deg_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.2f}%")
    else:
        # Standard evaluation
        dataset = DeepfakeDataset(
            root=args.dataset_root,
            dataset_type=args.dataset_type,
            split="test",
            transform=get_val_transforms(config.data.image_size),
        )
        
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )
        
        logger.info(f"Evaluating on {len(dataset)} samples")
        
        metrics = evaluate_dataset(model, loader, device)
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.2f}%")


if __name__ == "__main__":
    main()
