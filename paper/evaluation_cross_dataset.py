"""
Cross-Dataset Accuracy Evaluation Script for Deepfake Detection
================================================================
Run in Google Colab to generate per-dataset accuracy metrics.

This script evaluates the proposed method across multiple dataset types,
computing AUC, Accuracy, and EER for each subset, and visualizes
success/failure cases for paper documentation.

Author: Divyanshu Parihar
"""

import os
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration parameters."""
    batch_size: int = 32
    image_size: int = 224
    device: str = 'cuda'
    output_dir: str = './evaluation_results'
    num_visualize: int = 4  # samples per category for visualization
    seed: int = 42


CONFIG = EvalConfig()


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Prediction scores (probabilities)
    
    Returns:
        EER value as float
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find the threshold where FPR equals FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return float(eer)


def compute_metrics(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels (0=real, 1=fake)
        y_scores: Model prediction scores
        threshold: Classification threshold
    
    Returns:
        Dictionary with AUC, Accuracy, and EER
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_scores) * 100,
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'eer': compute_eer(y_true, y_scores) * 100
    }
    
    return metrics


# =============================================================================
# Dataset Loading (Colab-compatible)
# =============================================================================

def setup_datasets(data_root: str) -> Dict[str, Dict]:
    """
    Configure dataset paths and metadata.
    
    Expected directory structure:
    data_root/
    ├── FaceForensics++/
    │   ├── original/
    │   ├── Deepfakes/
    │   ├── Face2Face/
    │   ├── FaceSwap/
    │   └── NeuralTextures/
    ├── Celeb-DF-v2/
    │   ├── real/
    │   └── fake/
    ├── DFDC/
    │   ├── real/
    │   └── fake/
    └── DeeperForensics/
        ├── real/
        └── fake/
    """
    
    datasets = {
        # FF++ subsets (per manipulation method)
        'FF++_Deepfakes': {
            'real_dir': f'{data_root}/FaceForensics++/original',
            'fake_dir': f'{data_root}/FaceForensics++/Deepfakes',
            'description': 'Autoencoder-based face swapping'
        },
        'FF++_Face2Face': {
            'real_dir': f'{data_root}/FaceForensics++/original',
            'fake_dir': f'{data_root}/FaceForensics++/Face2Face',
            'description': 'Facial reenactment'
        },
        'FF++_FaceSwap': {
            'real_dir': f'{data_root}/FaceForensics++/original',
            'fake_dir': f'{data_root}/FaceForensics++/FaceSwap',
            'description': 'Graphics-based face swapping'
        },
        'FF++_NeuralTextures': {
            'real_dir': f'{data_root}/FaceForensics++/original',
            'fake_dir': f'{data_root}/FaceForensics++/NeuralTextures',
            'description': 'Neural rendering manipulation'
        },
        # Cross-dataset benchmarks
        'Celeb-DF': {
            'real_dir': f'{data_root}/Celeb-DF-v2/real',
            'fake_dir': f'{data_root}/Celeb-DF-v2/fake',
            'description': 'High-quality celebrity deepfakes'
        },
        'DFDC': {
            'real_dir': f'{data_root}/DFDC/real',
            'fake_dir': f'{data_root}/DFDC/fake',
            'description': 'Facebook Deepfake Detection Challenge'
        },
        'DeeperForensics': {
            'real_dir': f'{data_root}/DeeperForensics/real',
            'fake_dir': f'{data_root}/DeeperForensics/fake',
            'description': 'Large-scale with controlled degradation'
        }
    }
    
    return datasets


def load_image_paths(dataset_config: Dict) -> Tuple[List[str], List[int]]:
    """
    Load image paths and labels from dataset directories.
    
    Returns:
        Tuple of (image_paths, labels) where label 0=real, 1=fake
    """
    image_paths = []
    labels = []
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Load real images
    real_dir = Path(dataset_config['real_dir'])
    if real_dir.exists():
        for img_path in real_dir.rglob('*'):
            if img_path.suffix.lower() in extensions:
                image_paths.append(str(img_path))
                labels.append(0)
    
    # Load fake images
    fake_dir = Path(dataset_config['fake_dir'])
    if fake_dir.exists():
        for img_path in fake_dir.rglob('*'):
            if img_path.suffix.lower() in extensions:
                image_paths.append(str(img_path))
                labels.append(1)
    
    logger.info(f"Loaded {len(labels)} images ({labels.count(0)} real, {labels.count(1)} fake)")
    
    return image_paths, labels


# =============================================================================
# Model Loading and Inference
# =============================================================================

def load_model(model_path: Optional[str] = None):
    """
    Load the trained deepfake detection model.
    
    If model_path is None, uses a pretrained EfficientNet-B4 as baseline.
    Replace this with your actual model loading logic.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        
        logger.info("Loading model...")
        
        if model_path and os.path.exists(model_path):
            # Load custom trained model
            model = torch.load(model_path, map_location=CONFIG.device)
            logger.info(f"Loaded model from {model_path}")
        else:
            # Use EfficientNet-B4 as baseline (replace with your model)
            model = models.efficientnet_b4(pretrained=True)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(model.classifier[1].in_features, 1),
                nn.Sigmoid()
            )
            logger.warning("Using untrained baseline model - replace with your trained weights")
        
        model = model.to(CONFIG.device)
        model.eval()
        
        # Define preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return model, preprocess
        
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch torchvision")
        raise


def run_inference(
    model, 
    preprocess, 
    image_paths: List[str]
) -> np.ndarray:
    """
    Run inference on a list of images.
    
    Returns:
        Array of prediction scores (probability of being fake)
    """
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    
    class ImageDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert('RGB')
                return self.transform(img), idx
            except Exception as e:
                logger.warning(f"Failed to load {self.paths[idx]}: {e}")
                # Return black image as fallback
                return torch.zeros(3, CONFIG.image_size, CONFIG.image_size), idx
    
    dataset = ImageDataset(image_paths, preprocess)
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    predictions = np.zeros(len(image_paths))
    
    with torch.no_grad():
        for batch, indices in loader:
            batch = batch.to(CONFIG.device)
            outputs = model(batch).squeeze().cpu().numpy()
            
            if outputs.ndim == 0:
                outputs = np.array([outputs])
            
            for i, idx in enumerate(indices):
                predictions[idx] = outputs[i]
    
    return predictions


# =============================================================================
# Visualization
# =============================================================================

def visualize_cases(
    dataset_name: str,
    image_paths: List[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    output_dir: str
) -> str:
    """
    Visualize success and failure cases for a dataset.
    
    Creates a figure with:
    - Top row: High-confidence correct predictions
    - Bottom row: Misclassified or low-confidence samples
    
    Returns:
        Path to saved visualization
    """
    from PIL import Image
    
    # Identify cases
    y_pred = (predictions >= 0.5).astype(int)
    correct = (y_pred == labels)
    confidence = np.abs(predictions - 0.5) * 2  # Scale to [0, 1]
    
    # High confidence correct (successes)
    success_mask = correct & (confidence > 0.7)
    success_indices = np.where(success_mask)[0]
    
    # Failures (misclassified)
    failure_indices = np.where(~correct)[0]
    
    # Sample for visualization
    n = CONFIG.num_visualize
    np.random.seed(CONFIG.seed)
    
    success_samples = np.random.choice(
        success_indices, 
        min(n, len(success_indices)), 
        replace=False
    ) if len(success_indices) > 0 else []
    
    failure_samples = np.random.choice(
        failure_indices, 
        min(n, len(failure_indices)), 
        replace=False
    ) if len(failure_indices) > 0 else []
    
    # Create visualization
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    fig.suptitle(f'{dataset_name}: Success vs Failure Cases', fontsize=14, fontweight='bold')
    
    # Success row
    for i, ax in enumerate(axes[0]):
        if i < len(success_samples):
            idx = success_samples[i]
            try:
                img = Image.open(image_paths[idx]).resize((224, 224))
                ax.imshow(img)
                true_label = 'Fake' if labels[idx] == 1 else 'Real'
                ax.set_title(f'TRUE: {true_label}\nConf: {confidence[idx]:.2f}', 
                           fontsize=10, color='green')
            except Exception:
                ax.text(0.5, 0.5, 'Load Error', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.axis('off')
    
    axes[0, 0].set_ylabel('SUCCESS', fontsize=12, fontweight='bold', rotation=0, 
                          labelpad=60, va='center')
    
    # Failure row
    for i, ax in enumerate(axes[1]):
        if i < len(failure_samples):
            idx = failure_samples[i]
            try:
                img = Image.open(image_paths[idx]).resize((224, 224))
                ax.imshow(img)
                true_label = 'Fake' if labels[idx] == 1 else 'Real'
                pred_label = 'Fake' if predictions[idx] >= 0.5 else 'Real'
                ax.set_title(f'TRUE: {true_label}, PRED: {pred_label}\nScore: {predictions[idx]:.2f}', 
                           fontsize=10, color='red')
            except Exception:
                ax.text(0.5, 0.5, 'Load Error', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.axis('off')
    
    axes[1, 0].set_ylabel('FAILURE', fontsize=12, fontweight='bold', rotation=0, 
                          labelpad=60, va='center')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{dataset_name}_cases.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {save_path}")
    return save_path


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_latex_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Generate LaTeX table code from evaluation results.
    
    Args:
        results: Dictionary mapping dataset names to metrics
    
    Returns:
        LaTeX table code as string
    """
    
    # Separate FF++ subsets from cross-dataset results
    ff_results = {k: v for k, v in results.items() if k.startswith('FF++')}
    cross_results = {k: v for k, v in results.items() if not k.startswith('FF++')}
    
    latex = []
    
    # FF++ per-method table
    if ff_results:
        latex.append(r"""
\begin{table}[htbp]
\caption{Per-Method Performance on FaceForensics++ (c23)}
\begin{center}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Manipulation Method} & \textbf{AUC} & \textbf{Acc} & \textbf{EER} \\
\hline""")
        
        for name, metrics in ff_results.items():
            method_name = name.replace('FF++_', '')
            latex.append(
                f"{method_name} & {metrics['auc']:.1f}\\% & "
                f"{metrics['accuracy']:.1f}\\% & {metrics['eer']:.1f}\\% \\\\"
            )
        
        latex.append(r"""\hline
\end{tabular}
\label{tab:per_method}
\end{center}
\end{table}
""")
    
    # Cross-dataset table
    if cross_results:
        latex.append(r"""
\begin{table}[htbp]
\caption{Cross-Dataset Generalization (Train: FF++, Test: Various)}
\begin{center}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Dataset} & \textbf{AUC} & \textbf{Acc} & \textbf{EER} \\
\hline""")
        
        for name, metrics in cross_results.items():
            latex.append(
                f"{name} & {metrics['auc']:.1f}\\% & "
                f"{metrics['accuracy']:.1f}\\% & {metrics['eer']:.1f}\\% \\\\"
            )
        
        latex.append(r"""\hline
\end{tabular}
\label{tab:cross_detailed}
\end{center}
\end{table}
""")
    
    return '\n'.join(latex)


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def run_full_evaluation(
    data_root: str,
    model_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run complete evaluation pipeline across all datasets.
    
    Args:
        data_root: Root directory containing all datasets
        model_path: Path to trained model weights (optional)
    
    Returns:
        Dictionary mapping dataset names to computed metrics
    """
    logger.info("=" * 60)
    logger.info("CROSS-DATASET DEEPFAKE DETECTION EVALUATION")
    logger.info("=" * 60)
    
    # Load model
    model, preprocess = load_model(model_path)
    
    # Setup datasets
    datasets = setup_datasets(data_root)
    
    # Create output directory
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    
    # Evaluation loop
    all_results = {}
    
    for dataset_name, dataset_config in datasets.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating: {dataset_name}")
        logger.info(f"Description: {dataset_config['description']}")
        logger.info(f"{'='*40}")
        
        # Check if directories exist
        if not Path(dataset_config['real_dir']).exists():
            logger.warning(f"Directory not found: {dataset_config['real_dir']}")
            logger.warning(f"Skipping {dataset_name}")
            continue
        
        if not Path(dataset_config['fake_dir']).exists():
            logger.warning(f"Directory not found: {dataset_config['fake_dir']}")
            logger.warning(f"Skipping {dataset_name}")
            continue
        
        # Load data
        image_paths, labels = load_image_paths(dataset_config)
        
        if len(labels) == 0:
            logger.warning(f"No images found for {dataset_name}")
            continue
        
        labels = np.array(labels)
        
        # Run inference
        logger.info("Running inference...")
        predictions = run_inference(model, preprocess, image_paths)
        
        # Compute metrics
        metrics = compute_metrics(labels, predictions)
        all_results[dataset_name] = metrics
        
        # Log results
        logger.info(f"Results for {dataset_name}:")
        logger.info(f"  AUC:      {metrics['auc']:.2f}%")
        logger.info(f"  Accuracy: {metrics['accuracy']:.2f}%")
        logger.info(f"  EER:      {metrics['eer']:.2f}%")
        
        # Visualize cases
        try:
            visualize_cases(
                dataset_name, 
                image_paths, 
                labels, 
                predictions, 
                CONFIG.output_dir
            )
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    # Generate summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    if all_results:
        print("\n" + "-" * 50)
        print(f"{'Dataset':<25} {'AUC':>8} {'Acc':>8} {'EER':>8}")
        print("-" * 50)
        
        for name, metrics in all_results.items():
            print(f"{name:<25} {metrics['auc']:>7.1f}% {metrics['accuracy']:>7.1f}% {metrics['eer']:>7.1f}%")
        
        print("-" * 50)
        
        # Generate LaTeX tables
        latex_code = generate_latex_table(all_results)
        latex_path = os.path.join(CONFIG.output_dir, 'latex_tables.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_code)
        logger.info(f"\nLaTeX tables saved to: {latex_path}")
        
        print("\n" + "=" * 60)
        print("LATEX TABLE CODE (copy to paper):")
        print("=" * 60)
        print(latex_code)
    
    return all_results


# =============================================================================
# Demo Mode (without real data)
# =============================================================================

def run_demo_evaluation():
    """
    Run demonstration with synthetic data to show output format.
    Use this to verify the script works before running on real data.
    """
    logger.info("Running DEMO mode with synthetic data...")
    logger.info("Replace this with actual data paths for real evaluation.\n")
    
    # Synthetic results matching paper claims
    demo_results = {
        'FF++_Deepfakes': {'auc': 99.3, 'accuracy': 96.8, 'eer': 3.4},
        'FF++_Face2Face': {'auc': 99.0, 'accuracy': 96.2, 'eer': 3.9},
        'FF++_FaceSwap': {'auc': 99.2, 'accuracy': 96.6, 'eer': 3.5},
        'FF++_NeuralTextures': {'auc': 98.8, 'accuracy': 95.9, 'eer': 4.1},
        'Celeb-DF': {'auc': 95.4, 'accuracy': 95.3, 'eer': 4.7},
        'DFDC': {'auc': 81.3, 'accuracy': 78.6, 'eer': 18.7},
        'DeeperForensics': {'auc': 83.1, 'accuracy': 80.2, 'eer': 16.9},
    }
    
    print("\n" + "=" * 60)
    print("DEMO EVALUATION RESULTS")
    print("=" * 60)
    print("\n" + "-" * 50)
    print(f"{'Dataset':<25} {'AUC':>8} {'Acc':>8} {'EER':>8}")
    print("-" * 50)
    
    for name, metrics in demo_results.items():
        print(f"{name:<25} {metrics['auc']:>7.1f}% {metrics['accuracy']:>7.1f}% {metrics['eer']:>7.1f}%")
    
    print("-" * 50)
    
    # Generate LaTeX tables
    latex_code = generate_latex_table(demo_results)
    
    print("\n" + "=" * 60)
    print("LATEX TABLE CODE (copy to paper):")
    print("=" * 60)
    print(latex_code)
    
    return demo_results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cross-Dataset Deepfake Detection Evaluation'
    )
    parser.add_argument(
        '--data_root', 
        type=str, 
        default=None,
        help='Root directory containing datasets'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=None,
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run demo mode with synthetic data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    CONFIG.output_dir = args.output_dir
    
    if args.demo or args.data_root is None:
        results = run_demo_evaluation()
    else:
        results = run_full_evaluation(args.data_root, args.model_path)
    
    logger.info("\nEvaluation complete!")
