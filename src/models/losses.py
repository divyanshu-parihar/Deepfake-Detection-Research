"""
Loss Functions for Deepfake Detection.

Implements contrastive loss for generator-agnostic fake clustering
and combined loss for training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    Clusters all fake samples together regardless of generator type,
    while separating them from real samples.
    
    Based on: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    
    Attributes:
        temperature: Temperature scaling parameter (tau).
    """
    
    def __init__(self, temperature: float = 0.07) -> None:
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature for softmax scaling (default 0.07).
        """
        super().__init__()
        self.temperature = temperature
        logger.info(f"ContrastiveLoss initialized: temperature={temperature}")
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        For deepfake detection:
        - labels=0: real face
        - labels=1: fake face (any generator)
        
        We want all fakes to cluster together, regardless of which
        generator created them.
        
        Args:
            embeddings: L2-normalized embeddings (B, embed_dim).
            labels: Binary labels (B,). 0=real, 1=fake.
            
        Returns:
            Scalar contrastive loss.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        if batch_size < 2:
            logger.warning("Batch size < 2, skipping contrastive loss")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute similarity matrix: sim(z_i, z_j) = z_i @ z_j.T
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.view(-1, 1)
        mask_positives = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask_self = torch.eye(batch_size, device=device)
        mask_positives = mask_positives * (1 - mask_self)
        
        # Count positives per sample
        num_positives = mask_positives.sum(dim=1)
        
        # Handle samples with no positives
        has_positives = num_positives > 0
        if not has_positives.any():
            logger.warning("No positive pairs in batch")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute log softmax (exclude self)
        # For numerical stability, subtract max
        sim_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix - sim_max.detach()
        
        # Exp of similarities
        exp_sim = torch.exp(similarity_matrix) * (1 - mask_self)
        
        # Denominator: sum over all negatives and positives (excluding self)
        log_denominator = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Log probability of positives
        log_prob = similarity_matrix - log_denominator
        
        # Mean log probability over positive pairs
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / (num_positives + 1e-8)
        
        # Loss: negative mean log probability (only for samples with positives)
        loss = -mean_log_prob_pos[has_positives].mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for deepfake detection.
    
    Combines:
    - Cross-entropy loss for classification
    - Contrastive loss for generator-agnostic representations
    
    Attributes:
        ce_loss: Cross-entropy loss module.
        contrastive_loss: Contrastive loss module.
        contrastive_weight: Weight for contrastive loss term.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        contrastive_weight: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Initialize combined loss.
        
        Args:
            temperature: Temperature for contrastive loss.
            contrastive_weight: Weight for contrastive loss (lambda).
            label_smoothing: Label smoothing for CE loss.
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.contrastive_weight = contrastive_weight
        
        logger.info(
            f"CombinedLoss initialized: "
            f"contrastive_weight={contrastive_weight}, "
            f"label_smoothing={label_smoothing}"
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            logits: Classification logits (B, num_classes).
            embeddings: L2-normalized embeddings (B, embed_dim).
            labels: Ground truth labels (B,).
            
        Returns:
            Tuple of (total_loss, loss_dict).
            - total_loss: Weighted sum of CE and contrastive loss.
            - loss_dict: Dict with individual loss values for logging.
        """
        # Classification loss
        ce = self.ce_loss(logits, labels)
        
        # Contrastive loss
        con = self.contrastive_loss(embeddings, labels)
        
        # Combined
        total = ce + self.contrastive_weight * con
        
        loss_dict = {
            "loss_total": total.item(),
            "loss_ce": ce.item(),
            "loss_contrastive": con.item(),
        }
        
        return total, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    
    Reduces loss for well-classified examples, focusing training on
    hard misclassified samples. Proven effective for detection tasks.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for class imbalance.
            gamma: Focusing parameter (higher = more focus on hard examples).
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        logger.info(f"FocalLoss initialized: alpha={alpha}, gamma={gamma}")
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Predictions (B, num_classes).
            labels: Ground truth labels (B,).
            
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class EnhancedCombinedLoss(nn.Module):
    """
    Enhanced combined loss with focal loss, contrastive learning, and mixup support.
    
    Improvements over basic CombinedLoss:
    - Focal loss for hard example mining
    - Higher label smoothing (0.1) for better calibration
    - Optional center loss for tighter clustering
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        contrastive_weight: float = 0.5,
        focal_weight: float = 0.3,
        label_smoothing: float = 0.1,
        use_focal: bool = True,
    ) -> None:
        """
        Initialize enhanced combined loss.
        
        Args:
            temperature: Temperature for contrastive loss.
            contrastive_weight: Weight for contrastive loss.
            focal_weight: Weight for focal loss component.
            label_smoothing: Label smoothing factor.
            use_focal: Whether to use focal loss.
        """
        super().__init__()
        
        self.use_focal = use_focal
        self.focal_weight = focal_weight
        
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.contrastive_weight = contrastive_weight
        
        if use_focal:
            self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        logger.info(
            f"EnhancedCombinedLoss initialized: "
            f"contrastive_weight={contrastive_weight}, "
            f"focal_weight={focal_weight}, "
            f"label_smoothing={label_smoothing}, "
            f"use_focal={use_focal}"
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute enhanced combined loss.
        
        Args:
            logits: Classification logits (B, num_classes).
            embeddings: L2-normalized embeddings (B, embed_dim).
            labels: Ground truth labels (B,).
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Cross-entropy with label smoothing
        ce = self.ce_loss(logits, labels)
        
        # Contrastive loss for clustering
        con = self.contrastive_loss(embeddings, labels)
        
        # Combined base loss
        total = ce + self.contrastive_weight * con
        
        loss_dict = {
            "loss_ce": ce.item(),
            "loss_contrastive": con.item(),
        }
        
        # Add focal loss if enabled
        if self.use_focal:
            focal = self.focal_loss(logits, labels)
            total = total + self.focal_weight * focal
            loss_dict["loss_focal"] = focal.item()
        
        loss_dict["loss_total"] = total.item()
        
        return total, loss_dict

