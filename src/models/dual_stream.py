"""
Dual-Stream Deepfake Detection Model.

Combines RGB stream (EfficientNet-B4) with frequency stream (DCT features)
for cross-domain generalization.
"""
import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional

try:
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
except ImportError:
    from torchvision.models import efficientnet_b4

from .dct_extractor import DCTExtractor, FrequencyStreamCNN

logger = logging.getLogger(__name__)


class DualStreamDetector(nn.Module):
    """
    Dual-stream deepfake detector combining RGB and frequency features.
    
    Architecture:
        - RGB Stream: EfficientNet-B4 backbone for semantic features
        - Frequency Stream: DCT extraction + lightweight CNN
        - Fusion: Concatenation + FC layers
        - Output: Binary classification (real/fake) + embeddings for contrastive loss
    
    Attributes:
        rgb_backbone: EfficientNet-B4 feature extractor.
        dct_extractor: DCT frequency extraction module.
        freq_cnn: CNN for frequency feature extraction.
        fusion: Feature fusion and classification head.
    """
    
    def __init__(
        self,
        rgb_feature_dim: int = 1792,
        freq_feature_dim: int = 256,
        fusion_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
        dct_block_size: int = 8,
        freq_channels: list = None,
    ) -> None:
        """
        Initialize dual-stream detector.
        
        Args:
            rgb_feature_dim: Output dimension of EfficientNet-B4.
            freq_feature_dim: Output dimension of frequency stream.
            fusion_dim: Dimension after fusion.
            num_classes: Number of output classes.
            dropout: Dropout rate.
            pretrained: Use pretrained EfficientNet weights.
            dct_block_size: Block size for DCT.
            freq_channels: Channel sizes for frequency CNN.
        """
        super().__init__()
        
        if freq_channels is None:
            freq_channels = [64, 128, 256]
        
        # RGB Stream: EfficientNet-B4
        self.rgb_backbone = self._build_rgb_backbone(pretrained)
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        
        # Frequency Stream
        self.dct_extractor = DCTExtractor(block_size=dct_block_size)
        self.freq_cnn = FrequencyStreamCNN(
            in_channels=1,
            channels=freq_channels,
            feature_dim=freq_feature_dim,
            dropout=dropout,
        )
        
        # Fusion layers
        combined_dim = rgb_feature_dim + freq_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Embedding projection (for contrastive loss)
        self.embedding_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, 128),
        )
        
        # Classification head
        self.classifier = nn.Linear(fusion_dim, num_classes)
        
        self._init_fusion_weights()
        
        logger.info(
            f"DualStreamDetector initialized: "
            f"rgb_dim={rgb_feature_dim}, freq_dim={freq_feature_dim}, "
            f"fusion_dim={fusion_dim}"
        )
    
    def _build_rgb_backbone(self, pretrained: bool) -> nn.Module:
        """Build EfficientNet-B4 backbone without classification head."""
        try:
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b4(weights=weights)
        except NameError:
            model = efficientnet_b4(pretrained=pretrained)
        
        # Remove classifier, keep features
        model.classifier = nn.Identity()
        return model.features
    
    def _init_fusion_weights(self) -> None:
        """Initialize fusion layer weights."""
        for m in [self.fusion, self.embedding_proj, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def extract_rgb_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract RGB stream features.
        
        Args:
            x: Input image tensor (B, 3, H, W).
            
        Returns:
            RGB feature vector (B, rgb_feature_dim).
        """
        features = self.rgb_backbone(x)
        features = self.rgb_pool(features)
        features = features.flatten(1)
        return features
    
    def extract_freq_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency stream features.
        
        Args:
            x: Input image tensor (B, 3, H, W).
            
        Returns:
            Frequency feature vector (B, freq_feature_dim).
        """
        dct_features = self.dct_extractor(x)
        freq_features = self.freq_cnn(dct_features)
        return freq_features
    
    def forward(
        self, 
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W).
            return_embeddings: Whether to return embeddings for contrastive loss.
            
        Returns:
            Tuple of (logits, embeddings).
            - logits: Classification logits (B, num_classes).
            - embeddings: Normalized embeddings (B, 128) if return_embeddings=True.
        """
        # Extract features from both streams
        rgb_features = self.extract_rgb_features(x)
        freq_features = self.extract_freq_features(x)
        
        # Fuse features
        combined = torch.cat([rgb_features, freq_features], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        
        # Embeddings for contrastive loss
        embeddings = None
        if return_embeddings:
            embeddings = self.embedding_proj(fused)
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return logits, embeddings
    
    def get_trainable_params(self) -> list:
        """Get parameters grouped for different learning rates."""
        backbone_params = list(self.rgb_backbone.parameters())
        other_params = (
            list(self.dct_extractor.parameters()) +
            list(self.freq_cnn.parameters()) +
            list(self.fusion.parameters()) +
            list(self.embedding_proj.parameters()) +
            list(self.classifier.parameters())
        )
        return [
            {"params": backbone_params, "lr_scale": 0.1},
            {"params": other_params, "lr_scale": 1.0},
        ]


class ChannelAttention(nn.Module):
    """
    Channel attention module from CBAM.
    
    Focuses on 'what' is meaningful by computing channel-wise attention weights.
    """
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial attention module from CBAM.
    
    Focuses on 'where' is meaningful by computing spatial attention weights.
    """
    
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(combined))
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Sequentially applies channel and spatial attention for better feature focus.
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class EnhancedDualStreamDetector(nn.Module):
    """
    Enhanced dual-stream detector with CBAM attention and improved frequency stream.
    
    Improvements over base DualStreamDetector:
    - CBAM attention after RGB backbone
    - EnhancedFrequencyStream with SRM filters + multi-scale DCT
    - Larger embedding dimension for better contrastive learning
    
    Target: 85%+ AUC on cross-dataset (Celeb-DF)
    """
    
    def __init__(
        self,
        rgb_feature_dim: int = 1792,
        freq_feature_dim: int = 256,
        fusion_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
        use_cbam: bool = True,
        use_srm: bool = True,
        use_multiscale: bool = True,
    ) -> None:
        """
        Initialize enhanced dual-stream detector.
        
        Args:
            rgb_feature_dim: Output dimension of EfficientNet-B4.
            freq_feature_dim: Output dimension of frequency stream.
            fusion_dim: Dimension after fusion.
            num_classes: Number of output classes.
            dropout: Dropout rate.
            pretrained: Use pretrained EfficientNet weights.
            use_cbam: Whether to use CBAM attention.
            use_srm: Whether to use SRM filters in frequency stream.
            use_multiscale: Whether to use multi-scale DCT.
        """
        super().__init__()
        
        self.use_cbam = use_cbam
        
        # Import enhanced frequency stream
        from .dct_extractor import EnhancedFrequencyStream
        
        # RGB Stream: EfficientNet-B4 + optional CBAM
        self.rgb_backbone = self._build_rgb_backbone(pretrained)
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        
        if use_cbam:
            self.cbam = CBAM(channels=rgb_feature_dim, reduction=16)
        
        # Enhanced Frequency Stream
        self.freq_stream = EnhancedFrequencyStream(
            feature_dim=freq_feature_dim,
            dropout=dropout,
            use_srm=use_srm,
            use_multiscale=use_multiscale,
        )
        
        # Fusion layers with attention
        combined_dim = rgb_feature_dim + freq_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
        )
        
        # Embedding projection (256-dim for stronger contrastive learning)
        self.embedding_proj = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )
        
        # Classification head
        self.classifier = nn.Linear(fusion_dim, num_classes)
        
        self._init_weights()
        
        logger.info(
            f"EnhancedDualStreamDetector initialized: "
            f"cbam={use_cbam}, srm={use_srm}, multiscale={use_multiscale}"
        )
    
    def _build_rgb_backbone(self, pretrained: bool) -> nn.Module:
        """Build EfficientNet-B4 backbone."""
        try:
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b4(weights=weights)
        except NameError:
            model = efficientnet_b4(pretrained=pretrained)
        
        model.classifier = nn.Identity()
        return model.features
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in [self.fusion, self.embedding_proj, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W).
            return_embeddings: Whether to return embeddings.
            
        Returns:
            Tuple of (logits, embeddings).
        """
        # RGB stream with optional CBAM
        rgb_feat = self.rgb_backbone(x)
        if self.use_cbam:
            rgb_feat = self.cbam(rgb_feat)
        rgb_feat = self.rgb_pool(rgb_feat).flatten(1)
        
        # Enhanced frequency stream
        freq_feat = self.freq_stream(x)
        
        # Fusion
        combined = torch.cat([rgb_feat, freq_feat], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        
        # Embeddings
        embeddings = None
        if return_embeddings:
            embeddings = self.embedding_proj(fused)
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return logits, embeddings
    
    def get_trainable_params(self) -> list:
        """Get parameters grouped for different learning rates."""
        backbone_params = list(self.rgb_backbone.parameters())
        other_params = (
            list(self.freq_stream.parameters()) +
            list(self.fusion.parameters()) +
            list(self.embedding_proj.parameters()) +
            list(self.classifier.parameters())
        )
        if self.use_cbam:
            other_params.extend(list(self.cbam.parameters()))
        
        return [
            {"params": backbone_params, "lr_scale": 0.1},
            {"params": other_params, "lr_scale": 1.0},
        ]

