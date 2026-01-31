"""
DCT Frequency Extraction Module.

Implements Discrete Cosine Transform for extracting high-frequency
artifacts from images, as described in the paper methodology.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class DCTExtractor(nn.Module):
    """
    Extracts high-frequency DCT features from images.
    
    The module processes images in 8x8 blocks, applies DCT transform,
    and filters out low-frequency components to isolate upsampling artifacts.
    
    Attributes:
        block_size: Size of DCT blocks (default 8x8).
        dct_matrix: Precomputed DCT basis matrix.
        high_freq_mask: Mask for high-frequency coefficient selection.
    """
    
    def __init__(
        self,
        block_size: int = 8,
        keep_ratio: float = 0.5,
    ) -> None:
        """
        Initialize DCT extractor.
        
        Args:
            block_size: Size of blocks for DCT (default 8).
            keep_ratio: Ratio of high-frequency coefficients to keep.
        """
        super().__init__()
        self.block_size = block_size
        self.keep_ratio = keep_ratio
        
        # Precompute DCT basis matrix
        dct_matrix = self._compute_dct_matrix(block_size)
        self.register_buffer("dct_matrix", dct_matrix)
        
        # Create high-frequency mask (keep bottom-right, zero top-left)
        mask = self._create_highfreq_mask(block_size, keep_ratio)
        self.register_buffer("high_freq_mask", mask)
        
        logger.info(
            f"DCTExtractor initialized: block_size={block_size}, "
            f"keep_ratio={keep_ratio}"
        )
    
    def _compute_dct_matrix(self, n: int) -> torch.Tensor:
        """
        Compute the DCT-II basis matrix.
        
        The DCT of a signal x is: X[k] = sum_n x[n] * cos(pi*k*(2n+1)/(2N))
        
        Args:
            n: Size of the DCT matrix.
            
        Returns:
            DCT basis matrix of shape (n, n).
        """
        dct_mat = np.zeros((n, n), dtype=np.float32)
        
        for k in range(n):
            for i in range(n):
                if k == 0:
                    dct_mat[k, i] = 1.0 / np.sqrt(n)
                else:
                    dct_mat[k, i] = np.sqrt(2.0 / n) * np.cos(
                        np.pi * k * (2 * i + 1) / (2 * n)
                    )
        
        return torch.from_numpy(dct_mat)
    
    def _create_highfreq_mask(
        self, 
        block_size: int, 
        keep_ratio: float
    ) -> torch.Tensor:
        """
        Create mask to keep high-frequency DCT coefficients.
        
        Low frequencies are in the top-left corner of DCT block.
        We zero them out and keep the bottom-right (high frequencies).
        
        Args:
            block_size: Size of DCT block.
            keep_ratio: Ratio of coefficients to keep.
            
        Returns:
            Binary mask of shape (block_size, block_size).
        """
        mask = torch.ones(block_size, block_size)
        
        # Zero out top-left quadrant (low frequencies)
        cutoff = int(block_size * (1 - keep_ratio))
        mask[:cutoff, :cutoff] = 0.0
        
        return mask
    
    def _rgb_to_grayscale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB image to grayscale.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            Grayscale tensor of shape (B, 1, H, W).
        """
        # ITU-R BT.601 weights
        weights = torch.tensor(
            [0.299, 0.587, 0.114], 
            device=x.device, 
            dtype=x.dtype
        ).view(1, 3, 1, 1)
        
        return (x * weights).sum(dim=1, keepdim=True)
    
    def _apply_dct_2d(self, block: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D DCT to a block.
        
        2D DCT is separable: DCT_2D = DCT_matrix @ block @ DCT_matrix.T
        
        Args:
            block: Input block of shape (B, 1, block_size, block_size).
            
        Returns:
            DCT coefficients of same shape.
        """
        # block: (B, 1, H, W)
        dct_mat = self.dct_matrix  # (block_size, block_size)
        
        # Apply DCT: Y = D @ X @ D^T
        # First apply along rows, then columns
        result = torch.einsum("ij,...jk->...ik", dct_mat, block)
        result = torch.einsum("...ij,kj->...ik", result, dct_mat)
        
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract high-frequency DCT features from input image.
        
        Args:
            x: Input RGB image tensor of shape (B, 3, H, W).
               Values should be in [0, 1] range.
               
        Returns:
            High-frequency feature map of shape (B, 1, H, W).
        """
        batch_size, _, height, width = x.shape
        
        # Validate dimensions are divisible by block size
        if height % self.block_size != 0 or width % self.block_size != 0:
            # Pad to make divisible
            pad_h = (self.block_size - height % self.block_size) % self.block_size
            pad_w = (self.block_size - width % self.block_size) % self.block_size
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            height, width = x.shape[2], x.shape[3]
        
        # Convert to grayscale
        gray = self._rgb_to_grayscale(x)  # (B, 1, H, W)
        
        # Reshape into blocks
        # (B, 1, H, W) -> (B, 1, H//bs, bs, W//bs, bs)
        bs = self.block_size
        blocks = gray.unfold(2, bs, bs).unfold(3, bs, bs)
        # (B, 1, num_h, num_w, bs, bs)
        num_h, num_w = blocks.shape[2], blocks.shape[3]
        
        # Reshape for batch processing
        blocks = blocks.contiguous().view(batch_size, -1, bs, bs)
        # (B, num_blocks, bs, bs)
        
        # Apply DCT to each block
        dct_blocks = self._apply_dct_2d(blocks)
        
        # Apply high-frequency mask
        dct_blocks = dct_blocks * self.high_freq_mask
        
        # Reshape back to spatial layout
        dct_blocks = dct_blocks.view(batch_size, 1, num_h, num_w, bs, bs)
        dct_blocks = dct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        output = dct_blocks.view(batch_size, 1, num_h * bs, num_w * bs)
        
        return output


class FrequencyStreamCNN(nn.Module):
    """
    Lightweight CNN for processing DCT frequency features.
    
    Takes high-frequency DCT features and extracts discriminative
    representations for detecting upsampling artifacts.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: list = None,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize frequency stream CNN.
        
        Args:
            in_channels: Number of input channels (1 for grayscale DCT).
            channels: List of channel sizes for conv layers.
            feature_dim: Output feature dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        if channels is None:
            channels = [64, 128, 256]
        
        layers = []
        prev_ch = in_channels
        
        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch
        
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], feature_dim),
            nn.ReLU(inplace=True),
        )
        
        self._init_weights()
        logger.info(f"FrequencyStreamCNN initialized: feature_dim={feature_dim}")
    
    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: DCT feature map of shape (B, 1, H, W).
            
        Returns:
            Feature vector of shape (B, feature_dim).
        """
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class SRMFilterLayer(nn.Module):
    """
    Steganalysis Rich Model (SRM) high-pass filters.
    
    Fixed filters proven effective for detecting subtle image manipulations.
    These filters capture noise patterns that differ between real and fake images.
    """
    
    def __init__(self) -> None:
        """Initialize SRM filter bank with 3 high-pass filters."""
        super().__init__()
        
        # Define 3 SRM high-pass filter kernels (5x5)
        # Filter 1: Edge detection
        filter1 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32) / 12.0
        
        # Filter 2: Horizontal/Vertical
        filter2 = np.array([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32) / 4.0
        
        # Filter 3: Diagonal
        filter3 = np.array([
            [-1, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 0, -4, 0, 0],
            [0, 0, 0, 3, 0],
            [0, 0, 0, 0, -1]
        ], dtype=np.float32) / 2.0
        
        # Stack filters: (3, 1, 5, 5) for 3 output channels
        filters = np.stack([filter1, filter2, filter3])[:, np.newaxis, :, :]
        
        # Register as non-trainable buffer
        self.register_buffer("srm_filters", torch.from_numpy(filters))
        
        logger.info("SRMFilterLayer initialized with 3 high-pass filters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SRM filters to input image.
        
        Args:
            x: Input tensor of shape (B, 3, H, W) or (B, 1, H, W).
            
        Returns:
            Filtered output of shape (B, 3, H, W).
        """
        # Convert to grayscale if RGB
        if x.shape[1] == 3:
            weights = torch.tensor(
                [0.299, 0.587, 0.114],
                device=x.device,
                dtype=x.dtype
            ).view(1, 3, 1, 1)
            x = (x * weights).sum(dim=1, keepdim=True)
        
        # Apply SRM filters
        output = F.conv2d(x, self.srm_filters, padding=2)
        return output


class MultiScaleDCTExtractor(nn.Module):
    """
    Multi-scale DCT extraction at multiple block sizes.
    
    Captures artifacts at different spatial frequencies by processing
    DCT at 4x4, 8x8, and 16x16 block sizes.
    """
    
    def __init__(
        self,
        block_sizes: list = None,
        keep_ratio: float = 0.5,
    ) -> None:
        """
        Initialize multi-scale DCT extractor.
        
        Args:
            block_sizes: List of block sizes for DCT.
            keep_ratio: Ratio of high-frequency coefficients to keep.
        """
        super().__init__()
        
        if block_sizes is None:
            block_sizes = [4, 8, 16]
        
        self.block_sizes = block_sizes
        self.extractors = nn.ModuleList([
            DCTExtractor(block_size=bs, keep_ratio=keep_ratio)
            for bs in block_sizes
        ])
        
        logger.info(f"MultiScaleDCTExtractor initialized: block_sizes={block_sizes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale DCT features.
        
        Args:
            x: Input RGB image tensor of shape (B, 3, H, W).
            
        Returns:
            Concatenated multi-scale features of shape (B, num_scales, H, W).
        """
        outputs = []
        target_size = x.shape[2:4]  # (H, W)
        
        for extractor in self.extractors:
            out = extractor(x)  # (B, 1, H', W')
            # Resize to common size if needed
            if out.shape[2:4] != target_size:
                out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
            outputs.append(out)
        
        # Concatenate along channel dimension
        return torch.cat(outputs, dim=1)  # (B, num_scales, H, W)


class EnhancedFrequencyStream(nn.Module):
    """
    Enhanced frequency stream combining SRM filters, multi-scale DCT, and attention.
    
    This provides more robust frequency-based features for generalization.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        dropout: float = 0.3,
        use_srm: bool = True,
        use_multiscale: bool = True,
    ) -> None:
        """
        Initialize enhanced frequency stream.
        
        Args:
            feature_dim: Output feature dimension.
            dropout: Dropout rate.
            use_srm: Whether to use SRM filters.
            use_multiscale: Whether to use multi-scale DCT.
        """
        super().__init__()
        
        self.use_srm = use_srm
        self.use_multiscale = use_multiscale
        
        # Input channels: SRM (3) + Multi-scale DCT (3) or single DCT (1)
        in_channels = 0
        
        if use_srm:
            self.srm = SRMFilterLayer()
            in_channels += 3
        
        if use_multiscale:
            self.dct = MultiScaleDCTExtractor(block_sizes=[4, 8, 16])
            in_channels += 3
        else:
            self.dct = DCTExtractor(block_size=8)
            in_channels += 1
        
        # CNN for processing combined features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        self._init_weights()
        logger.info(
            f"EnhancedFrequencyStream initialized: "
            f"srm={use_srm}, multiscale={use_multiscale}, "
            f"in_channels={in_channels}, feature_dim={feature_dim}"
        )
    
    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract enhanced frequency features.
        
        Args:
            x: Input RGB image tensor of shape (B, 3, H, W).
            
        Returns:
            Feature vector of shape (B, feature_dim).
        """
        features = []
        
        if self.use_srm:
            srm_out = self.srm(x)  # (B, 3, H, W)
            features.append(srm_out)
        
        dct_out = self.dct(x)  # (B, 1 or 3, H, W)
        features.append(dct_out)
        
        # Concatenate all frequency features
        combined = torch.cat(features, dim=1)
        
        # Process through CNN
        x = self.conv_layers(combined)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x

