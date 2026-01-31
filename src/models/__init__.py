"""Models package for deepfake detection."""
from .dct_extractor import (
    DCTExtractor, 
    FrequencyStreamCNN,
    SRMFilterLayer,
    MultiScaleDCTExtractor,
    EnhancedFrequencyStream,
)
from .dual_stream import (
    DualStreamDetector,
    EnhancedDualStreamDetector,
    CBAM,
)
from .losses import (
    ContrastiveLoss,
    CombinedLoss,
    FocalLoss,
    EnhancedCombinedLoss,
)

__all__ = [
    "DCTExtractor",
    "FrequencyStreamCNN",
    "SRMFilterLayer",
    "MultiScaleDCTExtractor",
    "EnhancedFrequencyStream",
    "DualStreamDetector",
    "EnhancedDualStreamDetector",
    "CBAM",
    "ContrastiveLoss",
    "CombinedLoss",
    "FocalLoss",
    "EnhancedCombinedLoss",
]
