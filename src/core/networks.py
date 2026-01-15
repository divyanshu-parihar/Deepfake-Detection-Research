import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FrequencyStream(nn.Module):
    """
    Lightweight CNN to process the DCT maps.
    Input: (B, 1, 224, 224)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        # Using LazyLinear to handle variable input sizes automatically
        self.fc = nn.LazyLinear(512) 

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class RGBStream(nn.Module):
    """
    Semantic feature extractor.
    Default: Lightweight Custom CNN for CPU/Demo speed.
    """
    def __init__(self, mode='lightweight'):
        super().__init__()
        self.mode = mode
        
        if mode == 'production':
            # EfficientNet-B4 (Requires downloading weights)
            self.backbone = models.efficientnet_b4(weights='DEFAULT')
            self.backbone.classifier = nn.Identity()
            self.out_dim = 1792
        else:
            # Lightweight 3-layer CNN
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.LazyLinear(512)
            self.out_dim = 512

    def forward(self, x):
        if self.mode == 'production':
            return self.backbone(x)
        else:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

class DualStreamNetwork(nn.Module):
    def __init__(self, mode='lightweight'):
        super().__init__()
        self.rgb_stream = RGBStream(mode=mode)
        self.freq_stream = FrequencyStream()
        
        # Calculate fusion dimension
        rgb_dim = self.rgb_stream.out_dim
        freq_dim = 512
        
        self.fusion = nn.Linear(rgb_dim + freq_dim, 128)
        self.classifier = nn.Linear(128, 2) # Real vs Fake

    def forward(self, x_rgb, x_freq):
        feat_rgb = self.rgb_stream(x_rgb)
        feat_freq = self.freq_stream(x_freq)
        
        # Late Fusion
        combined = torch.cat((feat_rgb, feat_freq), dim=1)
        embedding = F.relu(self.fusion(combined))
        
        # Classification
        out = self.classifier(embedding)
        return out, embedding
