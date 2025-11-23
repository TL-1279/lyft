# model.py
"""
Simple ResNet50-based regressor that outputs flattened future (x,y) pairs.
This is a minimal example to mirror the notebook baseline.
"""

import torch.nn as nn
import torchvision.models as models

class ResNetTrajectory(nn.Module):
    def __init__(self, future_num_frames: int = 50, pretrained: bool = True):
        super().__init__()
        self.future_num_frames = future_num_frames
        num_targets = future_num_frames * 2  # x,y per future frame

        self.backbone = models.resnet50(pretrained=pretrained)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_targets)

    def forward(self, x):
        # input x expected shape: (B, C, H, W), float32
        return self.backbone(x)
