import torch
from torch import nn
from torchvision import models

class BaselineFusionModel(nn.Module):
    """Simple fusion model for CXR + Tabular data"""
    def __init__(self, tab_dim, num_classes=2):
        super().__init__()
        # CNN backbone for image
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()  # remove final classifier

        # Tabular MLP
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(512 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, tab):
        img_feat = self.cnn(img)
        tab_feat = self.tab_mlp(tab)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        out = self.fusion_mlp(combined)
        return out