import torch.nn as nn
import torch.nn.functional as F
import torch

class MichCnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # [B, 16, 144, 256]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # [B, 16, 72, 128]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 72, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # [B, 32, 36, 64]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 36, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # [B, 64, 18, 32]
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 18 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten before FC
        x = self.classifier(x)
        return x

