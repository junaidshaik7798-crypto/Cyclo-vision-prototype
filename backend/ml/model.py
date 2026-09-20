"""
CYCLO-VISION -- Modular Model Definition
=======================================
Defines a small CNN that classifies satellite imagery into cyclone
categories. Architecture is intentionally simple so the app runs
without heavyweight training, but it is *real* PyTorch code that can be
trained or swapped for a pretrained backbone (ResNet/EfficientNet) later.

The model exposes a `forward` returning logits over 7 classes, plus a
`heatmap` method (activation-mapping style saliency for explainable AI).

Classes (index -> label):
    0 No Cyclone
    1 Depression
    2 Deep Depression
    3 Cyclonic Storm
    4 Severe Cyclonic Storm
    5 Very Severe Cyclonic Storm
    6 Extremely Severe Cyclonic Storm
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False


CLASS_LABELS = [
    "No Cyclone",
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
]

# Peak wind-speed (knots) representative of each class (documented prototype
# values, not operational estimates).
CLASS_WIND_KNOTS = [15, 25, 32, 40, 60, 85, 115]
CLASS_PRESSURE_HPA = [1008, 1002, 996, 990, 972, 958, 938]


class CycloCNN(nn.Module):
    """Small 4-block CNN for satellite-image cyclone classification."""

    def __init__(self, in_channels: int = 3, num_classes: int = 7):
        super().__init__()
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed (needed for model mode).")
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: Any) -> Any:
        f = self.features(x)
        g = self.avgpool(f).flatten(1)
        return self.classifier(g)

    def activations(self, x: Any):
        """Return final feature map and pooled vector (for heatmap)."""
        f = self.features(x)
        g = self.avgpool(f).flatten(1)
        logits = self.classifier(g)
        return f, g, logits


def build_model(weights_path: str | None = None, device: str = "cpu"):
    """Instantiate CycloCNN, optionally load state dict from weights_path.

    Returns (model, device). Raises if PyTorch unavailable.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install it or run in demo mode."
        )
    model = CycloCNN()
    state = {"map_location": device}
    if weights_path:
        model.load_state_dict(torch.load(weights_path, **state))
    model.eval()
    return model, device