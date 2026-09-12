import torch
from torch import nn


def class_weights(labels) -> torch.Tensor:
    y = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(y, minlength=2).float().clamp_min(1.0)
    return y.numel() / (2.0 * counts)


def build_loss(labels, label_smoothing: float = 0.0) -> nn.Module:
    return nn.CrossEntropyLoss(weight=class_weights(labels), label_smoothing=label_smoothing)
