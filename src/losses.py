import torch
from torch import nn
import torch.nn.functional as F


def class_weights(labels) -> torch.Tensor:
    y = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(y, minlength=2).float().clamp_min(1.0)
    return y.numel() / (2.0 * counts)


class FocalLoss(nn.Module):
    """
    Focal Loss with support for class weights and label smoothing.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Calculate unreduced cross entropy loss
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction='none'
        )
        
        # Calculate pt (probability of the true class)
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()


def build_loss(labels, label_smoothing: float = 0.0, loss_type: str = "ce") -> nn.Module:
    weights = class_weights(labels)
    if loss_type == "focal":
        return FocalLoss(weight=weights, gamma=2.0, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(
        weight=weights, label_smoothing=label_smoothing
    )
