"""
Neural network architectures for lunar relief classification.
Supports timm backbones (e.g. ConvNeXt, EfficientNet, ResNet) with single-channel
grayscale stem and optional solar azimuth conditioning (none, concat, film).
"""

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import timm


class AzimuthMLP(nn.Module):
    """Encodes continuous [sin(az), cos(az)] coordinates into feature embedding."""
    def __init__(self, in_features: int = 2, hidden_dim: int = 64, out_features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_features),
            nn.SiLU(),
        )

    def forward(self, az_sincos: torch.Tensor) -> torch.Tensor:
        return self.net(az_sincos)


class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Generates per-feature scale (gamma) and shift (beta) from azimuth coordinates:
    FiLM(F) = (1 + gamma) * F + beta
    """
    def __init__(self, in_features: int = 2, feat_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim * 2),
        )
        # Initialize final layer weights and biases to zero so initial modulation is the identity
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, x: torch.Tensor, az_sincos: torch.Tensor) -> torch.Tensor:
        params = self.generator(az_sincos)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return (1.0 + gamma) * x + beta


class PareidoliaModel(nn.Module):
    """
    Unified model architecture for Pareidolia lunar relief classification.

    Args:
        backbone_name: timm model architecture name (e.g. 'convnext_tiny', 'resnet18').
        in_chans: Number of input channels (default 1 for grayscale lunar tiles).
        pretrained: Whether to load ImageNet pre-trained weights.
        conditioning: Azimuth conditioning mode: 'none', 'concat', or 'film'.
        drop_rate: Classifier dropout rate.
        drop_path_rate: Stochastic depth rate for backbones that support it.
        az_dim: Latent dimension for azimuth embedding / FiLM modulation.
    """
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        in_chans: int = 1,
        pretrained: bool = True,
        conditioning: str = "none",
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
        az_dim: int = 64,
    ):
        super().__init__()
        self.conditioning = conditioning.lower()
        if self.conditioning not in ("none", "concat", "film"):
            raise ValueError(f"Unknown conditioning mode '{conditioning}'. Must be 'none', 'concat', or 'film'.")

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.feat_dim = self.backbone.num_features

        if self.conditioning == "none":
            self.az_module = None
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Dropout(drop_rate),
                nn.Linear(self.feat_dim, 1),
            )
        elif self.conditioning == "concat":
            self.az_module = AzimuthMLP(in_features=2, hidden_dim=az_dim, out_features=az_dim)
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feat_dim + az_dim),
                nn.Dropout(drop_rate),
                nn.Linear(self.feat_dim + az_dim, 1),
            )
        elif self.conditioning == "film":
            self.az_module = FiLMModulation(in_features=2, feat_dim=self.feat_dim, hidden_dim=az_dim * 2)
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Dropout(drop_rate),
                nn.Linear(self.feat_dim, 1),
            )

    def forward(self, x: torch.Tensor, az_sincos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input images tensor of shape (B, in_chans, H, W).
            az_sincos: Solar azimuth continuous coordinates [sin(az), cos(az)] of shape (B, 2).
        Returns:
            Logits of shape (B,) representing log-odds for P(Rise).
        """
        feats = self.backbone(x)  # (B, feat_dim)

        if self.conditioning == "none" or az_sincos is None:
            logits = self.classifier(feats)
        elif self.conditioning == "concat":
            az_emb = self.az_module(az_sincos)
            combined = torch.cat([feats, az_emb], dim=-1)
            logits = self.classifier(combined)
        elif self.conditioning == "film":
            mod_feats = self.az_module(feats, az_sincos)
            logits = self.classifier(mod_feats)

        return logits.view(-1)


def build_model(config: Union[Dict[str, Any], Any]) -> PareidoliaModel:
    """Factory helper to build PareidoliaModel from config dict or OmegaConf object."""
    model_cfg = config.get("model", config) if hasattr(config, "get") else config.model
    training_cfg = config.get("training", {}) if hasattr(config, "get") else getattr(config, "training", {})

    backbone_name = model_cfg.get("backbone", "convnext_tiny")
    in_chans = model_cfg.get("in_chans", 1)
    pretrained = model_cfg.get("pretrained", True)
    conditioning = model_cfg.get("conditioning", "none")
    drop_path_rate = training_cfg.get("drop_path_rate", 0.1)

    return PareidoliaModel(
        backbone_name=backbone_name,
        in_chans=in_chans,
        pretrained=pretrained,
        conditioning=conditioning,
        drop_path_rate=drop_path_rate,
    )
