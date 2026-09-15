"""
src/models.py — Model factory and FiLM conditioning wrapper.

FiLM (Feature-wise Linear Modulation):
  Injects azimuth signal (sin, cos) into the backbone feature maps via
  learned gamma/beta scaling. This gives the model a residual azimuth signal
  even after canonicalization — useful when canonicalization is imperfect
  (real images have noise; calibration has δ error ~0.55°).

  Reference: Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer" (2018).

Usage in configs:
  model.conditioning: "none"   → raw backbone, no azimuth input
  model.conditioning: "film"   → FiLMConvNeXt wrapper (azimuth as sin/cos input)
"""

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Lightweight debug CNN for smoke tests (no timm dependency)."""

    def __init__(self, in_chans: int = 1, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMBackbone(nn.Module):
    """FiLM-conditioned backbone.

    Wraps any timm backbone (or SmallCNN) with Feature-wise Linear Modulation
    that injects the sun azimuth (as sin, cos) into the global average-pooled
    feature vector before the classification head.

    Architecture:
        backbone → global avg pool → features [B, D]
        azimuth (sin, cos) [B, 2] → MLP → gamma [B, D], beta [B, D]
        modulated = features * (1 + gamma) + beta
        → head → logits [B, num_classes]

    Args:
        backbone:     backbone nn.Module with .forward_features() or __call__.
        feature_dim:  dimension of the backbone's output feature vector.
        num_classes:  output classes (default 2).
        film_hidden:  hidden layer size of the FiLM MLP (default 64).
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 2,
        film_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

        # FiLM MLP: azimuth (sin, cos) → gamma + beta
        self.film_mlp = nn.Sequential(
            nn.Linear(2, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, 2 * feature_dim),  # gamma and beta concatenated
        )
        # Classification head
        self.head = nn.Linear(feature_dim, num_classes)

        # Initialise gamma near 0 so the early training is close to baseline
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

    def forward(self, x: torch.Tensor, az_sincos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          image tensor [B, C, H, W]
            az_sincos:  azimuth encoding [B, 2] (sin, cos)

        Returns:
            logits [B, num_classes]
        """
        # Extract features
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)  # [B, D, h, w] or [B, D]
        else:
            feats = self.backbone(x)

        # Global average pool if needed
        if feats.dim() == 4:
            feats = feats.mean(dim=(-2, -1))  # [B, D]
        elif feats.dim() == 3:
            feats = feats[:, 0]  # CLS token for ViT-style backbones

        # FiLM modulation
        gamma_beta = self.film_mlp(az_sincos)  # [B, 2*D]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        feats = feats * (1.0 + gamma) + beta

        return self.head(feats)


def _get_feature_dim(model: nn.Module, in_chans: int = 1) -> int:
    """Probe the backbone for its output feature dimension."""
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, in_chans, 256, 256)
        if hasattr(model, "forward_features"):
            out = model.forward_features(dummy)
        else:
            out = model(dummy)
        if out.dim() == 4:
            return out.shape[1]
        elif out.dim() == 3:
            return out.shape[2]
        else:
            return out.shape[1]


def build_model(cfg: dict) -> nn.Module:
    """Build a model from the config dict.

    Args:
        cfg: full config dict (OmegaConf-resolved to plain dict).

    Returns:
        nn.Module — either a plain backbone or a FiLMBackbone wrapper.
    """
    model_cfg = cfg.get("model", {})
    backbone = str(model_cfg.get("backbone", "small_cnn"))
    in_chans = int(model_cfg.get("in_chans", 1))
    pretrained = bool(model_cfg.get("pretrained", False))
    drop_path = float(model_cfg.get("drop_path_rate", 0.0))
    conditioning = str(model_cfg.get("conditioning", "none")).lower()

    extra_kwargs = {}
    if "img_size" in model_cfg:
        extra_kwargs["img_size"] = int(model_cfg["img_size"])

    if backbone == "small_cnn":
        base = SmallCNN(in_chans=in_chans)
    else:
        try:
            import timm

            base = timm.create_model(
                backbone,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0 if conditioning == "film" else 2,
                drop_path_rate=drop_path,
                **extra_kwargs,
            )
        except Exception as exc:
            if pretrained:
                raise
            print(
                f"Falling back to SmallCNN because timm model {backbone!r} failed: {exc}"
            )
            base = SmallCNN(in_chans=in_chans)

    if conditioning == "film":
        feature_dim = _get_feature_dim(base, in_chans=in_chans)
        film_hidden = int(model_cfg.get("film_hidden", 64))
        return FiLMBackbone(
            base, feature_dim=feature_dim, num_classes=2, film_hidden=film_hidden
        )

    return base


def predict_proba(
    model: nn.Module, images: torch.Tensor, az_sincos: torch.Tensor | None = None
) -> torch.Tensor:
    """Run model forward pass and return P(Rise) probability.

    Handles both plain backbone (images only) and FiLMBackbone (images + az_sincos).

    Args:
        model:     nn.Module
        images:    [B, C, H, W] float tensor
        az_sincos: [B, 2] float tensor (required for FiLMBackbone, ignored otherwise)

    Returns:
        [B] float tensor of P(Rise) probabilities.
    """
    if isinstance(model, FiLMBackbone):
        if az_sincos is None:
            raise ValueError("FiLMBackbone requires az_sincos")
        logits = model(images, az_sincos)
    else:
        logits = model(images)
    return torch.softmax(logits, dim=1)[:, 1]
