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
import torch.nn.functional as F
from torch import nn

# ─── Physics Tensor ────────────────────────────────────────────────────────────
# Pre-computed Sobel kernels (registered as buffers, not parameters).
# Normalised by 4 so outputs live in roughly the same range as the input pixel.
_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3) / 4.0  # shape: (1,1,3,3)

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3) / 4.0  # shape: (1,1,3,3)


def build_physics_tensor(gray: torch.Tensor) -> torch.Tensor:
    """Convert 1-channel grayscale to 3-channel physics tensor on-GPU.

    Physics encoding:
      Ch0 : original grayscale I(x,y)      — luminosity
      Ch1 : vertical Sobel   dI/dy         — shadow slope along solar vector
      Ch2 : horizontal Sobel dI/dx         — rim curvature / bilateral symmetry

    Args:
        gray: [B, 1, H, W] float32 tensor (normalised, typically in [-1, 1]).

    Returns:
        [B, 3, H, W] float32 tensor.
    """
    dev = gray.device
    ky = _SOBEL_Y.to(dev)
    kx = _SOBEL_X.to(dev)
    gy = F.conv2d(gray, ky, padding=1)   # shadow slope
    gx = F.conv2d(gray, kx, padding=1)   # rim curvature
    return torch.cat([gray, gy, gx], dim=1)  # [B, 3, H, W]


class PhysicsTensorWrapper(nn.Module):
    """Wrap a 3-channel backbone so it accepts 1-ch grayscale at the interface.

    At forward time the 1→3 channel expansion is done on-GPU via
    ``build_physics_tensor``.  The wrapped backbone itself is responsible
    for all its own parameters; this module adds zero parameters.

    This is purely an input-channel adapter and does NOT replace
    ``FiLMBackbone`` — it is composed with it.  Typical usage::

        base_3ch = PhysicsTensorWrapper(timm_backbone)
        model    = FiLMBackbone(base_3ch, feature_dim, ...)

    Args:
        backbone: any nn.Module whose first conv expects 3-channel RGB input.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 1-ch → 3-ch physics tensor, then run backbone feature extractor."""
        x3 = build_physics_tensor(x)  # [B, 1, H, W] → [B, 3, H, W]
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x3)
        return self.backbone(x3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = build_physics_tensor(x)
        return self.backbone(x3)


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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[:-1](x)

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


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for domain-adversarial / shortcut-debiasing training."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


class AdversarialBackbone(nn.Module):
    """Backbone with adversarial azimuth-quadrant head for shortcut removal.

    Forces backbone representations to be invariant to azimuth quadrant
    while optimizing the primary relief classification head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 2,
        num_az_classes: int = 4,
        adv_lambda: float = 0.5,
        film_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.adv_lambda = adv_lambda

        # FiLM modulation for classification branch
        self.film_mlp = nn.Sequential(
            nn.Linear(2, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, 2 * feature_dim),
        )
        self.head = nn.Linear(feature_dim, num_classes)

        # Adversarial azimuth quadrant head (4 quadrants: 0-90, 90-180, 180-270, 270-360)
        self.az_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_az_classes),
        )

        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        az_sincos: torch.Tensor,
        return_az_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)

        if feats.dim() == 4:
            feats = feats.mean(dim=(-2, -1))
        elif feats.dim() == 3:
            feats = feats[:, 0]

        gamma_beta = self.film_mlp(az_sincos)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        modulated = feats * (1.0 + gamma) + beta
        main_logits = self.head(modulated)

        if return_az_logits:
            reversed_feats = GradientReversal.apply(feats, self.adv_lambda)
            az_logits = self.az_head(reversed_feats)
            return main_logits, az_logits

        return main_logits


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
        nn.Module — either a plain backbone, FiLMBackbone, or AdversarialBackbone
        wrapper, optionally behind a PhysicsTensorWrapper for 3-ch Sobel input.
    """
    model_cfg = cfg.get("model", {})
    backbone_name = str(model_cfg.get("backbone", "small_cnn"))
    # physics_tensor: true → wrapper converts 1-ch input to 3-ch (I, dI/dy, dI/dx)
    # The backbone is always created with in_chans=3 when physics_tensor is set.
    use_physics_tensor = bool(model_cfg.get("physics_tensor", False))
    in_chans = 3 if use_physics_tensor else int(model_cfg.get("in_chans", 1))
    pretrained = bool(model_cfg.get("pretrained", False))
    drop_path = float(model_cfg.get("drop_path_rate", 0.0))
    conditioning = str(model_cfg.get("conditioning", "none")).lower()

    extra_kwargs = {}
    if "img_size" in model_cfg:
        extra_kwargs["img_size"] = int(model_cfg["img_size"])

    if backbone_name == "small_cnn":
        base = SmallCNN(in_chans=in_chans)
    else:
        try:
            import timm

            base = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0 if conditioning in ("film", "adversarial") else 2,
                drop_path_rate=drop_path,
                **extra_kwargs,
            )
        except Exception as exc:
            if pretrained:
                raise
            print(
                f"Falling back to SmallCNN because timm model {backbone_name!r} failed: {exc}"
            )
            base = SmallCNN(in_chans=in_chans)

    # Wrap with physics-tensor adapter so the public interface always takes 1-ch.
    # The wrapper converts 1-ch → 3-ch internally on GPU, then calls the backbone.
    if use_physics_tensor:
        base = PhysicsTensorWrapper(base)
        # After wrapping, feature dim is probed using 1-ch dummy (wrapper handles the rest).
        _probe_chans = 1
    else:
        _probe_chans = in_chans

    if conditioning == "film":
        feature_dim = _get_feature_dim(base, in_chans=_probe_chans)
        film_hidden = int(model_cfg.get("film_hidden", 64))
        return FiLMBackbone(
            base, feature_dim=feature_dim, num_classes=2, film_hidden=film_hidden
        )
    elif conditioning == "adversarial":
        feature_dim = _get_feature_dim(base, in_chans=_probe_chans)
        film_hidden = int(model_cfg.get("film_hidden", 64))
        adv_lambda = float(model_cfg.get("adv_lambda", 0.5))
        return AdversarialBackbone(
            base,
            feature_dim=feature_dim,
            num_classes=2,
            num_az_classes=4,
            adv_lambda=adv_lambda,
            film_hidden=film_hidden,
        )

    return base


def predict_proba(
    model: nn.Module, images: torch.Tensor, az_sincos: torch.Tensor | None = None
) -> torch.Tensor:
    """Run model forward pass and return P(Rise) probability.

    Handles plain backbone, FiLMBackbone, and AdversarialBackbone.

    Args:
        model:     nn.Module
        images:    [B, C, H, W] float tensor
        az_sincos: [B, 2] float tensor (required for FiLMBackbone/AdversarialBackbone)

    Returns:
        [B] float tensor of P(Rise) probabilities.
    """
    if isinstance(model, (FiLMBackbone, AdversarialBackbone)):
        if az_sincos is None:
            raise ValueError("Model requires az_sincos")
        logits = model(images, az_sincos)
    else:
        logits = model(images)
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.softmax(logits, dim=1)[:, 1]
