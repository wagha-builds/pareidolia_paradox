"""Visualization and explainability tools for Pareidolia Paradox.

Features:
- GradCAM and HiResCAM implementation supporting pure PyTorch and FiLM backbones.
- Canonical and original-frame heatmap visualization via decanonicalize().
- Quantitative shadow-mass fraction computation (CAM mass inside Otsu shadow mask).
- Diagnostic grid generation for correct vs. misclassified validation samples.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .canonical import decanonicalize


class GradCAM:
    """Class Activation Mapping (Grad-CAM and HiResCAM) for CNN backbones.

    Supports both standard backbones and FiLM-conditioned wrappers (FiLMBackbone).
    Automatically attaches to the final convolutional stage/layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.model.eval()

        if target_layer is None:
            target_layer = self._find_target_layer(model)

        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        """Find the last convolutional stage or residual layer."""
        m = model.backbone if hasattr(model, "backbone") else model
        if hasattr(m, "stages"):
            # ConvNeXt stage
            return m.stages[-1]
        elif hasattr(m, "layer4"):
            # ResNet stage
            return m.layer4
        elif hasattr(m, "blocks"):
            # Vision Transformer / ConvNeXt V2 block
            return m.blocks[-1]
        else:
            # Fallback to last child module
            return list(m.children())[-1]

    def _register_hooks(self) -> None:
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def __call__(
        self,
        x: torch.Tensor,
        az_sincos: torch.Tensor | None = None,
        target_class: int | None = None,
        hires: bool = False,
    ) -> tuple[np.ndarray, int, float]:
        """Compute Grad-CAM for a single input.

        Args:
            x:            input tensor [1, C, H, W]
            az_sincos:    optional azimuth encoding [1, 2] for FiLM models
            target_class: class index (0 for Depth, 1 for Rise), or None for predicted class
            hires:        if True, use HiResCAM elementwise product

        Returns:
            cam:          float32 ndarray [H, W] normalized to [0, 1]
            pred_class:   predicted class index (0 or 1)
            prob_rise:    predicted probability of class 1 (Rise)
        """
        self.model.zero_grad()

        # Forward pass
        if az_sincos is not None and hasattr(self.model, "film_mlp"):
            logits = self.model(x, az_sincos)
        else:
            logits = self.model(x)

        probs = torch.softmax(logits, dim=1)
        prob_rise = float(probs[0, 1].item())
        pred_class = int(torch.argmax(logits, dim=1).item())

        if target_class is None:
            target_class = pred_class

        # Backward pass on target class logit
        loss = logits[0, target_class]
        loss.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks failed to capture gradients or activations.")

        grads = self.gradients.detach()
        acts = self.activations.detach()

        if hires:
            # HiResCAM: elementwise product before spatial pooling
            cam_map = torch.relu(grads * acts).sum(dim=1, keepdim=True)
        else:
            # Grad-CAM: global average pool gradients as channel weights
            weights = grads.mean(dim=(-2, -1), keepdim=True)
            cam_map = torch.relu((weights * acts).sum(dim=1, keepdim=True))

        # Bilinear upsample to input resolution
        cam_upsampled = F.interpolate(
            cam_map,
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam_upsampled[0, 0].detach().cpu().numpy()

        # Normalize to [0, 1]
        c_min, c_max = float(cam.min()), float(cam.max())
        if c_max - c_min > 1e-7:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32), pred_class, prob_rise


def compute_otsu_shadow_mask(img_uint8: np.ndarray) -> np.ndarray:
    """Compute binary mask of dark shadow pixels using Otsu thresholding."""
    if img_uint8.ndim == 3:
        img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_uint8.copy()

    val, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Shadows are pixels at or below the Otsu threshold
    return img_gray <= val


def compute_shadow_mass_fraction(cam: np.ndarray, shadow_mask: np.ndarray) -> float:
    """Calculate the fraction of CAM activation mass located inside the shadow mask."""
    total_mass = float(np.sum(cam))
    if total_mass <= 1e-7:
        return 0.0
    shadow_mass = float(np.sum(cam[shadow_mask]))
    return float(shadow_mass / total_mass)


def overlay_cam_on_image(
    img_uint8: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.40,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """Overlay CAM heatmap onto grayscale uint8 image, returning RGB [H, W, 3]."""
    heatmap = (np.clip(cam, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if img_uint8.ndim == 2:
        base_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = img_uint8.copy()

    overlay = alpha * heatmap_rgb.astype(np.float32) + (1.0 - alpha) * base_rgb.astype(
        np.float32
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def decanonicalize_cam(
    cam: np.ndarray, azimuth: float, delta: float, s: int
) -> np.ndarray:
    """Map canonical-frame CAM heatmap back to the original raw image frame."""
    cam_uint8 = (np.clip(cam, 0.0, 1.0) * 255.0).astype(np.uint8)
    decanon = decanonicalize(cam_uint8, azimuth, delta, s)
    return decanon.astype(np.float32) / 255.0


def generate_explainability_grid(
    samples: list[dict],
    output_path: str | Path,
    title: str = "Grad-CAM Explainability Grid",
    max_samples: int = 15,
) -> None:
    """Generate a multi-panel visual grid of explainability maps.

    Columns per sample:
    1. Raw Image
    2. Canonical Frame (Sun at Top)
    3. Canonical Grad-CAM Heatmap
    4. Decanonicalized CAM Overlay on Raw Image
    """
    n = min(len(samples), max_samples)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(16, 3.8 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.995)

    headers = [
        "Raw Image",
        "Canonical Frame (Sun @ Top)",
        "Canonical Grad-CAM Heatmap",
        "Decanonicalized CAM Overlay",
    ]

    for col, h in enumerate(headers):
        axes[0, col].set_title(h, fontsize=12, fontweight="bold", pad=10)

    class_names = {0: "Depth", 1: "Rise"}

    for i in range(n):
        s = samples[i]
        raw_img = s["raw_img"]
        canon_img = s["canon_img"]
        cam_canon = s["cam_canon"]
        overlay_raw = s["overlay_raw"]
        y_true = s["true_label"]
        y_pred = s["pred_label"]
        p_rise = s["p_rise"]
        smf = s["shadow_mass_fraction"]
        az = s["azimuth"]

        # Display images
        axes[i, 0].imshow(raw_img, cmap="gray")
        axes[i, 0].set_ylabel(
            f"ID: {s['image_id']}\nTrue: {class_names[y_true]} | Pred: {class_names[y_pred]}\n"
            f"P(Rise)={p_rise:.2f} | SMF={smf:.1%}\nAz={az:.1f}°",
            fontsize=10,
            fontweight="semibold",
            rotation=0,
            labelpad=110,
            va="center",
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(canon_img, cmap="gray")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(cam_canon, cmap="viridis", vmin=0, vmax=1)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay_raw)
        axes[i, 3].axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved explainability grid to {output_path}")
