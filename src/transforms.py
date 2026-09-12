from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RawTransform:
    """Raw-frame tensor preprocessing without geometric augmentation."""

    mean: float
    std: float
    training: bool = False
    brightness: float = 0.0
    contrast: float = 0.0
    noise_std: float = 0.0

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Convert a uint8 HxW image to a normalized 1xHxW float tensor."""
        x = image.astype(np.float32) / 255.0
        if self.training:
            if self.contrast > 0:
                scale = np.random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
                x = (x - x.mean()) * scale + x.mean()
            if self.brightness > 0:
                x = x + np.random.uniform(-self.brightness, self.brightness)
            if self.noise_std > 0:
                x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
            x = np.clip(x, 0.0, 1.0)
        x = (x - float(self.mean)) / max(float(self.std), 1e-6)
        return torch.from_numpy(x).unsqueeze(0).float()


def build_transform(cfg, training: bool) -> RawTransform:
    frozen = cfg.get("frozen", {})
    norm = frozen.get("norm", {})
    aug = cfg.get("augmentation", {})
    return RawTransform(
        mean=float(norm.get("mean", 0.0)),
        std=float(norm.get("std", 1.0)),
        training=training,
        brightness=float(aug.get("brightness", 0.0)),
        contrast=float(aug.get("contrast", 0.0)),
        noise_std=float(aug.get("noise_std", 0.0)),
    )
