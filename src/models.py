import torch
from torch import nn


class SmallCNN(nn.Module):
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

    def forward(self, x):
        return self.net(x)


def build_model(cfg) -> nn.Module:
    model_cfg = cfg.get("model", {})
    backbone = str(model_cfg.get("backbone", "small_cnn"))
    in_chans = int(model_cfg.get("in_chans", 1))
    pretrained = bool(model_cfg.get("pretrained", False))
    drop_path = float(model_cfg.get("drop_path_rate", 0.0))

    if backbone == "small_cnn":
        return SmallCNN(in_chans=in_chans)

    try:
        import timm

        return timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=2,
            drop_path_rate=drop_path,
        )
    except Exception as exc:
        if pretrained:
            raise
        print(f"Falling back to SmallCNN because timm model {backbone!r} failed: {exc}")
        return SmallCNN(in_chans=in_chans)


def predict_proba(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    logits = model(images)
    return torch.softmax(logits, dim=1)[:, 1]
