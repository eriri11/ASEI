"""U-Net and EMA teacher for ASEI."""
import math
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn

def _normalization_layer(
    channels: int,
    normalization: str,
    norm_groups: int,
) -> nn.Module:
    """Build normalization, choosing a GroupNorm count that divides channels."""

    name = normalization.lower().replace("-", "_")
    if name in {"batchnorm", "batch_norm", "bn"}:
        return nn.BatchNorm2d(channels)
    if name in {"groupnorm", "group_norm", "gn"}:
        if norm_groups < 1:
            raise ValueError(f"norm_groups must be positive, got {norm_groups}")
        groups = math.gcd(channels, min(norm_groups, channels))
        return nn.GroupNorm(num_groups=max(1, groups), num_channels=channels)
    if name in {"instancenorm", "instance_norm", "in"}:
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unknown normalization: {normalization}")

class DoubleConv(nn.Module):
    """Two convolution blocks with normalization and ReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str = "batchnorm",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _normalization_layer(out_channels, normalization, norm_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _normalization_layer(out_channels, normalization, norm_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.block(image)

class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str = "batchnorm",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, normalization, norm_groups),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.block(image)

class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str = "batchnorm",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels, normalization, norm_groups)

    def forward(self, decoder_feature: torch.Tensor, skip_feature: torch.Tensor) -> torch.Tensor:
        decoder_feature = self.up(decoder_feature)
        height_delta = skip_feature.size(2) - decoder_feature.size(2)
        width_delta = skip_feature.size(3) - decoder_feature.size(3)
        decoder_feature = F.pad(
            decoder_feature,
            [
                width_delta // 2,
                width_delta - width_delta // 2,
                height_delta // 2,
                height_delta - height_delta // 2,
            ],
        )
        return self.conv(torch.cat([skip_feature, decoder_feature], dim=1))

class UNet2D(nn.Module):
    """2D U-Net for single-channel segmentation."""

    def __init__(self, num_classes: int, in_channels: int=1, base_channels: int=32, normalization: str='batchnorm', norm_groups: int=8) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError(f'base_channels must be >= 4, got {base_channels}')
        channels = base_channels
        self.input_block = DoubleConv(in_channels, channels, normalization, norm_groups)
        self.down1 = Down(channels, channels * 2, normalization, norm_groups)
        self.down2 = Down(channels * 2, channels * 4, normalization, norm_groups)
        self.down3 = Down(channels * 4, channels * 8, normalization, norm_groups)
        self.down4 = Down(channels * 8, channels * 16, normalization, norm_groups)
        self.up1 = Up(channels * 16 + channels * 8, channels * 8, normalization, norm_groups)
        self.up2 = Up(channels * 8 + channels * 4, channels * 4, normalization, norm_groups)
        self.up3 = Up(channels * 4 + channels * 2, channels * 2, normalization, norm_groups)
        self.up4 = Up(channels * 2 + channels, channels, normalization, norm_groups)
        self.head = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.skip_log_gates = nn.Parameter(torch.zeros(4), requires_grad=False)

    @property
    def checkpoint_metadata(self):
        """Normalization is not identifiable from affine tensor shapes alone."""
        layer = self.input_block.block[1]
        if isinstance(layer, nn.GroupNorm):
            return {'model_normalization': 'groupnorm', 'model_norm_group_counts': [m.num_groups for m in self.modules() if isinstance(m, nn.GroupNorm)]}
        if isinstance(layer, nn.InstanceNorm2d):
            return {'model_normalization': 'instancenorm'}
        return {'model_normalization': 'batchnorm'}

    def _encode(self, image: torch.Tensor):
        x1 = self.input_block(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return (x1, x2, x3, x4, x5)

    def _decode(self, encoded_features):
        (x1, x2, x3, x4, x5) = encoded_features
        gates = self.skip_log_gates.clamp(-0.7, 0.7).exp()
        features = self.up1(x5, x4 * gates[0])
        features = self.up2(features, x3 * gates[1])
        features = self.up3(features, x2 * gates[2])
        return self.up4(features, x1 * gates[3])

    def forward(self, image: torch.Tensor):
        return self.head(self._decode(self._encode(image)))

class EMATeacher(nn.Module):
    def __init__(self, student: nn.Module, decay: float = 0.98) -> None:
        super().__init__()
        self.teacher = deepcopy(student)
        self.decay = decay
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        student_state = student.state_dict()
        teacher_state = self.teacher.state_dict()
        for name, value in teacher_state.items():
            source = student_state[name].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)

    def forward(self, image: torch.Tensor):
        self.teacher.eval()
        with torch.no_grad():
            return self.teacher(image)

def build_model(config):
    return UNet2D(num_classes=5,in_channels=1,base_channels=config["model"].get("base_channels",32),normalization=config["model"]["normalization"],norm_groups=config["model"].get("norm_groups",8))
