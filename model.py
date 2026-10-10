import itertools
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import load_torch_file


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.add_module(
            "c",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.zeros_(self.bn.bias)


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.m = module

    def forward(self, x: Tensor) -> Tensor:
        return x + self.m(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden_channels = max(1, int(channels * ratio))
        self.conv_reduce = nn.Conv2d(channels, hidden_channels, 1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.conv_expand(self.act1(self.conv_reduce(scale))).sigmoid()
        return x * scale


class FeedForward(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pw1 = ConvBN(channels, channels * 2)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = ConvBN(channels * 2, channels, bn_weight_init=0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.pw2(self.act(self.pw1(x)))


class RepVGGDepthwise(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = ConvBN(channels, channels, 3, 1, 1, groups=channels)
        self.conv1 = ConvBN(channels, channels, 1, 1, 0, groups=channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x) + self.conv1(x) + x


class SmallKernelAggregation(nn.Module):
    """Grouped dynamic local aggregation implemented with native PyTorch."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("The SKA kernel size must be odd")
        self.kernel_size = int(kernel_size)

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        positions = self.kernel_size**2
        if weights.ndim != 5:
            raise ValueError("Dynamic weights must be a five-dimensional tensor")
        weight_channels = weights.shape[1]
        if weight_channels < 1 or channels % weight_channels != 0:
            raise ValueError("The input channels must be divisible by weight channels")
        expected = (batch_size, weight_channels, positions, height, width)
        if tuple(weights.shape) != expected:
            raise ValueError(
                "Dynamic weights must have shape [B, C/G, K*K, H, W]; "
                "received {}".format(tuple(weights.shape))
            )
        channel_indices = torch.arange(channels, device=x.device) % weight_channels
        expanded_weights = weights[:, channel_indices]
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        ).view(batch_size, channels, positions, height, width)
        return (patches * expanded_weights).sum(dim=2)


class MultiScaleLargeKernelPerception(nn.Module):
    def __init__(
        self,
        channels: int,
        small_context_kernel: int,
        large_context_kernel: int,
        ska_kernel: int,
        groups: int,
        context_mode: str = "adaptive",
    ) -> None:
        super().__init__()
        if channels % 2 != 0 or channels % groups != 0:
            raise ValueError("AdaLSConv channels must be even and divisible by groups")
        if small_context_kernel % 2 == 0 or large_context_kernel % 2 == 0:
            raise ValueError("Contextual kernels must be odd")
        if context_mode not in {"adaptive", "fixed", "small", "large"}:
            raise ValueError("context_mode must be adaptive, fixed, small, or large")
        compressed_channels = channels // 2
        self.cv1 = ConvBN(channels, compressed_channels)
        self.act = nn.ReLU(inplace=True)
        self.dw_small = ConvBN(
            compressed_channels,
            compressed_channels,
            small_context_kernel,
            1,
            small_context_kernel // 2,
            groups=compressed_channels,
        )
        self.dw_large = ConvBN(
            compressed_channels,
            compressed_channels,
            large_context_kernel,
            1,
            large_context_kernel // 2,
            groups=compressed_channels,
        )
        self.scale_logit = nn.Parameter(torch.zeros(()))
        self.cv3 = ConvBN(compressed_channels, compressed_channels)
        dynamic_channels = ska_kernel**2 * channels // groups
        self.cv4 = nn.Conv2d(compressed_channels, dynamic_channels, 1)
        self.norm = nn.GroupNorm(channels // groups, dynamic_channels)
        self.channels = int(channels)
        self.ska_kernel = int(ska_kernel)
        self.groups = int(groups)
        self.context_mode = context_mode

    @property
    def alpha(self) -> Tensor:
        return self.scale_logit.sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        compressed = self.act(self.cv1(x))
        if self.context_mode == "small":
            fused = self.dw_small(compressed)
        elif self.context_mode == "large":
            fused = self.dw_large(compressed)
        else:
            small = self.dw_small(compressed)
            large = self.dw_large(compressed)
            if self.context_mode == "fixed":
                fused = 0.5 * (small + large)
            else:
                fused = self.alpha * small + (1.0 - self.alpha) * large
        fused = self.act(self.cv3(fused))
        weights = self.norm(self.cv4(fused))
        batch_size, _, height, width = weights.shape
        return weights.view(
            batch_size,
            self.channels // self.groups,
            self.ska_kernel**2,
            height,
            width,
        )


class AdaLSConv(nn.Module):
    def __init__(
        self,
        channels: int,
        small_context_kernel: int = 7,
        large_context_kernel: int = 11,
        ska_kernel: int = 3,
        groups: int = 8,
        context_mode: str = "adaptive",
    ) -> None:
        super().__init__()
        self.lkp = MultiScaleLargeKernelPerception(
            channels,
            small_context_kernel,
            large_context_kernel,
            ska_kernel,
            groups,
            context_mode,
        )
        self.ska = SmallKernelAggregation(ska_kernel)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.ska(x, self.lkp(x))) + x


class Attention(nn.Module):
    def __init__(
        self,
        channels: int,
        key_dim: int,
        num_heads: int,
        attention_ratio: float,
        resolution: int,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.scale = key_dim**-0.5
        self.query_key_channels = key_dim * num_heads
        self.value_dim = int(attention_ratio * key_dim)
        self.value_channels = self.value_dim * num_heads
        qkv_channels = 2 * self.query_key_channels + self.value_channels
        self.qkv = ConvBN(channels, qkv_channels)
        self.proj = nn.Sequential(
            nn.ReLU(inplace=True),
            ConvBN(self.value_channels, channels, bn_weight_init=0.0),
        )
        self.dw = ConvBN(
            self.query_key_channels,
            self.query_key_channels,
            3,
            1,
            1,
            groups=self.query_key_channels,
        )
        points = list(itertools.product(range(resolution), range(resolution)))
        offset_to_index: Dict[Tuple[int, int], int] = {}
        indices: List[int] = []
        for first in points:
            for second in points:
                offset = (abs(first[0] - second[0]), abs(first[1] - second[1]))
                if offset not in offset_to_index:
                    offset_to_index[offset] = len(offset_to_index)
                indices.append(offset_to_index[offset])
        token_count = len(points)
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(offset_to_index))
        )
        self.register_buffer(
            "attention_bias_idxs",
            torch.tensor(indices, dtype=torch.long).view(token_count, token_count),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape
        token_count = height * width
        if self.attention_bias_idxs.shape != (token_count, token_count):
            raise ValueError(
                "The input resolution does not match the configured attention resolution"
            )
        qkv = self.qkv(x)
        query, key, value = qkv.split(
            [self.query_key_channels, self.query_key_channels, self.value_channels],
            dim=1,
        )
        query = self.dw(query)
        query = query.view(batch_size, self.num_heads, self.key_dim, token_count)
        key = key.view(batch_size, self.num_heads, self.key_dim, token_count)
        value = value.view(batch_size, self.num_heads, self.value_dim, token_count)
        bias = self.attention_biases[:, self.attention_bias_idxs]
        attention = ((query.transpose(-2, -1) @ key) * self.scale + bias).softmax(
            dim=-1
        )
        output = (value @ attention.transpose(-2, -1)).reshape(
            batch_size, self.value_channels, height, width
        )
        return self.proj(output)


class FeatureBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        key_dim: int,
        num_heads: int,
        attention_ratio: float,
        resolution: int,
        stage_index: int,
        depth_index: int,
        small_context_kernel: int,
        large_context_kernel: int,
        ska_kernel: int,
        ska_groups: int,
        context_mode: str,
        stage4_block: str,
    ) -> None:
        super().__init__()
        if depth_index % 2 == 0:
            self.mixer = RepVGGDepthwise(channels)
            self.se = SqueezeExcite(channels, 0.25)
        else:
            self.se = nn.Identity()
            if stage_index == 3 and stage4_block == "msa":
                self.mixer = Residual(
                    Attention(
                        channels,
                        key_dim,
                        num_heads,
                        attention_ratio,
                        resolution,
                    )
                )
            else:
                self.mixer = AdaLSConv(
                    channels,
                    small_context_kernel,
                    large_context_kernel,
                    ska_kernel,
                    ska_groups,
                    context_mode,
                )
        self.ffn = Residual(FeedForward(channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(self.se(self.mixer(x)))


class CGSFNet(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        num_classes: int = 180,
        embedding_dim: int = 512,
        channels: Iterable[int] = (64, 128, 256, 384),
        depths: Iterable[int] = (0, 2, 8, 10),
        key_dims: Iterable[int] = (16, 16, 16, 16),
        num_heads: Iterable[int] = (3, 3, 3, 4),
        small_context_kernel: int = 7,
        large_context_kernel: int = 11,
        ska_kernel: int = 3,
        ska_groups: int = 8,
        context_mode: str = "adaptive",
        stage4_block: str = "msa",
    ) -> None:
        super().__init__()
        channels = list(channels)
        depths = list(depths)
        key_dims = list(key_dims)
        num_heads = list(num_heads)
        if len(channels) != 4 or not (
            len(depths) == len(key_dims) == len(num_heads) == 4
        ):
            raise ValueError("CGSFNet requires four stages")
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8")
        if stage4_block not in {"msa", "adals"}:
            raise ValueError("stage4_block must be msa or adals")
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)

        self.patch_embed = nn.Sequential(
            ConvBN(3, channels[0] // 4, 3, 2, 1),
            nn.ReLU(inplace=True),
            ConvBN(channels[0] // 4, channels[0] // 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            ConvBN(channels[0] // 2, channels[0], 3, 2, 1),
        )
        resolution = image_size // 8
        stages: List[nn.Sequential] = []
        for stage_index in range(4):
            modules: List[nn.Module] = []
            if stage_index > 0:
                previous_channels = channels[stage_index - 1]
                modules.extend(
                    [
                        ConvBN(
                            previous_channels,
                            previous_channels,
                            3,
                            2,
                            1,
                            groups=previous_channels,
                        ),
                        ConvBN(previous_channels, channels[stage_index]),
                    ]
                )
                resolution = (resolution - 1) // 2 + 1
            attention_ratio = channels[stage_index] / (
                key_dims[stage_index] * num_heads[stage_index]
            )
            for depth_index in range(depths[stage_index]):
                modules.append(
                    FeatureBlock(
                        channels[stage_index],
                        key_dims[stage_index],
                        num_heads[stage_index],
                        attention_ratio,
                        resolution,
                        stage_index,
                        depth_index,
                        small_context_kernel,
                        large_context_kernel,
                        ska_kernel,
                        ska_groups,
                        context_mode,
                        stage4_block,
                    )
                )
            stages.append(nn.Sequential(*modules))
        self.blocks1, self.blocks2, self.blocks3, self.blocks4 = stages
        self.identity_projection = nn.Linear(channels[-1], embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, FeedForward):
                nn.init.zeros_(module.pw2.bn.weight)
            elif isinstance(module, Attention):
                nn.init.zeros_(module.proj[1].bn.weight)

    def forward_features(self, x: Tensor, normalize: bool = True) -> Tensor:
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.identity_projection(x)
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x: Tensor, return_features: bool = False):
        features = self.forward_features(x, normalize=True)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def _adals_prefixes(self) -> Tuple[str, ...]:
        prefixes = []
        for name, module in self.named_modules():
            if isinstance(module, AdaLSConv):
                prefixes.append(name + ".")
        return tuple(prefixes)

    def load_compatible_lsnet_t_weights(
        self, checkpoint: Union[str, Path, Mapping[str, Tensor]]
    ) -> Dict[str, List[str]]:
        if isinstance(checkpoint, (str, Path)):
            payload = load_torch_file(checkpoint, map_location="cpu", weights_only=True)
        else:
            payload = checkpoint
        if isinstance(payload, Mapping):
            for key in ("model", "state_dict", "model_state"):
                if key in payload and isinstance(payload[key], Mapping):
                    payload = payload[key]
                    break
        source = {}
        for key, value in payload.items():
            key = str(key)
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            source[key] = value

        target = self.state_dict()
        adals_prefixes = self._adals_prefixes()
        compatible = {}
        skipped = []
        for key, value in source.items():
            if key.startswith(
                ("head.", "head_dist.", "classifier.", "identity_projection.")
            ):
                skipped.append(key)
                continue
            if any(key.startswith(prefix) for prefix in adals_prefixes):
                skipped.append(key)
                continue
            if key in target and target[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped.append(key)
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        if not compatible:
            warnings.warn(
                "No compatible LSNet-T parameters were loaded", RuntimeWarning
            )
        return {
            "loaded": sorted(compatible.keys()),
            "skipped_source": sorted(skipped),
            "missing_target": sorted(missing),
            "unexpected_target": sorted(unexpected),
        }


def build_cgsfnet(config: Mapping[str, object]) -> CGSFNet:
    keys = {
        "image_size",
        "num_classes",
        "embedding_dim",
        "channels",
        "depths",
        "key_dims",
        "num_heads",
        "small_context_kernel",
        "large_context_kernel",
        "ska_kernel",
        "ska_groups",
        "context_mode",
        "stage4_block",
    }
    arguments = {key: config[key] for key in keys if key in config}
    return CGSFNet(**arguments)
