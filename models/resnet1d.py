"""
ResNet-18 audio encoder adapted for spectrogram input.

Matches the paper's "Reproducibility Details" (Sec. 5): a standard ResNet-18
adapted for 1D audio (taking spectrogram inputs), consisting of L=8
splittable blocks with output embedding dimension d=128. The L=8 blocks are
exactly the 8 BasicBlocks of a standard ResNet-18 (4 stages x 2 blocks),
excluding the stem, which is treated as always-edge-side feature extraction
(analogous to FFT/mel-spectrogram extraction always running locally).
"""

import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    """Standard ResNet basic block (2x Conv+BN+ReLU with residual connection)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class AudioResNet18(nn.Module):
    """ResNet-18 audio encoder with L=8 splittable blocks (paper Sec. 5)."""

    NUM_BLOCKS = 8

    def __init__(self, embedding_dim: int = 128, input_channels: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Stem: always-edge-side feature extraction, not part of the L=8
        # splittable blocks exposed to the Control Plane (Sec. 4.2.2).
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 stages x 2 blocks = 8 BasicBlocks, matching L=8.
        stage_channels = [64, 64, 128, 128, 256, 256, 512, 512]
        stage_strides = [1, 1, 2, 1, 2, 1, 2, 1]

        blocks = []
        in_channels = 64
        for out_channels, stride in zip(stage_channels, stage_strides):
            blocks.append(BasicBlock1D(in_channels, out_channels, stride))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        assert len(self.blocks) == self.NUM_BLOCKS

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(in_channels, embedding_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass. Returns unnormalized embedding [batch, embedding_dim]."""
        features = self.forward_up_to(x, self.NUM_BLOCKS)
        return self.forward_from(features, self.NUM_BLOCKS)

    def forward_up_to(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Edge-side partial execution for split point k (Sec. 4.2.2).

        Runs the stem plus the first k of the L=8 blocks.

        Args:
            x: Input spectrogram [batch, channels, n_mels, time]
            k: Number of blocks to execute locally, 0 <= k <= L

        Returns:
            Intermediate feature map to transmit if k < L
        """
        if not 0 <= k <= self.NUM_BLOCKS:
            raise ValueError(f"k must be in [0, {self.NUM_BLOCKS}], got {k}")
        x = self.stem(x)
        for block in self.blocks[:k]:
            x = block(x)
        return x

    def forward_from(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """
        Server-side completion of a split forward pass.

        Runs the remaining L-k blocks, then pools and projects to the
        embedding space.

        Args:
            features: Intermediate feature map from forward_up_to(x, k)
            k: Split point the features were produced at

        Returns:
            Embedding [batch, embedding_dim]
        """
        if not 0 <= k <= self.NUM_BLOCKS:
            raise ValueError(f"k must be in [0, {self.NUM_BLOCKS}], got {k}")
        x = features
        for block in self.blocks[k:]:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.project(x)
