"""Pure-PyTorch reconstruction of NVIDIA Modulus AFNO (FourCastNet v1).

Architecture mirrors modulus.models.afno.afno.AFNO with the exact constructor
args read from the released nvidia/fourcastnet1 .mdlus archive:
  inp_shape=[720,1440], in/out=26, patch=[8,8], embed=768, depth=12,
  mlp_ratio=4, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0
The block-diagonal AFNO mixing operates on the FULL 2D rfft of the token grid,
giving a globally-coupled (all-to-all) effective receptive field per forward pass.
"""

from __future__ import annotations

import torch
import torch.fft
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    """Adaptive Fourier Neural Operator mixing block (block-diagonal in channels)."""

    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        hidden_size_factor=1,
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.scale = 0.02
        bs = self.block_size
        f = hidden_size_factor
        self.w1 = nn.Parameter(self.scale * torch.randn(2, num_blocks, bs, bs * f))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, num_blocks, bs * f))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, num_blocks, bs * f, bs))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, num_blocks, bs))

    def forward(self, x):
        # x: (B, H, W, C)
        bias = x
        B, H, W, C = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # global 2D FFT over token grid
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device,
        )
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept = int(total_modes * self.hard_thresholding_fraction)

        def E(a, b):  # einsum helper
            return torch.einsum("nyxbi,bio->nyxbo", a, b)

        o1_real[:, :kept] = torch.relu(
            E(x[:, :kept].real, self.w1[0]) - E(x[:, :kept].imag, self.w1[1]) + self.b1[0]
        )
        o1_imag[:, :kept] = torch.relu(
            E(x[:, :kept].imag, self.w1[0]) + E(x[:, :kept].real, self.w1[1]) + self.b1[1]
        )
        o2_real[:, :kept] = (
            E(o1_real[:, :kept], self.w2[0]) - E(o1_imag[:, :kept], self.w2[1]) + self.b2[0]
        )
        o2_imag[:, :kept] = (
            E(o1_imag[:, :kept], self.w2[0]) + E(o1_real[:, :kept], self.w2[1]) + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = torch.nn.functional.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        return x + bias


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class PatchEmbed(nn.Module):
    def __init__(self, inp_shape, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.h = inp_shape[0] // patch_size[0]
        self.w = inp_shape[1] // patch_size[1]
        self.num_patches = self.h * self.w

    def forward(self, x):
        x = self.proj(x)  # (B, embed, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed)
        return x


class AFNONet(nn.Module):
    def __init__(
        self,
        inp_shape=(720, 1440),
        patch_size=(8, 8),
        in_channels=26,
        out_channels=26,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(inp_shape, patch_size, in_channels, embed_dim)
        self.h = self.patch_embed.h
        self.w = self.patch_embed.w
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.register_buffer("device_buffer", torch.zeros(0))
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    mlp_ratio,
                    drop_rate,
                    num_blocks,
                    sparsity_threshold,
                    hard_thresholding_fraction,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(embed_dim, out_channels * patch_size[0] * patch_size[1], bias=False)

    def forward(self, x):
        # x: (B, C, H, W) normalized
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.head(x)  # (B, h, w, C*p*p)
        # un-patchify
        x = x.reshape(B, self.h, self.w, self.patch_size[0], self.patch_size[1], self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(
            B, self.out_channels, self.h * self.patch_size[0], self.w * self.patch_size[1]
        )
        return x


VARIABLES = [
    "u10m",
    "v10m",
    "t2m",
    "sp",
    "msl",
    "t850",
    "u1000",
    "v1000",
    "z1000",
    "u850",
    "v850",
    "z850",
    "u500",
    "v500",
    "z500",
    "t500",
    "z50",
    "r500",
    "r850",
    "tcwv",
    "u100m",
    "v100m",
    "u250",
    "v250",
    "z250",
    "t250",
]
