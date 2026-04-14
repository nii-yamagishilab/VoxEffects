from __future__ import annotations

import torch


def gaussian_noise(audio: torch.Tensor, mean: float = 0.0, std: float = 0.1) -> torch.Tensor:
    return torch.randn_like(audio) * std + mean


def quantize(audio: torch.Tensor, num_bits: int) -> torch.Tensor:
    quant_levels = 2 ** int(num_bits)
    return torch.round((audio + 1) * (quant_levels / 2 - 1)) / (quant_levels / 2 - 1) - 1
