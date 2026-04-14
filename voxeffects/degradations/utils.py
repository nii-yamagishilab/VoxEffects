from __future__ import annotations

from typing import Dict, Iterable

import torch


def ste(original: torch.Tensor, compressed: torch.Tensor) -> torch.Tensor:
    return original + (compressed - original).detach()


def choose_random_uniform_val(min_val: float, max_val: float, num_samples: int = 1):
    rand_val = torch.rand(num_samples) * (max_val - min_val) + min_val
    return rand_val.item() if num_samples == 1 else rand_val


def filter_kwargs(fn, kwargs: Dict):
    import inspect

    valid = inspect.signature(fn).parameters.keys()
    return {k: v for k, v in kwargs.items() if k in valid}


def normalize_weights(weight_items: Iterable[tuple[str, float]]) -> Dict[str, float]:
    weight_items = list(weight_items)
    total = sum(weight for _, weight in weight_items)
    return {name: weight / total for name, weight in weight_items}
