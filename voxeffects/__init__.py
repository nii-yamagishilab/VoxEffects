"""Utilities for reproducing the VoxEffects dataset release."""

from .effects import (
    decode_variant_index,
    load_class_map,
    load_presets,
    stable_item_seed,
)

__all__ = [
    "VoxEffectsDataset",
    "collate_pad",
    "decode_variant_index",
    "load_class_map",
    "load_presets",
    "stable_item_seed",
]


def __getattr__(name):
    if name in {"VoxEffectsDataset", "collate_pad"}:
        from .dataset import VoxEffectsDataset, collate_pad

        return {"VoxEffectsDataset": VoxEffectsDataset, "collate_pad": collate_pad}[name]
    raise AttributeError(name)
