#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from typing import List, Tuple

import torch

from voxeffects import VoxEffectsDataset


def parse_prefix_map(items: list[str] | None) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--path-prefix-map must use OLD=NEW, got: {item}")
        old, new = item.split("=", 1)
        out.append((old, new))
    return out


def tensor_digest(x: torch.Tensor) -> str:
    y = x.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(y.shape)).encode("utf-8"))
    h.update(str(y.dtype).encode("utf-8"))
    h.update(y.numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check deterministic VoxEffects rendering for a local source setup.")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--presets-json", default="config/speech_effect_chain_v2.json")
    parser.add_argument("--class-map-csv", default="config/class_map.csv")
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--input-sec", type=float, default=6.0, help="Crop from start; use -1 to keep full audio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-items", type=int, default=16)
    parser.add_argument("--path-prefix-map", action="append", default=[], help="Format: OLD=NEW")
    args = parser.parse_args()

    kwargs = dict(
        dataset_csv=args.dataset_csv,
        presets_json=args.presets_json,
        class_map_csv=args.class_map_csv,
        samplerate=args.samplerate,
        return_fbank=False,
        random_crop=False,
        input_sec=None if args.input_sec < 0 else args.input_sec,
        deterministic_seed=args.seed,
        path_prefix_map=parse_prefix_map(args.path_prefix_map),
    )
    a = VoxEffectsDataset(**kwargs)
    b = VoxEffectsDataset(**kwargs)
    n = min(args.num_items, len(a))
    for idx in range(n):
        item_a = a[idx]
        item_b = b[idx]
        fields = ["main_class_id", "num_active_effects", "seed", "orig_sr", "final_sr"]
        for field in fields:
            if item_a[field] != item_b[field]:
                raise AssertionError(f"Mismatch at index {idx}, field {field}: {item_a[field]} != {item_b[field]}")
        if not torch.equal(item_a["effect_indices"], item_b["effect_indices"]):
            raise AssertionError(f"effect_indices mismatch at index {idx}")
        if tensor_digest(item_a["x"]) != tensor_digest(item_b["x"]):
            raise AssertionError(f"audio tensor mismatch at index {idx}")
    print(f"deterministic check passed for {n} items")


if __name__ == "__main__":
    main()
