from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

EFFECT_GROUP_KEYS = [
    "de_noise_presets",
    "dynamic_presets",
    "eq_presets",
    "de_esser_presets",
    "reverb_presets",
    "limiter_presets",
]

EFFECT_NAMES = ["de_noise", "dynamic", "eq", "de_esser", "reverb", "limiter"]


def stable_item_seed(base_seed: int, wav_path: str, variant_id: int, mode_tag: str = "none") -> int:
    """Stable 31-bit seed used by the original project for per-item determinism."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base_seed).encode("utf-8"))
    h.update(b"|")
    h.update(mode_tag.encode("utf-8"))
    h.update(b"|")
    h.update(wav_path.encode("utf-8"))
    h.update(b"|")
    h.update(str(int(variant_id)).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


def load_presets(path: str | Path) -> List[List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [data[key] for key in EFFECT_GROUP_KEYS]


def load_class_map(path: str | Path) -> Dict[int, Dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(path, sep="|")
    df["main_class_id"] = df["main_class_id"].astype(int)
    return df.set_index("main_class_id").to_dict(orient="index")


def combo_counts(preset_lists: Sequence[Sequence[Dict[str, Any]]]) -> List[int]:
    return [len(group) for group in preset_lists]


def combos_per_file(preset_lists: Sequence[Sequence[Dict[str, Any]]]) -> int:
    total = 1
    for count in combo_counts(preset_lists):
        total *= count
    return total


def decode_variant_index(idx: int, bases: Sequence[int]) -> List[int]:
    """Decode a mixed-radix class id. The last effect group changes fastest."""
    out = []
    idx = int(idx)
    for base in reversed(bases):
        out.append(idx % int(base))
        idx //= int(base)
    return list(reversed(out))


def build_board_from_indices(
    preset_lists: Sequence[Sequence[Dict[str, Any]]],
    indices: Sequence[int],
):
    from pedalboard import (
        Compressor,
        Gain,
        HighShelfFilter,
        Limiter,
        LowShelfFilter,
        NoiseGate,
        PeakFilter,
        Pedalboard,
        Reverb,
    )

    plugin_classes = {
        "NoiseGate": NoiseGate,
        "Compressor": Compressor,
        "Gain": Gain,
        "LowShelfFilter": LowShelfFilter,
        "HighShelfFilter": HighShelfFilter,
        "PeakFilter": PeakFilter,
        "Reverb": Reverb,
        "Limiter": Limiter,
    }

    chosen_groups = [preset_lists[group_id][int(indices[group_id])] for group_id in range(len(preset_lists))]
    specs: List[Dict[str, Any]] = []
    for group in chosen_groups:
        if not group.get("bypass", False):
            specs.extend(group.get("effects", []))

    plugins = []
    for spec in specs:
        plugin_type = spec.get("type")
        if plugin_type not in plugin_classes:
            raise ValueError(f"Unsupported Pedalboard plugin type: {plugin_type}")
        plugins.append(plugin_classes[plugin_type](**spec.get("params", {})))
    return Pedalboard(plugins), chosen_groups


def remap_path(path: str, path_prefix_map: Sequence[Tuple[str, str]] | None = None) -> str:
    if not path_prefix_map:
        return path
    for old, new in path_prefix_map:
        if path.startswith(old):
            return new + path[len(old) :]
    return path
