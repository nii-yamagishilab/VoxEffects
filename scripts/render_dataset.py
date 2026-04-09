#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch

from voxeffects.audio_io import save_audio
from voxeffects import VoxEffectsDataset


def parse_prefix_map(items: list[str] | None) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--path-prefix-map must use OLD=NEW, got: {item}")
        old, new = item.split("=", 1)
        out.append((old, new))
    return out


def save_wav_pcm16(path: Path, audio: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().cpu().to(torch.float32)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    save_audio(path, audio, int(sr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render VoxEffects audio files and labels from source protocols.")
    parser.add_argument("--dataset-csv", required=True, help="Pipe-delimited CSV with audio_filepath.")
    parser.add_argument("--output-dir", required=True, help="Directory for audio/, manifest.csv, and dataset.csv.")
    parser.add_argument("--presets-json", default="config/speech_effect_chain_v2.json")
    parser.add_argument("--class-map-csv", default="config/class_map.csv")
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--input-sec", type=float, default=6.0, help="Crop from start; use -1 to keep full audio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        default=[],
        help="Rewrite protocol paths at load time. May be repeated. Format: OLD=NEW",
    )
    args = parser.parse_args()

    input_sec = None if args.input_sec < 0 else args.input_sec
    out_dir = Path(args.output_dir)
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = VoxEffectsDataset(
        dataset_csv=args.dataset_csv,
        presets_json=args.presets_json,
        class_map_csv=args.class_map_csv,
        samplerate=args.samplerate,
        return_fbank=False,
        random_crop=False,
        input_sec=input_sec,
        deterministic_seed=args.seed,
        path_prefix_map=parse_prefix_map(args.path_prefix_map),
    )

    n_items = len(ds) if args.max_items is None else min(len(ds), args.max_items)
    rows = []
    for index in range(n_items):
        item = ds[index]
        wav_path = item["wav_path"]
        variant_id = int(item["main_class_id"])
        path_hash = hashlib.blake2b(wav_path.encode("utf-8"), digest_size=4).hexdigest()
        stem = Path(wav_path).stem
        out_name = f"{stem}__p{path_hash}__vid{variant_id:06d}.wav"
        out_path = audio_dir / out_name
        save_wav_pcm16(out_path, item["x"], int(item["final_sr"]))

        rows.append(
            {
                "audio_filepath": str(out_path),
                "orig_audio_filepath": wav_path,
                "main_class_id": variant_id,
                "effect_indices": " ".join(str(int(v)) for v in item["effect_indices"].tolist()),
                "binary_class_id": " ".join(str(int(v)) for v in item["binary_class_id"].tolist()),
                "num_active_effects": int(item["num_active_effects"]),
                "seed": int(item["seed"]),
                "orig_sr": int(item["orig_sr"]),
                "final_sr": int(item["final_sr"]),
            }
        )
        if (index + 1) % 200 == 0:
            print(f"rendered {index + 1}/{n_items}", flush=True)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", sep="|", index=False, quoting=csv.QUOTE_MINIMAL)
    manifest[["audio_filepath", "main_class_id"]].to_csv(out_dir / "dataset.csv", sep="|", index=False)
    print(f"done: {n_items} items")
    print(f"audio: {audio_dir}")
    print(f"manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
