from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from .audio_io import load_audio
from .effects import (
    EFFECT_NAMES,
    build_board_from_indices,
    combo_counts,
    combos_per_file,
    decode_variant_index,
    load_class_map,
    load_presets,
    remap_path,
    stable_item_seed,
)


class VoxEffectsDataset(Dataset):
    """On-the-fly VoxEffects renderer.

    Each item is addressed by (source file, effect-combination id). The class id
    is therefore deterministic and independent of DataLoader worker order.
    """

    def __init__(
        self,
        dataset_csv: str | Path,
        presets_json: str | Path,
        class_map_csv: str | Path,
        samplerate: int = 16000,
        return_fbank: bool = True,
        fbank_num_mel_bins: int = 128,
        fbank_target_length: int = 1024,
        random_crop: bool = False,
        fbank_norm_mean: float = 0.0,
        fbank_norm_std: float = 1.0,
        mono: bool = True,
        input_sec: float | None = 6.0,
        deterministic_seed: int = 42,
        path_prefix_map: Sequence[Tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_csv = str(dataset_csv)
        self.files = pd.read_csv(dataset_csv, sep="|")["audio_filepath"].tolist()
        self.files = [remap_path(path, path_prefix_map) for path in self.files]
        self.preset_lists = load_presets(presets_json)
        self.class_map = load_class_map(class_map_csv)
        self.bases = combo_counts(self.preset_lists)
        self.combos_per_file = combos_per_file(self.preset_lists)
        self.samplerate = int(samplerate)
        self.return_fbank = bool(return_fbank)
        self.fbank_num_mel_bins = int(fbank_num_mel_bins)
        self.fbank_target_length = int(fbank_target_length)
        self.random_crop = bool(random_crop)
        self.fbank_norm_mean = float(fbank_norm_mean)
        self.fbank_norm_std = float(fbank_norm_std)
        self.mono = bool(mono)
        self.input_sec = None if input_sec is None else float(input_sec)
        self.deterministic_seed = int(deterministic_seed)
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}
        self._board_cache: Dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.files) * self.combos_per_file

    def _ensure_resampler(self, sr_in: int):
        if int(sr_in) == self.samplerate:
            return None
        if int(sr_in) not in self._resamplers:
            self._resamplers[int(sr_in)] = torchaudio.transforms.Resample(
                orig_freq=int(sr_in),
                new_freq=self.samplerate,
            )
        return self._resamplers[int(sr_in)]

    def _crop_or_pad_from_start(self, waveform: torch.Tensor, sr: int, pad_short: bool = False) -> torch.Tensor:
        if self.input_sec is None:
            return waveform
        target_t = int(round(float(sr) * self.input_sec))
        if target_t <= 0:
            return waveform
        if waveform.shape[-1] >= target_t:
            return waveform[:, :target_t]
        if not pad_short:
            return waveform
        return torch.nn.functional.pad(waveform, (0, target_t - waveform.shape[-1]))

    def _to_fbank(self, waveform: torch.Tensor, seed: int) -> torch.Tensor:
        wf = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            wf,
            htk_compat=True,
            sample_frequency=self.samplerate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.fbank_num_mel_bins,
            dither=0.0,
            frame_shift=10,
        )
        n_frames = fbank.shape[0]
        pad = self.fbank_target_length - n_frames
        if pad > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, pad))
        elif pad < 0:
            if self.random_crop:
                rng = torch.Generator().manual_seed(seed)
                start = torch.randint(0, -pad + 1, (1,), generator=rng).item()
                fbank = fbank[start : start + self.fbank_target_length, :]
            else:
                fbank = fbank[: self.fbank_target_length, :]
        fbank = (fbank - self.fbank_norm_mean) / (self.fbank_norm_std * 2.0)
        return fbank.unsqueeze(0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        file_idx = int(index) // self.combos_per_file
        variant_id = int(index) % self.combos_per_file
        wav_path = self.files[file_idx]
        item_seed = stable_item_seed(self.deterministic_seed, wav_path, variant_id, "none")

        board = self._board_cache.get(variant_id)
        chosen_groups = None
        if board is None:
            idx_vec = decode_variant_index(variant_id, self.bases)
            board, chosen_groups = build_board_from_indices(self.preset_lists, idx_vec)
            if len(self._board_cache) > 256:
                self._board_cache.clear()
            self._board_cache[variant_id] = board
        else:
            idx_vec = decode_variant_index(variant_id, self.bases)

        waveform, orig_sr = load_audio(wav_path)
        orig_sr = int(orig_sr)
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = self._crop_or_pad_from_start(waveform, orig_sr)

        effected = board(waveform.numpy().astype("float32"), orig_sr)
        effected = torch.from_numpy(effected)
        cur_sr = orig_sr
        resampler = self._ensure_resampler(cur_sr)
        if resampler is not None:
            effected = resampler(effected)
            cur_sr = self.samplerate

        if self.return_fbank:
            if effected.shape[0] > 1:
                effected = effected.mean(dim=0, keepdim=True)
            x = self._to_fbank(effected, item_seed)
        else:
            x = effected

        label_row = self.class_map[variant_id]
        binary_id = int(label_row["binary_class_id"])
        binary_vec = torch.tensor([(binary_id >> i) & 1 for i in range(len(EFFECT_NAMES))], dtype=torch.float32)

        item: Dict[str, Any] = {
            "x": x,
            "main_class_id": variant_id,
            "effect_indices": torch.tensor(idx_vec, dtype=torch.long),
            "binary_class_id": binary_vec,
            "num_active_effects": int(label_row["num_active_effects"]),
            "wav_path": wav_path,
            "orig_sr": orig_sr,
            "final_sr": cur_sr,
            "seed": item_seed,
        }
        if chosen_groups is not None:
            item["effect_groups"] = chosen_groups
        for name in EFFECT_NAMES:
            item[f"{name}_profile"] = label_row.get(f"{name}_profile")
            item[f"{name}_reg"] = label_row.get(f"{name}_reg")
        item["mean_reg"] = label_row.get("mean_reg")
        return item


def collate_pad(batch):
    xs = [item["x"] for item in batch]
    dims = xs[0].dim()
    if dims == 2:
        max_t = max(x.shape[-1] for x in xs)
        x_batch = torch.stack([torch.nn.functional.pad(x, (0, max_t - x.shape[-1])) for x in xs], 0)
    elif dims == 3:
        max_t = max(x.shape[1] for x in xs)
        padded = [torch.nn.functional.pad(x, (0, 0, 0, max_t - x.shape[1])) for x in xs]
        x_batch = torch.stack(padded, 0)
    else:
        raise ValueError(f"Unexpected tensor dims: {dims}")
    out = {key: [item[key] for item in batch] for key in batch[0].keys() if key != "x"}
    out["x"] = x_batch
    for key in ("main_class_id", "effect_indices", "binary_class_id", "num_active_effects"):
        if key in batch[0]:
            out[key] = torch.stack([torch.as_tensor(item[key]) for item in batch], 0)
    return out
