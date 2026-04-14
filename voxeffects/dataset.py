from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from .audio_io import load_audio
from .degradations import AudioAttack
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
        apply_audio_attacks_pre: bool = False,
        apply_audio_attacks_post: bool = False,
        apply_audio_attacks_both: bool = False,
        apply_single_audio_attack: bool = True,
        attacks_config_path: str | Path | None = None,
        mixing_data_dir: str | None = None,
        mixing_train_filepath: str | None = None,
        ffmpeg4codecs: str | None = None,
        deterministic_aug: bool = False,
        deterministic_aug_seed: int | None = None,
        aug_prob: float = 1.0,
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
        self.apply_audio_attacks_pre = bool(apply_audio_attacks_pre)
        self.apply_audio_attacks_post = bool(apply_audio_attacks_post)
        self.apply_audio_attacks_both = bool(apply_audio_attacks_both)
        self.deterministic_aug = bool(deterministic_aug)
        self.deterministic_aug_seed = (
            self.deterministic_seed if deterministic_aug_seed is None else int(deterministic_aug_seed)
        )
        self.aug_prob = float(aug_prob)
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}
        self._board_cache: Dict[int, Any] = {}

        self.attack = None
        if self.apply_audio_attacks_pre or self.apply_audio_attacks_post:
            if attacks_config_path is None:
                raise ValueError("attacks_config_path is required when degradations are enabled.")
            self.attack = AudioAttack(
                data_dir=mixing_data_dir,
                mode="train",
                config_path=str(attacks_config_path),
                ffmpeg4codecs=ffmpeg4codecs,
                mixing_train_filepath=mixing_train_filepath,
                single_attack=apply_single_audio_attack,
                device="cpu",
            )

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

    def _audio_attacks_mode_tag(self) -> str:
        if self.apply_audio_attacks_pre and not self.apply_audio_attacks_post:
            return "pre_only"
        if (not self.apply_audio_attacks_pre) and self.apply_audio_attacks_post:
            return "post_only"
        if self.apply_audio_attacks_pre and self.apply_audio_attacks_post:
            return "pre_and_post" if self.apply_audio_attacks_both else "pre_or_post"
        return "none"

    def _should_augment(self, rng: torch.Generator | None = None) -> bool:
        if not (self.apply_audio_attacks_pre or self.apply_audio_attacks_post):
            return False
        if self.aug_prob >= 1.0:
            return True
        if self.aug_prob <= 0.0:
            return False
        return torch.rand((), generator=rng).item() < self.aug_prob

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
        item_seed = stable_item_seed(
            self.deterministic_aug_seed if self.deterministic_aug else self.deterministic_seed,
            wav_path,
            variant_id,
            self._audio_attacks_mode_tag(),
        )
        rng = torch.Generator().manual_seed(item_seed)

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

        attack_type_pre = "n/a"
        attack_type_post = "n/a"
        do_aug = self._should_augment(rng if self.deterministic_aug else None)
        apply_audio_attacks_pre = False
        apply_audio_attacks_post = False

        if do_aug:
            if self.apply_audio_attacks_pre and not self.apply_audio_attacks_post:
                apply_audio_attacks_pre = True
            elif (not self.apply_audio_attacks_pre) and self.apply_audio_attacks_post:
                apply_audio_attacks_post = True
            elif self.apply_audio_attacks_pre and self.apply_audio_attacks_post:
                if self.apply_audio_attacks_both:
                    apply_audio_attacks_pre = True
                    apply_audio_attacks_post = True
                else:
                    random_choice = torch.randint(0, 2, (1,), generator=rng if self.deterministic_aug else None).item()
                    apply_audio_attacks_pre = random_choice == 0
                    apply_audio_attacks_post = random_choice == 1

        _py_state = random.getstate()
        _np_state = np.random.get_state()
        _torch_state = torch.random.get_rng_state()
        if self.deterministic_aug:
            random.seed(item_seed)
            np.random.seed(item_seed % (2**32 - 1))
            torch.manual_seed(item_seed)

        try:
            if apply_audio_attacks_pre:
                waveform, attack_type_pre, _ = self.attack(
                    audio=waveform,
                    audio_sr=orig_sr,
                    return_attack_params=True,
                )
                waveform = waveform.squeeze(0)

            effected = board(waveform.numpy().astype("float32"), orig_sr)
            effected = torch.from_numpy(effected)
            cur_sr = orig_sr

            if apply_audio_attacks_post:
                effected, attack_type_post, _ = self.attack(
                    audio=effected,
                    audio_sr=cur_sr,
                    return_attack_params=True,
                )
                effected = effected.squeeze(0)

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
        finally:
            if self.deterministic_aug:
                random.setstate(_py_state)
                np.random.set_state(_np_state)
                torch.random.set_rng_state(_torch_state)

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
            "attack_type_pre": attack_type_pre,
            "attack_type_post": attack_type_post,
            "attack_type": (
                f"pre:{attack_type_pre} | post:{attack_type_post}"
                if apply_audio_attacks_pre and apply_audio_attacks_post
                else f"pre:{attack_type_pre}"
                if apply_audio_attacks_pre
                else f"post:{attack_type_post}"
                if apply_audio_attacks_post
                else "n/a"
            ),
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
