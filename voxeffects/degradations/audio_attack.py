from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import torch
import torchaudio
import yaml
from torch import nn

from ..audio_io import audio_info, load_audio
from .compression import aac_wrapper, mp3_wrapper, vorbis_wrapper
from .noise import gaussian_noise, quantize
from .utils import choose_random_uniform_val, filter_kwargs, normalize_weights, ste


class AudioAttack(nn.Module):
    """Public reproduction of the degradation module used in the paper experiments."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        mode: str = "train",
        config_path: Optional[str] = None,
        ffmpeg4codecs: Optional[str] = None,
        mixing_train_filepath: Optional[str] = None,
        delimiter: str = "|",
        single_attack: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        if config_path is None:
            raise ValueError("config_path is required when degradations are enabled.")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.data_dir = data_dir
        self.mode = mode
        self.single_attack = single_attack
        self.device = device
        self.ffmpeg4codecs = ffmpeg4codecs
        self.df_train_mixing = (
            pd.read_csv(mixing_train_filepath, sep=delimiter) if (mode == "train" and mixing_train_filepath) else None
        )

        self.dict_attacks = {}
        for attack_dict in self.config["attacks"]:
            attack_type = list(attack_dict.keys())[0]
            self.dict_attacks[attack_type] = getattr(self, f"apply_{attack_type}")

        self.attack_probs = normalize_weights(
            (list(d.keys())[0], float(list(d.values())[0])) for d in self.config["attacks"]
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_sr: int,
        attack_type: Optional[str] = None,
        return_attack_params: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, str, dict]]:
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        if audio.shape[1] > 1:
            raise NotImplementedError("Only mono audio is supported.")

        kwargs = {"x_sr": audio_sr, **kwargs}

        if self.single_attack:
            if attack_type is None:
                names = list(self.attack_probs.keys())
                probs = torch.tensor([self.attack_probs[name] for name in names], dtype=torch.float)
                attack_type = names[torch.multinomial(probs, num_samples=1).item()]
            attack_fn = self.dict_attacks[attack_type]
            distorted_audio, attack_type, attack_params = attack_fn(audio, **filter_kwargs(attack_fn, kwargs))
        else:
            names = list(self.attack_probs.keys())
            probs = torch.tensor([self.attack_probs[name] for name in names], dtype=torch.float)
            num_samples = 2 if len(names) >= 2 else 1
            idxs = torch.multinomial(probs, num_samples=num_samples, replacement=False)
            selected = [names[i] for i in idxs.tolist()]
            distorted_audio = audio
            chain = []
            for name in selected:
                fn = self.dict_attacks[name]
                distorted_audio, _, params = fn(distorted_audio, **filter_kwargs(fn, kwargs))
                chain.append({"type": name, "params": params})
            attack_type = "+".join(selected)
            attack_params = {"chain": chain}

        if distorted_audio.abs().max() > 1 and "gain" not in attack_type.split("+"):
            distorted_audio = distorted_audio / distorted_audio.abs().max()
        if return_attack_params:
            return distorted_audio, attack_type, attack_params
        return distorted_audio

    def apply_gaussian_noise(self, x: torch.Tensor, snr: Optional[float] = None):
        cfg = self.config["gaussian_noise"]
        if snr is None:
            if self.mode == "train":
                snr = choose_random_uniform_val(cfg["min_snr"], cfg["max_snr"], 1)
            else:
                raise ValueError("snr should be provided in val and test modes.")
        noise = gaussian_noise(x, std=float(cfg["std"]))
        snr_t = torch.tensor([snr], device=x.device).unsqueeze(0)
        x_power = x.pow(2).mean()
        noise_power = noise.pow(2).mean() + 1e-12
        alpha = torch.sqrt(x_power / noise_power) * (10 ** (-snr_t / 20))
        return x + alpha * noise, "gaussian_noise", {"snr": float(snr_t.item())}

    def apply_background_noise(
        self,
        x: torch.Tensor,
        x_sr: int,
        snr: Optional[float] = None,
        noise_filepath: Optional[str] = None,
        noise_sr: Optional[int] = None,
    ):
        cfg = self.config["background_noise"]
        if self.mode == "train" and snr is None:
            snr = choose_random_uniform_val(cfg["min_snr"], cfg["max_snr"], 1)
        elif self.mode in {"test", "val"} and (snr is None or noise_filepath is None):
            raise ValueError("snr and noise_filepath should be provided in val and test modes.")

        audio_len = x.shape[-1]
        if noise_filepath is None:
            if self.df_train_mixing is None or self.data_dir is None:
                raise ValueError("background noise attack requires mixing_train_filepath and mixing_data_dir.")
            random_row = self.df_train_mixing.sample(n=1, random_state=int(torch.initial_seed() % (2**32 - 1))).iloc[0]
            noise_filepath = str(random_row["audio_filepath"])
            full_path = os.path.join(self.data_dir, noise_filepath)
            info = audio_info(full_path)
            orig_noise_sr = info["sample_rate"]
            required_frames = int(audio_len * (orig_noise_sr / x_sr))
            max_start = max(0, info["num_frames"] - required_frames)
            start_idx = int(torch.randint(low=0, high=max_start + 1, size=(1,)).item())
            noise, _ = load_audio(full_path, frame_offset=start_idx, num_frames=required_frames)
        else:
            full_path = noise_filepath if os.path.isabs(noise_filepath) else os.path.join(self.data_dir or "", noise_filepath)
            info = audio_info(full_path)
            orig_noise_sr = info["sample_rate"]
            required_frames = int(audio_len * (orig_noise_sr / x_sr))
            noise, _ = load_audio(full_path, num_frames=required_frames)

        if orig_noise_sr != x_sr:
            noise = torchaudio.transforms.Resample(orig_freq=orig_noise_sr, new_freq=x_sr)(noise)
        noise = noise.unsqueeze(0).to(x.device)
        if noise.shape[-1] < audio_len:
            noise = torch.nn.functional.pad(noise, (0, audio_len - noise.shape[-1]))
        elif noise.shape[-1] > audio_len:
            noise = noise[:, :, :audio_len]

        snr_t = torch.tensor([snr], device=x.device).unsqueeze(0)
        x_power = x.pow(2).mean()
        noise_power = noise.pow(2).mean() + 1e-12
        alpha = torch.sqrt(x_power / noise_power) * (10 ** (-snr_t / 20))
        return x + alpha * noise, "background_noise", {"snr": float(snr_t.item())}

    def apply_mp3(self, x: torch.Tensor, x_sr: int, bitrate: Optional[str] = None, ffmpeg4codecs: Optional[str] = None):
        bitrate = bitrate or self._choose_bitrate("mp3")
        ffmpeg4codecs = ffmpeg4codecs or self.ffmpeg4codecs
        return ste(x, mp3_wrapper(x, sr=x_sr, bitrate=bitrate, ffmpeg4codecs=ffmpeg4codecs)), "mp3", {"bitrate": bitrate}

    def apply_vorbis(self, x: torch.Tensor, x_sr: int, bitrate: Optional[str] = None, ffmpeg4codecs: Optional[str] = None):
        bitrate = bitrate or self._choose_bitrate("vorbis")
        ffmpeg4codecs = ffmpeg4codecs or self.ffmpeg4codecs
        return ste(x, vorbis_wrapper(x, sr=x_sr, bitrate=bitrate, ffmpeg4codecs=ffmpeg4codecs)), "vorbis", {"bitrate": bitrate}

    def apply_aac(self, x: torch.Tensor, x_sr: int, bitrate: Optional[str] = None, ffmpeg4codecs: Optional[str] = None):
        bitrate = bitrate or self._choose_bitrate("aac")
        ffmpeg4codecs = ffmpeg4codecs or self.ffmpeg4codecs
        return ste(x, aac_wrapper(x, sr=x_sr, bitrate=bitrate, ffmpeg4codecs=ffmpeg4codecs)), "aac", {"bitrate": bitrate}

    def apply_quantization(self, x: torch.Tensor, num_bits: Optional[int] = None):
        cfg = self.config["quantization"]
        if num_bits is None:
            if self.mode == "train":
                num_bits = torch.randint(int(cfg["min_bits"]), int(cfg["max_bits"]) + 1, (1,)).item()
            else:
                raise ValueError("num_bits should be provided in val and test modes.")
        return quantize(x, num_bits=int(num_bits)), "quantization", {"num_bits": int(num_bits)}

    def apply_resample(self, x: torch.Tensor, x_sr: int, target_sr: Optional[int] = None):
        cfg = self.config["resample"]
        if target_sr is None:
            if self.mode == "train":
                choices = cfg["sr_list"]
                target_sr = choices[torch.randint(0, len(choices), (1,)).item()]
            else:
                raise ValueError("target_sr should be provided in val and test modes.")
        y = torchaudio.transforms.Resample(orig_freq=x_sr, new_freq=target_sr)(x)
        y = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=x_sr)(y)
        return y, "resample", {"sr": int(target_sr)}

    def _choose_bitrate(self, key: str) -> str:
        if self.mode != "train":
            raise ValueError(f"bitrate should be provided in val and test modes for {key}.")
        choices = self.config[key]
        return choices[torch.randint(0, len(choices), (1,)).item()]
