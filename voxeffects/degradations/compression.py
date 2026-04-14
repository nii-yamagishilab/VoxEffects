from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch

from ..audio_io import load_audio, save_audio


def _run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _codec_roundtrip(
    wav_tensor: torch.Tensor,
    sr: int,
    suffix: str,
    codec_args: list[str],
    ffmpeg4codecs: Optional[str] = None,
) -> torch.Tensor:
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape
    wav_tensor_flat = wav_tensor.view(1, -1).detach().cpu()

    with tempfile.NamedTemporaryFile(suffix=".wav") as f_in, tempfile.NamedTemporaryFile(suffix=suffix) as f_out:
        save_audio(Path(f_in.name), wav_tensor_flat, sr)
        ffmpeg = "ffmpeg" if ffmpeg4codecs is None else ffmpeg4codecs
        command = [ffmpeg, "-y", "-i", f_in.name, "-ar", str(sr), *codec_args, f_out.name]
        _run_ffmpeg(command)
        compressed, _ = load_audio(f_out.name)

    original_length_flat = batch_size * channels * original_length
    compressed_length_flat = compressed.shape[-1]
    if compressed_length_flat > original_length_flat:
        compressed = compressed[:, :original_length_flat]
    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(1, original_length_flat - compressed_length_flat, dtype=compressed.dtype)
        compressed = torch.cat((compressed, padding), dim=-1)

    return compressed.view(batch_size, channels, -1).to(device)


def mp3_wrapper(wav_tensor: torch.Tensor, sr: int, bitrate: str = "64k", ffmpeg4codecs: Optional[str] = None):
    match = re.search(r"\d+(\.\d+)?", bitrate)
    if not match:
        raise ValueError(f"Invalid bitrate specified: {bitrate}")
    return _codec_roundtrip(
        wav_tensor,
        sr,
        suffix=".mp3",
        codec_args=["-b:a", f"{match.group()}k", "-c:a", "libmp3lame"],
        ffmpeg4codecs=ffmpeg4codecs,
    )


def aac_wrapper(wav_tensor: torch.Tensor, sr: int, bitrate: str = "64k", ffmpeg4codecs: Optional[str] = None):
    match = re.search(r"\d+(\.\d+)?", bitrate)
    if not match:
        raise ValueError(f"Invalid bitrate specified: {bitrate}")
    return _codec_roundtrip(
        wav_tensor,
        sr,
        suffix=".aac",
        codec_args=["-b:a", f"{match.group()}k", "-c:a", "aac"],
        ffmpeg4codecs=ffmpeg4codecs,
    )


def vorbis_wrapper(wav_tensor: torch.Tensor, sr: int, bitrate: str = "64k", ffmpeg4codecs: Optional[str] = None):
    quality_map = {"48k": "-1", "64k": "0", "96k": "2", "128k": "4", "256k": "8"}
    if bitrate not in quality_map:
        raise ValueError(f"Invalid bitrate: {bitrate}")
    return _codec_roundtrip(
        wav_tensor,
        sr,
        suffix=".ogg",
        codec_args=["-aq", quality_map[bitrate], "-c:a", "libvorbis"],
        ffmpeg4codecs=ffmpeg4codecs,
    )
