from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(path: str | Path):
    try:
        return torchaudio.load(str(path))
    except ImportError as exc:
        if "TorchCodec" not in str(exc):
            raise
        try:
            import soundfile as sf
        except ImportError as sf_exc:
            raise ImportError(
                "torchaudio.load requires TorchCodec in this environment, and the soundfile fallback is unavailable. "
                "Install torchcodec or soundfile."
            ) from sf_exc
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())
        return waveform, int(sr)


def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int) -> None:
    try:
        torchaudio.save(str(path), audio, int(sample_rate), encoding="PCM_S", bits_per_sample=16)
    except ImportError as exc:
        if "TorchCodec" not in str(exc):
            raise
        try:
            import soundfile as sf
        except ImportError as sf_exc:
            raise ImportError(
                "torchaudio.save requires TorchCodec in this environment, and the soundfile fallback is unavailable. "
                "Install torchcodec or soundfile."
            ) from sf_exc
        x = audio.detach().cpu().to(torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        sf.write(str(path), x.T.numpy(), int(sample_rate), subtype="PCM_16")
