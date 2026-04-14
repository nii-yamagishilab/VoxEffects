from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def _needs_soundfile_fallback(exc: Exception) -> bool:
    return "TorchCodec" in str(exc)


def load_audio(path: str | Path, frame_offset: int = 0, num_frames: int = -1):
    try:
        kwargs = {}
        if frame_offset:
            kwargs["frame_offset"] = int(frame_offset)
        if num_frames is not None and int(num_frames) >= 0:
            kwargs["num_frames"] = int(num_frames)
        return torchaudio.load(str(path), **kwargs)
    except ImportError as exc:
        if not _needs_soundfile_fallback(exc):
            raise
        try:
            import soundfile as sf
        except ImportError as sf_exc:
            raise ImportError(
                "torchaudio.load requires TorchCodec in this environment, and the soundfile fallback is unavailable. "
                "Install torchcodec or soundfile."
            ) from sf_exc
        start = int(frame_offset or 0)
        frames = -1 if num_frames is None else int(num_frames)
        data, sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())
        return waveform, int(sr)


def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int) -> None:
    try:
        torchaudio.save(str(path), audio, int(sample_rate), encoding="PCM_S", bits_per_sample=16)
    except ImportError as exc:
        if not _needs_soundfile_fallback(exc):
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


def audio_info(path: str | Path):
    try:
        info = torchaudio.info(str(path))
        return {"sample_rate": int(info.sample_rate), "num_frames": int(info.num_frames)}
    except Exception as exc:
        if not isinstance(exc, ImportError) and not isinstance(exc, AttributeError):
            raise
        if isinstance(exc, ImportError) and not _needs_soundfile_fallback(exc):
            raise
        try:
            import soundfile as sf
        except ImportError as sf_exc:
            raise ImportError(
                "torchaudio.info requires TorchCodec in this environment, and the soundfile fallback is unavailable. "
                "Install torchcodec or soundfile."
            ) from sf_exc
        info = sf.info(str(path))
        return {"sample_rate": int(info.samplerate), "num_frames": int(info.frames)}
