# Reproducibility Notes

This note summarizes the deterministic behavior mirrored from the project code.

## Deterministic Parts

- The published test protocols are fixed CSV files.
- Effect labels are fixed by `config/class_map.csv`.
- The preset Cartesian product is fixed by `config/speech_effect_chain_v2.json`.
- `variant_id == main_class_id`.
- `variant_id` is decoded as mixed radix over `[3, 5, 7, 3, 4, 2]`.
- The release loader uses the same group order: `de_noise`, `dynamic`, `eq`, `de_esser`, `reverb`, `limiter`.
- For evaluation, set `shuffle=False` and `random_crop=False`.

## Stable Seed

The original project uses a stable per-item BLAKE2b seed:

```text
seed = blake2b(base_seed | mode_tag | wav_path | variant_id, digest_size=8) & 0x7fffffff
```

In this release, `mode_tag` is `none` because the released renderer covers the deterministic effect-chain dataset without the separate audio-attack augmentation pipeline.

## What Can Still Differ

Audio tensors can differ across machines if:

- the source audio files differ,
- the local path mapping differs and path-dependent seeding is used,
- the Pedalboard, torch, torchaudio, or resampler versions differ,
- `random_crop=True` is used,
- DataLoader shuffling is enabled for benchmark evaluation.

For benchmark reproduction, use the released protocol CSVs, keep `shuffle=False`, keep `random_crop=False`, keep the same placeholder-root remapping across runs, and document package versions.
