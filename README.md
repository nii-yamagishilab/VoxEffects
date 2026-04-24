# VoxEffects: A Speech-Oriented Audio Effects Dataset and Benchmark

Paper: [arXiv:2604.12389](https://arxiv.org/abs/2604.12389)

This repository releases reproducibility utilities for the VoxEffects datasets described in our paper. **If you use this program or the VoxEffects database, you must cite the paper above.** See the [Citation](#citation) section for details.

The rendered audio files are not included. VoxEffects is derived from source utterances in external speech datasets, so this release provides:

- effect presets and class labels,
- scripts to render audio locally from the original source audio,
- an on-the-fly PyTorch data loader,
- fixed test protocols for reproducing the reported benchmark splits.

## Contents

```text
config/
  speech_effect_chain_v2.json   # six effect groups and Pedalboard parameters
  attack.yaml                   # degradation settings used in robustness experiments
  class_map.csv                 # 2,520 effect-combination labels
protocols/
  test/                         # fixed DAPS-NII, EARS, TSP, and VCTK test protocols
  gender/                       # fixed gender-balanced protocols
scripts/
  render_dataset.py             # offline rendering to WAV + manifest
  check_reproducibility.py      # local deterministic rendering check
voxeffects/
  dataset.py                    # on-the-fly data loader
  degradations/                 # pre/post capture and platform degradations
  effects.py                    # preset, label, and deterministic-id utilities
```

The six preset groups are `de_noise`, `dynamic`, `eq`, `de_esser`, `reverb`, and `limiter`.
The group sizes are `[3, 5, 7, 3, 4, 2]`, giving `3 * 5 * 7 * 3 * 4 * 2 = 2520` effect combinations per source utterance.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the PyTorch and torchaudio builds that match your platform if the default packages are not appropriate for your environment.
If your torchaudio build expects TorchCodec for WAV IO, the release code also supports a `soundfile` fallback.

## Render Audio Files Offline

The published protocol CSVs use neutral placeholder roots under `/datasets/...`. Use `--path-prefix-map OLD=NEW` to point each placeholder root to your local dataset copy.

```bash
python scripts/render_dataset.py \
  --dataset-csv protocols/test/vctk_test_sample.csv \
  --output-dir rendered/vctk_test \
  --presets-json config/speech_effect_chain_v2.json \
  --class-map-csv config/class_map.csv \
  --path-prefix-map /datasets/CSTR_VCTK-Corpus=/data/CSTR_VCTK-Corpus
```

This writes:

- `rendered/vctk_test/audio/*.wav`
- `rendered/vctk_test/manifest.csv`
- `rendered/vctk_test/dataset.csv`

`manifest.csv` includes the source path, effect-combination class id, effect index vector, binary effect-presence label, active-effect count, seed, and sample rates.

One clean input utterance expands to 2520 rendered outputs if you materialize every effect combination locally. That is manageable for a small protocol, but it grows quickly for larger source sets, which is why we recommend the on-the-fly loader for most training and evaluation workflows.

To reproduce the robustness experiments, the renderer and loader also support the same degradation modes used in the paper:

- `none`
- `pre_only`
- `post_only`
- `pre_or_post`
- `pre_and_post`

These degradations cover background noise, Gaussian noise, AAC/MP3/Vorbis compression, quantization, and resampling, using the public [attack.yaml](/Users/zhangzhe/VibeProjects/AudioMAE/VoxEffects/config/attack.yaml) configuration.

## Use the On-The-Fly Loader

```python
from torch.utils.data import DataLoader
from voxeffects import VoxEffectsDataset, collate_pad

dataset = VoxEffectsDataset(
    dataset_csv="protocols/test/vctk_test_sample.csv",
    presets_json="config/speech_effect_chain_v2.json",
    class_map_csv="config/class_map.csv",
    return_fbank=True,
    random_crop=False,
    path_prefix_map=[
        ("/datasets/CSTR_VCTK-Corpus", "/data/CSTR_VCTK-Corpus"),
    ],
)

loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_pad)
batch = next(iter(loader))
```

Each item corresponds to `(source file, variant id)`, where `variant id == main_class_id`.
The dataset length is `number_of_source_files * 2520`.

To enable degradation in the on-the-fly loader:

```python
dataset = VoxEffectsDataset(
    dataset_csv="protocols/test/vctk_test_sample.csv",
    presets_json="config/speech_effect_chain_v2.json",
    class_map_csv="config/class_map.csv",
    attacks_config_path="config/attack.yaml",
    apply_audio_attacks_pre=True,
    apply_audio_attacks_post=True,
    apply_audio_attacks_both=False,   # pre_or_post
    deterministic_aug=True,
    deterministic_aug_seed=42,
    ffmpeg4codecs="/path/to/ffmpeg",
    mixing_data_dir="/path/to/noise_root",
    mixing_train_filepath="/path/to/noise_manifest.csv",
    path_prefix_map=[
        ("/datasets/CSTR_VCTK-Corpus", "/data/CSTR_VCTK-Corpus"),
    ],
)
```

For deterministic benchmark reproduction with degradations, keep `deterministic_aug=True`, keep the same `deterministic_aug_seed`, and keep the same path remapping and attack config across runs.

### Noise Dataset For `background_noise`

The `background_noise` degradation used in the paper experiments relies on the DEMAND noise corpus:

- Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent, "The Diverse Environments Multi-Channel Acoustic Noise Database (DEMAND): A database of multichannel environmental noise recordings," 2013.
- Dataset: [Zenodo](https://zenodo.org/records/1227121)
- Dataset index: [DCASE datalist entry](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/demand.html)
- License: CC BY-SA 3.0, as listed on the Zenodo record

The repository already includes the noise manifest format expected by the attack code in [config/attack.yaml](/Users/zhangzhe/VibeProjects/AudioMAE/VoxEffects/config/attack.yaml): each row contains a relative `audio_filepath`, plus duration and sample rate metadata.

To reproduce the paper's `background_noise` setup:

1. Download the DEMAND archive from Zenodo.
2. Extract the noise WAV directories under one root folder, for example:
   - `/data/DEMAND/SCAFE_48k/...`
   - `/data/DEMAND/DKITCHEN_16k/...`
3. Prepare a `|`-delimited manifest with the same columns as the project metadata:
   - `audio_filepath|duration|sample_rate`
4. Pass that root and manifest to the loader or renderer:

```bash
python scripts/render_dataset.py \
  --dataset-csv protocols/test/vctk_test_sample.csv \
  --output-dir rendered/vctk_test_pre_only \
  --aug-mode pre_only \
  --attacks-config-path config/attack.yaml \
  --mixing-data-dir /data/DEMAND \
  --mixing-train-filepath path/to/demand_metadata.csv \
  --ffmpeg4codecs /path/to/ffmpeg \
  --deterministic-aug \
  --deterministic-aug-seed 42 \
  --path-prefix-map /datasets/CSTR_VCTK-Corpus=/data/CSTR_VCTK-Corpus
```

If you do not need the `background_noise` degradation, you can still reproduce the other degradations with `attack.yaml` alone, plus `ffmpeg` for codec-based attacks.

## Test Protocols

The released benchmark protocols are:

- In-domain:
  `protocols/test/daps_nii_test_sample.csv`, `protocols/test/ears_test_sample.csv`, and `protocols/test/tsp_test_sample.csv`
- Out-of-domain:
  `protocols/test/vctk_test_sample.csv`
- Gender-balanced:
  `protocols/gender/gender_balanced_test_female_60.csv` and `protocols/gender/gender_balanced_test_male_60.csv`

Protocol sizes:

- `protocols/test/daps_nii_test_sample.csv`: 20 utterances
- `protocols/test/ears_test_sample.csv`: 20 utterances
- `protocols/test/tsp_test_sample.csv`: 20 utterances
- `protocols/test/vctk_test_sample.csv`: 60 utterances
- `protocols/gender/gender_balanced_test_female_60.csv`: 60 utterances
- `protocols/gender/gender_balanced_test_male_60.csv`: 60 utterances

The in-domain protocol uses three source corpora that are also part of the clean-source pool used in the project: DAPS-NII, EARS, and TSP. Each contributes 20 utterances, for 60 clean inputs total. If fully materialized, that becomes `60 * 2520 = 151200` effected files.

The out-of-domain protocol uses VCTK only, with 60 clean utterances from speakers outside the in-domain source pool. If fully materialized, that also becomes `60 * 2520 = 151200` effected files.

The gender-balanced protocol provides two additional 60-utterance test lists, one female and one male, with 20 utterances each from DAPS, EARS, and VCTK. TSP is not used in the gender-balanced split because speaker-gender labels were not available in the project metadata for that corpus.

CSV line counts are one higher than the utterance counts because each file includes a header row.

Placeholder dataset roots used in the published CSVs:

- `/datasets/Adobe_DAPS`
- `/datasets/EARS`
- `/datasets/TSP-Speech-Database`
- `/datasets/CSTR_VCTK-Corpus`

## Reproducibility

The effect-combination protocol is deterministic:

- `main_class_id` is the mixed-radix variant id decoded over preset group sizes `[3, 5, 7, 3, 4, 2]`.
- The last group, `limiter`, changes fastest, matching Python `itertools.product` order in the original generation code.
- Labels come from `config/class_map.csv`.
- Rendering is deterministic by item index, source waveform, and Pedalboard version.
- Fbank extraction uses `dither=0.0`; with `random_crop=False`, evaluation features are deterministic.
- The per-item seed is the original project's stable BLAKE2b hash over `(base_seed, mode_tag, wav_path, variant_id)`.

Run a local check after configuring path remapping:

```bash
python scripts/check_reproducibility.py \
  --dataset-csv protocols/test/vctk_test_sample.csv \
  --path-prefix-map /datasets/CSTR_VCTK-Corpus=/data/CSTR_VCTK-Corpus
```

Important caveat: because the seed includes the waveform path string, deterministic checks are identical only when the same canonical path mapping is used across rendering and evaluation. Use the same `--path-prefix-map` values consistently.

## Source Datasets

VoxEffects is derived from original clean speech from the following corpora. Please obtain them from their original sources and cite them in downstream work as appropriate.

- DAPS:
  Gautham J. Mysore, "Can We Automatically Transform Speech Recorded on Common Consumer Devices in Real-World Environments into Professional Production Quality Speech? A Dataset, Insights, and Challenges," 2015.
  Dataset: [Zenodo](https://doi.org/10.5281/zenodo.4660670)
  Project page: [CCRMA Stanford](https://ccrma.stanford.edu/~gautham/Site/daps.html)
  License: CC BY-NC 4.0
- EARS:
  Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, and Timo Gerkmann, "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation," 2024.
  Dataset: [GitHub](https://github.com/facebookresearch/ears_dataset)
  Paper: [arXiv:2406.06185](https://arxiv.org/abs/2406.06185)
  License: CC BY-NC 4.0
- TSP Speech Database:
  Peter Kabal, "TSP Speech Database," McGill University, 2002.
  Dataset and report: [McGill MMSP](https://www.mmsp.ece.mcgill.ca/Documents/Data/)
  License: Simplified BSD licence
- VCTK:
  Junichi Yamagishi, Christophe Veaux, and Kirsten MacDonald, "CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)," 2019.
  Dataset: [University of Edinburgh DataShare](https://doi.org/10.7488/ds/2645)
  Corpus page: [DataShare handle](https://datashare.ed.ac.uk/handle/10283/3443)
  License: attribution-style corpus terms in the published package
- DEMAND:
  Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent, "The Diverse Environments Multi-Channel Acoustic Noise Database (DEMAND): A database of multichannel environmental noise recordings," 2013.
  Dataset: [Zenodo](https://zenodo.org/records/1227121)
  Dataset index: [DCASE datalist entry](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/demand.html)
  License: CC BY-SA 3.0

## Citation

If you use this program, the VoxEffects database, or any derived audio produced with this code in your research or product, you are required to cite the following paper:

> *VoxEffects: A Speech-Oriented Audio Effects Dataset and Benchmark.* arXiv preprint [arXiv:2604.12389](https://arxiv.org/abs/2604.12389).

BibTeX:

```bibtex
@article{voxeffects2026,
  title  = {VoxEffects: A Speech-Oriented Audio Effects Dataset and Benchmark},
  journal = {arXiv preprint arXiv:2604.12389},
  year   = {2026},
  url    = {https://arxiv.org/abs/2604.12389}
}
```

## License

This repository distributes release metadata, configuration files, and rendering scripts only. It does not redistribute the original source datasets or the fully rendered VoxEffects audio. Users must obtain the original datasets under their respective licenses and render derived audio locally.

The code and release metadata in this repository are licensed under the Modified BSD (3-Clause BSD) License. See `LICENSE`.

The program itself allows commercial use. However, any processed audio created using this program is subject to the license terms of the original audio data you use. For example, if you add effects with this program to audio from DAPS or EARS, which are non-commercial use only (CC BY-NC 4.0), the resulting audio will be governed by the original database license, and the effects-included audio will remain non-commercial use only (CC BY-NC 4.0). In contrast, if you add effects to audio from TSP or VCTK, which allow commercial use (CC BY 4.0), the output audio can be used commercially.

The source datasets used in this work follow their original licenses:

- DAPS follows CC BY-NC 4.0.
- EARS follows CC BY-NC 4.0.
- TSP follows a Simplified BSD licence.
- VCTK uses attribution-style corpus terms in the published package.
- DEMAND follows CC BY-SA 3.0.

Copyright (c) 2026, National Institute of Informatics. All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
