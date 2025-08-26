# LipSound: Multimodal Speech Enhancement via Audio and Visual Cues 🎥🗣️

LipSound is a research project focused on multimodal speech enhancement by leveraging both audio and visual information. The goal is to improve the quality and intelligibility of speech signals that have been degraded by noise, using cues not only from the audio signal but also from the speaker's visual lip movements.

This project employs a deep learning pipeline that processes synchronized video and audio data to perform speech denoising. The visual modality provides complementary information—such as lip movement patterns—that helps the model to recover missing or masked speech content, especially in noisy environments where traditional audio-only methods often struggle.

The core contributions and features of this project are:

- **Multimodal architecture:** Combines Mel‑spectrogram features from noisy audio with visual features extracted from video frames for enhanced speech restoration.
- **Offline feature extraction:** All audio is preprocessed into Mel‑spectrograms, supporting fast and efficient dataloading during model training.
- **Consistent data splits:** Audio, noisy audio, and video files are split identically into train/val/test subsets, ensuring perfect alignment across modalities.
- **Extensibility:** The modular data structure and scripts allow easy addition of new speakers, augmentation, or extension to other multimodal tasks.

> **Important:** to run the full end‑to‑end demo, simply execute:
>
> ```bash
> ./start.sh
> ```
> Make sure the script is executable (`chmod +x start.sh`). The script will sanitize the input video if needed, generate the noisy/enhanced audio, and mux two output videos for A/B comparison.

---

## Dataset

The GRID Corpus is used for training and evaluation. It contains 34 speakers with ~1,000 short utterances each. Data are organized per speaker under `data/dataset/` with precomputed features and a small on‑disk frame cache.

### Structure

```
data/
├── dataset/
│   ├── s1_processed/
│   │   ├── audio/
│   │   │   ├── train/           # .wav 16 kHz mono (clean)
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── audio_noisy/
│   │   │   ├── train/           # degraded .wav (controlled SNR)
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── audio_mel/
│   │   │   ├── train/           # .npy (Mel clean, 80×T)
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── audio_noisy_mel/
│   │   │   ├── train/           # .npy (Mel noisy, 80×T)
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── videos/
│   │   │   ├── train/           # .mpg / .mp4 (≈25 fps)
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── frames_cache/        # cached RGB frames (auto‑generated)
│   ├── s2_processed/
│   ├── s3_processed/
│   └── ...
├── model/
│   ├── best_model.pth
│   ├── checkpoint_epoch_5.pth
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_epoch_15.pth
│   ├── checkpoint_epoch_20.pth
│   ├── checkpoint_epoch_25.pth
│   └── checkpoint_epoch_30.pth
└── runs/
    └── enhance_demo/
        ├── audio_noisy.wav
        ├── audio_enhanced.wav
        ├── <stem>_noisy.mp4
        └── <stem>_enhanced.mp4
```

**Notes**
- The same file *stem* is kept across `audio/`, `audio_noisy/`, and `videos/`, and splits are identical across modalities.
- `frames_cache/` is populated automatically to avoid re‑decoding frames on each epoch.
- No forced alignments are required for enhancement in this project.

---

## Preprocessing (summary)

1. **Extract audio** from each GRID video → save mono 16 kHz `.wav` in `audio/`.
2. **Degrade** each clean file with controlled SNR → save in `audio_noisy/`.
3. **Compute Mel features** (80 filters, `n_fft=1024`, hop `160`, window `400`) → save `.npy` in `audio_mel/` and `audio_noisy_mel/`.
4. **Keep split parity** across audio/noisy/video (`train/val/test`).

---

## Model

- **HeavyFusionUNet**
  - *Audio branch*: U‑Net on Mel patches \(1×80×94\) with residual prediction on per‑sample z‑scored inputs and a learnable scale.
  - *Video branch*: lightweight 3D‑CNN on 12 RGB frames \(112×112\).
  - *Fusion*: two‑layer MLP on concatenated embeddings → residual \(80×94\) (tanh‑limited) → denormalization.
- **Loss**: Charbonnier + \(0.05\times\) MSE; gradient clipping with max‑norm 5.0.
- **Optimization**: AdamW (lr \(3e{-4}\), wd \(1e{-4}\)) + ReduceLROnPlateau.
- **Windowing**: Mel windows \(80×94\) with hop \(=47\); overlap–add at reconstruction.

---

## Training

Training is handled from the provided notebook/cells. Default batch size is 16, target of 40 epochs with validation each epoch and checkpointing every 5 epochs. In the reported run, early stopping triggered at epoch 30 and the best validation model was saved to `model/best_model.pth`.

---

## Inference

Two ways to run inference:

### 1) One‑liner (recommended)
Use the launcher script:
```bash
./start.sh
```
It sanitizes the input (if needed), runs enhancement, and writes two synchronized videos:
`<stem>_noisy.mp4` and `<stem>_enhanced.mp4`, plus `audio_noisy.wav` and `audio_enhanced.wav`.

### 2) Direct CLI
Call the Python entrypoint manually, e.g.:
```bash
pip install -r requirements.txt


echo "Start enhancement..."

python code/python/test.py \
  --video data/dataset/s1_processed/videos/test/bbal8p.mpg \
  --ckpt  data/model/best_model.pth \
  --outdir runs/enhance_demo \
  --snr_db 8 \
  --recon mask \
  --smooth_t 3 \
  --hf_gain_db 5 \
  --hf_corner_hz 2500 \
  --hf_slope 1.2 \
  --audio_bitrate 256k \
  --container mp4 \
  --gain_db 25 \
  --lf_cut_db 4 --lf_corner_hz 160 --lf_slope 1.2 \
  --presence_gain_db 3 --presence_hz 3500 --presence_bw_hz 2000 \
  --hf_gain_db 5 --hf_corner_hz 3000 --hf_slope 1.1

echo "Done!\n";

echo "Extracting wave...";

python code/python/extract_wave.py \
  --noisy runs/enhance_demo/audio_noisy.wav \
  --enhanced runs/enhance_demo/audio_enhanced.wav \
  --clean runs/enhance_demo/clean_original.wav \
  --outdir spec_figs
```

---

## Testing & Validation

- Dataloader shapes: Mel `[B, 1, 80, 94]`, video `[B, F, 3, 112, 112]` with `F=12`.
- Forward pass of `HeavyFusionUNet` returns `[B, 80, 94]`.
- Device handling unified via `DEVICE` (CPU / Apple M‑series MPS / CUDA).

---

## Requirements

- Python 3.10+
- `torch`, `numpy`, `librosa`, `soundfile`, `opencv-python`
- `moviepy>=2.0`, `imageio-ffmpeg` (and **ffmpeg** available in `PATH`)

---

## Troubleshooting

- **HEVC/HDR phone videos**: if demuxing fails, pre‑sanitize with ffmpeg (H.264 + `-pix_fmt yuv420p`) or let the launcher handle it.
- **Audio sample‑rate**: processing uses 16 kHz internally; output is resampled back to the source rate before muxing.
- **Low volume**: a light peak normalization is applied; additional gain can be added in the CLI or post‑processing if desired.
