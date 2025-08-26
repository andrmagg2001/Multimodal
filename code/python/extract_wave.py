#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

def load_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load mono audio, resample to target_sr."""
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr

def stft_db(y: np.ndarray, sr: int,
            n_fft: int = 2048, hop: int = 320, win: int = 800,
            top_db: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear-magnitude STFT converted to dB.
    Returns (S_db, freqs_hz, times_s).
    """
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window="hann", center=True)
    S = np.abs(D)  # magnitude
    S_db = librosa.amplitude_to_db(S, ref=np.max)  # <= 0 dB max
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # Hz
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop)
    # Clip dynamic range to top_db (uniform across figures later)
    S_db = np.maximum(S_db, S_db.max() - top_db)
    return S_db, freqs, times

def save_spec_png(S_db: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                  out_path: Path, title: str | None,
                  vmin: float, vmax: float,
                  ymax_hz: float = 16000.0, add_cbar: bool = False, tight: bool = True):
    """Save a single spectrogram PNG with fixed dB scale and 0..ymax_hz limits."""
    fig, ax = plt.subplots(figsize=(9, 3))
    extent = [times[0] if len(times) else 0.0,
              times[-1] if len(times) else 0.0,
              freqs[0], freqs[-1]]
    im = ax.imshow(S_db, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    ax.set_ylim(0, min(ymax_hz, freqs[-1]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title)
    if add_cbar:
        fig.colorbar(im, ax=ax, pad=0.01, label="dB")
    if tight:
        plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_triptych(noisy_db, enh_db, clean_db, freqs, times,
                  out_path: Path, titles=("Noisy","Enhanced","Clean"),
                  vmax: float = 0.0, top_db: int = 80, ymax_hz: float = 16000.0):
    """Save a 1x3 montage with shared dB range and y-limit."""
    vmin = vmax - top_db
    fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharex=False, sharey=True)
    extent = [times[0] if len(times) else 0.0,
              times[-1] if len(times) else 0.0,
              freqs[0], freqs[-1]]

    for ax, S, title in zip(axs, [noisy_db, enh_db, clean_db], titles):
        im = ax.imshow(S, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
        ax.set_ylim(0, min(ymax_hz, freqs[-1]))
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
    axs[0].set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Generate comparable 0–16kHz spectrograms (linear STFT, same dB scale).")
    ap.add_argument("--noisy",     type=Path, default=Path("runs/enhance_demo/audio_noisy.wav"))
    ap.add_argument("--enhanced",  type=Path, default=Path("runs/enhance_demo/audio_enhanced.wav"))
    ap.add_argument("--clean",     type=Path, default=Path("runs/enhance_demo/clean_original.wav"))
    ap.add_argument("--outdir",    type=Path, default=Path("spec_figs"))
    ap.add_argument("--sr",        type=int,  default=32000, help="Resample target SR to cover up to 16 kHz")
    ap.add_argument("--top_db",    type=int,  default=80,    help="Dynamic range (dB) for all plots")
    args = ap.parse_args()

    # Check files
    for p in [args.noisy, args.enhanced, args.clean]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load and compute
    y_noisy, sr = load_mono(args.noisy, args.sr)
    y_enh,  _   = load_mono(args.enhanced, args.sr)
    y_clean, _  = load_mono(args.clean, args.sr)

    S_noisy, freqs, times = stft_db(y_noisy, sr, top_db=args.top_db)
    S_enh,   _,     _     = stft_db(y_enh,  sr, top_db=args.top_db)
    S_clean, _,     _     = stft_db(y_clean, sr, top_db=args.top_db)

    # Shared color scale across all figures: use the global max as 0 dB
    global_max = max(S_noisy.max(), S_enh.max(), S_clean.max())  # should be 0.0
    vmin = global_max - args.top_db
    vmax = global_max

    # Save single PNGs
    save_spec_png(S_noisy, freqs, times, args.outdir / "spec_noisy.png",
                  "Noisy (linear STFT, dB)", vmin, vmax, ymax_hz=16000.0)
    save_spec_png(S_enh, freqs, times, args.outdir / "spec_enhanced.png",
                  "Enhanced (linear STFT, dB)", vmin, vmax, ymax_hz=16000.0)
    save_spec_png(S_clean, freqs, times, args.outdir / "spec_clean.png",
                  "Clean (linear STFT, dB)", vmin, vmax, ymax_hz=16000.0)

    # Save 3-up montage
    save_triptych(S_noisy, S_enh, S_clean, freqs, times,
                  args.outdir / "spec_triptych.png",
                  titles=("Noisy","Enhanced","Clean"),
                  vmax=vmax, top_db=args.top_db, ymax_hz=16000.0)

    print(f"Saved to: {args.outdir.resolve()}")
    print(" - spec_noisy.png")
    print(" - spec_enhanced.png")
    print(" - spec_clean.png")
    print(" - spec_triptych.png")

if __name__ == "__main__":
    main()