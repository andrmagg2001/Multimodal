import argparse
import shutil
import subprocess
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.feature
import librosa.feature.inverse
import soundfile as sf
from moviepy import VideoFileClip, AudioArrayClip  

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print("Using device:", DEVICE)
if DEVICE.type == 'cuda':
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, groups: int = 8):
        super().__init__()
        g = min(groups, out_ch)
        if out_ch % g != 0:
            for gg in range(g, 0, -1):
                if out_ch % gg == 0:
                    g = gg
                    break
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            *([nn.Dropout2d(dropout)] if dropout > 0 else []),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            *([nn.Dropout2d(dropout)] if dropout > 0 else []),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)



class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                 features: list = [32, 64, 128, 256, 512], dropout: float = 0.1):
        super().__init__()
        self.encs, self.pools = nn.ModuleList(), nn.ModuleList()
        prev_ch = in_ch
        for feat in features:
            self.encs.append(DoubleConv(prev_ch, feat, dropout=dropout))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = feat
        self.bottleneck = DoubleConv(prev_ch, prev_ch * 2, dropout=dropout)
        prev_ch *= 2
        self.ups, self.decs = nn.ModuleList(), nn.ModuleList()
        for feat in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(prev_ch, feat, kernel_size=1, bias=False)
            ))
            self.decs.append(DoubleConv(feat * 2, feat, dropout=dropout))
            prev_ch = feat
        self.final = nn.Conv2d(prev_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encs, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.final(x)



class DeepVideoCNN(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 out_features: int = 256, dropout: float = 0.2, groups: int = 8):
        super().__init__()
        def GN(c):
            g = min(groups, c)
            if c % g != 0:
                for gg in range(g, 0, -1):
                    if c % gg == 0:
                        g = gg
                        break
            return nn.GroupNorm(g, c)
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), bias=False),
            GN(base_channels), nn.ReLU(inplace=True), nn.Dropout3d(dropout),
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, stride=(1,2,2), padding=1, bias=False),
            GN(base_channels*2), nn.ReLU(inplace=True), nn.Dropout3d(dropout),
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, stride=1, padding=1, bias=False),
            GN(base_channels*4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(base_channels*4, out_features),
                                nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0,2,1,3,4).contiguous()
        return self.fc(self.conv3d(x))



class HeavyFusionUNet(nn.Module):
    def __init__(self, audio_feat: int = 512, video_feat: int = 256,
                 fusion_size: int = 512, out_shape: tuple = (80,94), dropout: float = 0.3,
                 res_alpha_init: float = 5.0):
        super().__init__()
        self.audio_unet = UNet(in_ch=1, out_ch=audio_feat,
                               features=[32,64,128,256,audio_feat//2], dropout=dropout)
        self.video_cnn  = DeepVideoCNN(in_channels=3, base_channels=32,
                                       out_features=video_feat, dropout=dropout)
        self.fusion     = nn.Sequential(
            nn.Linear(audio_feat+video_feat, fusion_size),
            nn.LayerNorm(fusion_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_size, fusion_size),
            nn.LayerNorm(fusion_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_size, out_shape[0]*out_shape[1])
        )
        self.out_shape = out_shape
        self.res_alpha = nn.Parameter(torch.tensor(res_alpha_init, dtype=torch.float32))

    def forward(self, mel_noisy: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        mu  = mel_noisy.mean(dim=[2,3], keepdim=True)
        std = mel_noisy.std(dim=[2,3], keepdim=True) + 1e-5
        mel_z = (mel_noisy - mu) / std
        a_feat = self.audio_unet(mel_z).mean(dim=[2,3])
        v_feat = self.video_cnn(frames)
        x = torch.cat([a_feat, v_feat], dim=1)
        resid = self.fusion(x).view(-1, self.out_shape[0], self.out_shape[1])
        pred_z = mel_z.squeeze(1) + torch.tanh(resid) * self.res_alpha
        pred   = pred_z * std.squeeze(1) + mu.squeeze(1)
        return pred

#----------------------------------------Function Start----------------------------------------#


def degrade_audio_snr(y: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add white Gaussian noise to an audio signal to achieve a target SNR (in dB).

    Args:
        y: Input audio array, 1D (mono) or 2D (multi-channel). Cast to float32 internally.
        snr_db: Desired signal-to-noise ratio in decibels.

    Returns:
        Noisy audio array (float32) with the same shape as the input.
    """
    y = y.astype(np.float32)
    p_sig = np.mean(y**2) + 1e-12
    p_noise = p_sig / (10.0**(snr_db / 10.0))
    noise = np.random.randn(*y.shape).astype(np.float32) * np.sqrt(p_noise).astype(np.float32)
    return (y + noise).astype(np.float32)




def mel_spectrogram(y: np.ndarray,sr: int, n_mels: int = 80, n_fft: int = 1024, hop_length: int = 160, win_length: int = 400) -> np.ndarray:
    """
    Compute a Mel spectrogram (magnitude) from an audio signal using Slaney-normalized filters.

    Args:
        y: Input audio waveform (mono or multi-channel mixed to the provided array), float-like.
        sr: Sampling rate of the signal in Hz.
        n_mels: Number of Mel bands.
        n_fft: STFT window size.
        hop_length: STFT hop size (in samples).
        win_length: STFT window length (in samples).

    Returns:
        Mel spectrogram array of shape [n_mels, T], non-negative float32 (power=1.0 → magnitude).
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, power=1.0, center=True, norm='slaney'
    )
    return np.maximum(S, 0.0).astype(np.float32)




def mel_to_audio(S: np.ndarray, sr: int,n_fft: int = 1024, hop_length: int = 160, win_length: int = 400, n_iter: int = 48) -> np.ndarray:
    """
    Reconstruct a time-domain waveform from a Mel spectrogram via iterative phase recovery.

    Uses librosa's Mel inversion with Slaney-normalized filters (power=1.0 → magnitude) and
    Griffin–Lim iterations to estimate phase. Very small magnitudes are floored to avoid
    numerical issues.

    Args:
        S: Mel spectrogram of shape [n_mels, T] (magnitude).
        sr: Target sampling rate for the reconstructed audio.
        n_fft: STFT window size used for inversion.
        hop_length: STFT hop size (in samples).
        win_length: STFT window length (in samples).
        n_iter: Number of Griffin–Lim iterations (higher → better quality, slower).

    Returns:
        Reconstructed waveform as float32 array.
    """
    S = np.maximum(S, 1e-8)
    y = librosa.feature.inverse.mel_to_audio(
        M=S, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_iter=n_iter, power=1.0, center=True, norm='slaney'
    )
    return y.astype(np.float32)


def mel_to_audio_noisy_phase(M: np.ndarray, y_noisy: np.ndarray, sr: int, n_fft=1024, hop_length=160, win_length=400) -> np.ndarray:
    """
    Reconstruct a waveform from a Mel spectrogram using the noisy signal's phase.

    The Mel spectrogram M is first projected to a linear-frequency STFT magnitude,
    then combined with the phase of the STFT of `y_noisy`. The result is inverted
    with ISTFT. Time axes are aligned by truncating to the minimum number of frames.

    Args:
        M: Mel spectrogram of shape [n_mels, T] (magnitude, Slaney-normalized, power=1.0).
        y_noisy: Noisy reference waveform used to provide phase.
        sr: Sampling rate (Hz).
        n_fft: STFT window size used for inversion.
        hop_length: STFT hop size (in samples).
        win_length: STFT window length (in samples).

    Returns:
        Reconstructed waveform as float32 array.
    """
    S_mag = librosa.feature.inverse.mel_to_stft(M, sr=sr, n_fft=n_fft, power=1.0, norm='slaney')
    D_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, center=True, window='hann')
    T = min(S_mag.shape[1], D_noisy.shape[1])
    S_mag = S_mag[:, :T]
    phase = np.exp(1j * np.angle(D_noisy[:, :T]))
    y = librosa.istft(S_mag * phase, hop_length=hop_length,
                      win_length=win_length, center=True, window='hann')
    return y.astype(np.float32)



def safe_audio(y: np.ndarray) -> np.ndarray:
    """
    Sanitize and peak-normalize an audio waveform.

    Ensures finite values, scales down if the absolute peak exceeds 1.0,
    and clamps the signal to [-1, 1]. Output is always float32.

    Args:
        y: Input waveform array.

    Returns:
        A float32 waveform safe for writing/muxing (finite, peak-normalized, clipped to [-1, 1]).
    """
    y = y.astype(np.float32)
    if not np.any(np.isfinite(y)):
        return np.zeros_like(y, dtype=np.float32)
    m = float(np.max(np.abs(y)) + 1e-8)
    if m > 1.0:
        y = y / m
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def apply_gain_db(y: np.ndarray, gain_db: float, prevent_clipping: bool = True) -> np.ndarray:
    """
    Apply a gain (in dB) to a waveform with optional peak protection.

    Args:
        y: Input waveform array.
        gain_db: Gain in decibels (positive boosts, negative attenuates).
        prevent_clipping: If True, peak-normalizes the result when |peak| > 1.0.

    Returns:
        Float32 waveform after gain and optional peak normalization.
    """
    if y is None or y.size == 0:
        return np.array([], dtype=np.float32)
    if abs(gain_db) < 1e-8:
        return y.astype(np.float32)
    g = 10.0 ** (gain_db / 20.0)
    y_out = y.astype(np.float32) * g
    if prevent_clipping:
        peak = float(np.max(np.abs(y_out)) + 1e-8)
        if peak > 1.0:
            y_out /= peak
    return y_out.astype(np.float32)



def build_mel_fbank(sr, n_fft, n_mels=80):
    """
    Construct a Mel filterbank (Slaney-style) and row-normalize it.

    Args:
        sr: Sampling rate in Hz.
        n_fft: STFT FFT size (determines number of linear freq bins: n_fft//2 + 1).
        n_mels: Number of Mel bands.

    Returns:
        Float32 filterbank matrix of shape (n_mels, n_fft//2 + 1), where each row
        sums to 1 (useful for stable Mel-domain masking/projection).
    """
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, norm='slaney', dtype=np.float32)
    fb = fb / (fb.sum(axis=1, keepdims=True) + 1e-8)
    return fb  


def mel_mask_from_pred(S_pred_mel, D_noisy, sr, n_fft, lam=1e-2):
    """
    Project a predicted Mel spectrogram to a linear-frequency magnitude mask.

    Given a predicted Mel magnitude S_pred_mel [n_mels, T] and the complex noisy
    STFT D_noisy [n_fft//2+1, T], this computes a per-bin mask M by:
      (1) mel_noisy = F @ |D_noisy|
      (2) Mel-domain gain G_mel = S_pred_mel / (mel_noisy + eps), clipped to [0, 3]
      (3) Linear projection with Tikhonov regularization:
          M = F^T (F F^T + λI)^{-1} G_mel
    The result is clipped to [0, 3] for stability.

    Args:
        S_pred_mel: Predicted Mel magnitudes, shape (n_mels, T).
        D_noisy: Complex noisy STFT, shape (n_fft//2 + 1, T).
        sr: Sampling rate in Hz.
        n_fft: FFT size used for STFT/Mel filterbank.
        lam: Tikhonov regularization constant (λ).

    Returns:
        Float32 mask M of shape (n_fft//2 + 1, T), non-negative and clipped.
    """
    fb = build_mel_fbank(sr, n_fft, n_mels=S_pred_mel.shape[0])            
    mag_noisy = np.abs(D_noisy)
    mel_noisy = fb @ mag_noisy
    G_mel = S_pred_mel / (mel_noisy + 1e-6)
    G_mel = np.clip(G_mel, 0.0, 3.0)
    fb_T = fb.T                                                            
    A = fb @ fb_T + lam * np.eye(fb.shape[0], dtype=np.float32)           
    A_inv = np.linalg.inv(A)
    proj = fb_T @ (A_inv @ G_mel)                                          
    M = np.clip(proj, 0.0, 3.0).astype(np.float32)
    return M


def temporal_smooth(S, k=3):
    """
    Apply temporal moving-average smoothing along the time axis.

    A length-k uniform filter is applied independently to each frequency bin
    of a 2D array S with shape (F, T). Padding uses edge replication so the
    output keeps the same time length. For k <= 1, the input is returned
    unchanged (cast to float32).

    Args:
        S (np.ndarray): Spectrogram-like array of shape (F, T).
        k (int): Window size in frames for the moving average.

    Returns:
        np.ndarray: Smoothed array of shape (F, T), dtype float32.
    """
    k = int(k)
    if k <= 1:
        return S.astype(np.float32)
    pad = k // 2
    kernel = np.ones(k, dtype=np.float32) / float(k)
    S_pad = np.pad(S, ((0, 0), (pad, pad)), mode='edge')
    S_smooth = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='valid'), 1, S_pad)
    return S_smooth.astype(np.float32)



def mel_lows_cut(S: np.ndarray, sr: int, cut_db: float = 3.0, corner_hz: float = 180.0, slope: float = 1.0) -> np.ndarray:
    """
    Apply a Mel-domain low-frequency attenuation (low-cut) below a corner frequency.

    A per-Mel-band gain is computed from the bands' center frequencies and
    applied to S (shape: [n_mels, T]). Bands below `corner_hz` are attenuated
    up to `cut_db` dB at the lowest frequencies. The `slope` (> = 1) controls
    how steeply the attenuation ramps toward low frequencies.

    Args:
        S (np.ndarray): Mel spectrogram array of shape (n_mels, T).
        sr (int): Sample rate used to interpret Mel band centers (Hz).
        cut_db (float): Maximum attenuation at low frequencies (in dB).
                        If <= 0, the input is returned unchanged.
        corner_hz (float): Corner frequency (Hz) where attenuation transitions.
        slope (float): Shape parameter (>=1) for the attenuation curve
                       (higher values = steeper roll-off).

    Returns:
        np.ndarray: Low-cut Mel spectrogram, same shape as S, dtype float32.
    """
    if cut_db <= 0:
        return S.astype(np.float32)
    f = librosa.mel_frequencies(n_mels=S.shape[0], fmin=0.0, fmax=sr/2.0).astype(np.float32)
    x = np.clip((corner_hz - f) / max(1.0, corner_hz), 0.0, 1.0)  
    x = x ** max(1.0, float(slope))
    gain_db = -abs(cut_db) * x
    gain = 10.0 ** (gain_db / 20.0)
    return (S * gain[:, None]).astype(np.float32)


def mel_presence_boost(S: np.ndarray, sr: int, center_hz: float = 3500.0, bw_hz: float = 1800.0, gain_db: float = 3.0) -> np.ndarray:
    """
    Apply a Gaussian-shaped presence boost in the Mel domain.

    Args:
        S (np.ndarray): Mel spectrogram of shape (n_mels, T).
        sr (int): Sample rate used to derive Mel bin center frequencies.
        center_hz (float): Center frequency of the boost (Hz).
        bw_hz (float): Approximate full width at half maximum (FWHM) of the boost (Hz).
        gain_db (float): Peak gain at the center (dB).

    Returns:
        np.ndarray: Spectrogram with per-bin gains applied (same shape, float32).
    """
    if abs(gain_db) <= 1e-8 or bw_hz <= 0:
        return S.astype(np.float32)
    f = librosa.mel_frequencies(n_mels=S.shape[0], fmin=0.0, fmax=sr/2.0).astype(np.float32)
    sigma = bw_hz / 2.355  # FWHM -> sigma
    w = np.exp(-0.5 * ((f - center_hz) / max(1.0, sigma))**2).astype(np.float32)
    gain = 10.0 ** ((gain_db * w) / 20.0)
    return (S * gain[:, None]).astype(np.float32)

def mel_highs_boost(S: np.ndarray, sr: int, amount_db: float = 4.0, corner_hz: float = 3000.0, slope: float = 1.0) -> np.ndarray:
    """
    Apply a Mel-domain high-shelf boost above a corner frequency.

    A per-Mel-band gain is computed from band center frequencies and applied to
    S (shape: [n_mels, T]). Bands above `corner_hz` are amplified up to
    `amount_db` dB at the highest frequencies. The `slope` (>= 1) controls how
    quickly the boost ramps in toward high frequencies.

    Args:
        S (np.ndarray): Mel spectrogram of shape (n_mels, T).
        sr (int): Sample rate used to compute Mel band centers (Hz).
        amount_db (float): Maximum high-frequency boost in dB.
                           If <= 0, S is returned unchanged.
        corner_hz (float): Corner frequency (Hz) where the shelf begins.
        slope (float): Shape parameter (>= 1) for the boost curve
                       (higher = steeper transition).

    Returns:
        np.ndarray: High-shelf boosted Mel spectrogram, same shape as S, float32.
    """
    if amount_db <= 0:
        return S.astype(np.float32)
    f_centers = librosa.mel_frequencies(n_mels=S.shape[0], fmin=0.0, fmax=sr/2.0).astype(np.float32)
    fmax = float(sr) / 2.0
    x = np.clip((f_centers - corner_hz) / max(1.0, (fmax - corner_hz)), 0.0, 1.0)
    x = x ** max(1.0, float(slope))
    gain_db = amount_db * x
    gain = 10.0 ** (gain_db / 20.0)
    return (S * gain[:, None]).astype(np.float32)



def extract_frames_window(video_path: str, t0: float, t1: float, n_frames=12, size=(112,112)) -> np.ndarray:
    """
    Extract `n_frames` RGB frames uniformly from the time window [t0, t1] of a video.

    Frames are resized to `size` (W, H), converted from BGR to RGB, normalized to [0, 1],
    laid out channel-first, and returned as float32 with shape (n_frames, 3, H, W).
    If the video cannot be opened (or a specific frame cannot be decoded), zeros are returned
    for those frames. When `t1 <= t0`, the function falls back to a minimal span.

    Args:
        video_path (str): Path to the input video file.
        t0 (float): Start time in seconds (inclusive).
        t1 (float): End time in seconds (inclusive upper bound; if <= t0, a minimal range is used).
        n_frames (int, optional): Number of frames to sample uniformly. Defaults to 12.
        size (tuple[int, int], optional): Output frame size as (W, H). Defaults to (112, 112).

    Returns:
        np.ndarray: Array of shape (n_frames, 3, H, W), dtype float32, values in [0, 1].
                    Zeros are returned if the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        C = 3; H, W = size[1], size[0]
        return np.zeros((n_frames, C, H, W), dtype=np.float32)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    start_idx = max(0, int(t0 * fps))
    end_idx   = min(total-1, int(t1 * fps)) if t1 > 0 else total-1
    if end_idx <= start_idx:
        end_idx = min(total-1, start_idx + 1)
    idxs = np.linspace(start_idx, end_idx, n_frames).astype(int)
    H, W = size[1], size[0]
    out = np.zeros((n_frames, 3, H, W), dtype=np.float32)
    for i, fidx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        out[i] = (np.transpose(frame, (2,0,1)) / 255.0).astype(np.float32)
    cap.release()
    return out


def slice_mel_windows(S: np.ndarray, win_w=94, hop_w=47):
    """
    Split a Mel spectrogram into fixed-width, overlapping time windows.

    The input S has shape (n_mels, T). Windows are extracted every `hop_w`
    frames with width `win_w`. The last window is always included: if the tail
    is shorter than `win_w`, the patch is right-padded with zeros so that all
    returned patches have shape (n_mels, win_w).

    Args:
        S (np.ndarray): Mel spectrogram of shape (n_mels, T).
        win_w (int): Window width in time frames (default: 94).
        hop_w (int): Hop size in time frames between consecutive windows (default: 47).

    Returns:
        tuple[list[tuple[int, np.ndarray]], int]:
            - windows: list of (start_index, patch) where `start_index` is the
              starting frame in S and `patch` is a zero-padded array of shape
              (n_mels, win_w).
            - T: original total number of time frames in S.
    """
    T = S.shape[1]
    starts = list(range(0, max(T - win_w + 1, 1), hop_w))
    if not starts or (starts[-1] + win_w < T):
        starts.append(max(T - win_w, 0))
    windows = []
    for s in starts:
        patch = np.zeros((S.shape[0], win_w), dtype=np.float32)
        e = min(s + win_w, T)
        w = e - s
        patch[:, :w] = S[:, s:e]
        windows.append((s, patch))
    return windows, T

def stitch_mel_windows(windows, T_total, win_w=94, hop_w=47):
    """
    Reconstruct a full-length Mel spectrogram by overlap-adding fixed-width windows.

    Args:
        windows (list[tuple[int, np.ndarray]]): Sequence of (start_idx, patch),
            where `start_idx` is the time-frame offset in the target spectrogram,
            and `patch` has shape (80, win_w) (n_mels assumed to be 80).
        T_total (int): Total number of time frames of the target spectrogram.
        win_w (int): Window width (frames) used during slicing (default: 94).
        hop_w (int): Hop size (frames) used during slicing (default: 47).

    Returns:
        np.ndarray: Reconstructed Mel spectrogram of shape (80, T_total).
    """
    acc = np.zeros((80, T_total), dtype=np.float32)
    weight = np.zeros((1, T_total), dtype=np.float32)
    if hop_w * 2 == win_w:
        w = np.sqrt(np.hanning(win_w).astype(np.float32))
    else:
        w = np.hanning(win_w).astype(np.float32)
    w = np.maximum(w, 1e-6)
    for s, patch in windows:
        e = min(s + win_w, T_total)
        L = e - s
        acc[:, s:e]    += (patch[:, :L] * w[:L][None, :])
        weight[:, s:e] += w[:L]
    return acc / np.maximum(weight, 1e-6)


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg and try again.")

def mux_single_audio(in_video: str, in_wav: str, out_path: Path, audio_bitrate="256k", container="mkv", title=None, debug=False):
    """
    Mux a single WAV track into a video using ffmpeg, stream-copying the video
    and encoding the audio to AAC.

    Args:
        in_video (str): Path to the input video file.
        in_wav (str): Path to the input WAV audio to be muxed (replaces the video's audio).
        out_path (Path): Output path *without* extension; the `container` suffix is appended.
        audio_bitrate (str): Target AAC audio bitrate (e.g., "192k", "256k"). Default: "256k".
        container (str): Output container format ("mkv" or "mp4"). Default: "mkv".
        title (str | None): Optional audio stream title metadata.
        debug (bool): If True, print the constructed ffmpeg command.

    Returns:
        Path: Full path to the created output file (with container extension).

    Raises:
        RuntimeError: If ffmpeg is not found in PATH.
        subprocess.CalledProcessError: If the ffmpeg process fails.
    """
    check_ffmpeg()
    out_path = out_path.with_suffix("." + container)
    cmd = [
        "ffmpeg", "-y",
        "-i", in_video,
        "-i", in_wav,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-shortest"
    ]
    if title:
        cmd += ["-metadata:s:a:0", f"title={title}"]
    cmd += [str(out_path)]
    if debug:
        print("FFmpeg cmd:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def process_video(in_path: str, ckpt_path: str, outdir: str, snr_db: float = 0.0,
                  n_mels=80, n_fft=1024, hop_length=160, win_length=400,
                  win_w=94, hop_w=47, frames_n=12, frame_size=(112,112),
                  resample_sr: int | None = 16000, recon_mode: str = "mask",
                  smooth_t: int = 3, audio_bitrate: str = "256k", gain_db: float = 0.0,
                  container: str = "mkv",
                  lf_cut_db: float = 3.0, lf_corner_hz: float = 180.0, lf_slope: float = 1.0,
                  presence_gain_db: float = 2.5, presence_hz: float = 3500.0, presence_bw_hz: float = 1800.0,
                  hf_gain_db: float = 4.0, hf_corner_hz: float = 3000.0, hf_slope: float = 1.0):
    in_path = str(in_path)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)


    vclip = VideoFileClip(in_path)
    if vclip.audio is None:
        raise RuntimeError("No audio file in the video.")
    sr_src = int(vclip.audio.fps)
    y = vclip.audio.to_soundarray(fps=sr_src)
    y_mono = y.mean(axis=1).astype(np.float32) if y.ndim == 2 else y.astype(np.float32).flatten()

    if resample_sr and resample_sr != sr_src:
        y_proc = librosa.resample(y_mono, orig_sr=sr_src, target_sr=resample_sr)
        sr = int(resample_sr)
    else:
        y_proc = y_mono
        sr = sr_src

    y_noisy = degrade_audio_snr(y_proc, snr_db=snr_db)
    if sr != sr_src:
        y_noisy_out = librosa.resample(y_noisy, orig_sr=sr, target_sr=sr_src)
    else:
        y_noisy_out = y_noisy
    y_noisy_out = safe_audio(y_noisy_out)
    noisy_wav = outdir / "audio_noisy.wav"
    sf.write(noisy_wav, y_noisy_out, sr_src)

    model = HeavyFusionUNet().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    S_noisy = mel_spectrogram(y_noisy, sr=sr, n_mels=n_mels, n_fft=n_fft,
                              hop_length=hop_length, win_length=win_length)
    win_seconds = (win_w * hop_length) / float(sr)
    duration = float(vclip.duration)

    mel_windows, T_total = slice_mel_windows(S_noisy, win_w=win_w, hop_w=hop_w)
    pred_windows = []
    for t_start, patch in mel_windows:
        t0 = t_start * (hop_length / float(sr))
        t1 = min(t0 + win_seconds, duration)
        frames_np = extract_frames_window(in_path, t0, t1, n_frames=frames_n, size=frame_size)
        mel_in    = torch.from_numpy(patch[None, None, :, :]).float().to(DEVICE)
        frames_in = torch.from_numpy(frames_np[None, ...]).float().to(DEVICE)
        with torch.no_grad():
            if DEVICE.type == 'cuda':
                with torch.autocast("cuda", enabled=True):
                    pred_mel = model(mel_in, frames_in)
            else:
                pred_mel = model(mel_in, frames_in)
        pred_windows.append((t_start, pred_mel.squeeze(0).cpu().numpy()))

    S_pred = stitch_mel_windows(pred_windows, T_total, win_w=win_w, hop_w=hop_w)
    if smooth_t and smooth_t > 1:
        S_pred = temporal_smooth(S_pred, k=int(smooth_t))

    if lf_cut_db > 0.0:
        S_pred = mel_lows_cut(S_pred, sr=sr, cut_db=lf_cut_db,
                              corner_hz=lf_corner_hz, slope=lf_slope)
    if abs(presence_gain_db) > 1e-8:
        S_pred = mel_presence_boost(S_pred, sr=sr, center_hz=presence_hz,
                                    bw_hz=presence_bw_hz, gain_db=presence_gain_db)
    if hf_gain_db > 0.0:
        S_pred = mel_highs_boost(S_pred, sr=sr, amount_db=hf_gain_db,
                                 corner_hz=hf_corner_hz, slope=hf_slope)

    if recon_mode == "mask":
        D_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length,
                               win_length=win_length, center=True, window='hann')
        M = mel_mask_from_pred(S_pred, D_noisy, sr=sr, n_fft=n_fft)
        mag_enh = np.abs(D_noisy) * M
        y_pred = librosa.istft(mag_enh * np.exp(1j * np.angle(D_noisy)),
                               hop_length=hop_length, win_length=win_length,
                               center=True, window='hann').astype(np.float32)
    elif recon_mode == "noisy_phase":
        y_pred = mel_to_audio_noisy_phase(S_pred, y_noisy, sr=sr,
                                          n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    elif recon_mode == "griffinlim":
        y_pred = mel_to_audio(S_pred, sr=sr, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, n_iter=48)
    else:
        raise ValueError(f"Unknown recon_mode: {recon_mode}")

    if sr != sr_src:
        y_pred_out = librosa.resample(y_pred, orig_sr=sr, target_sr=sr_src)
    else:
        y_pred_out = y_pred
    if abs(gain_db) > 1e-8:
        y_pred_out = apply_gain_db(y_pred_out, gain_db, prevent_clipping=True)
    y_pred_out = safe_audio(y_pred_out)
    enhanced_wav = outdir / "audio_enhanced.wav"
    sf.write(enhanced_wav, y_pred_out, sr_src)

    stem = Path(in_path).stem
    noisy_out = mux_single_audio(
        in_video=in_path,
        in_wav=str(noisy_wav),
        out_path=outdir / f"{stem}_noisy",
        audio_bitrate=audio_bitrate,
        container=container,
        title="Noisy"
    )
    print("Video with NOISY audio   ->", noisy_out)

    enhanced_out = mux_single_audio(
        in_video=in_path,
        in_wav=str(enhanced_wav),
        out_path=outdir / f"{stem}_enhanced",
        audio_bitrate=audio_bitrate,
        container=container,
        title="Enhanced"
    )
    print("Video with ENHANCED audio ->", enhanced_out)

    vclip.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",   type=str, required=True, help="Input video path (mpg/mp4)")                                        # Path to source video
    ap.add_argument("--ckpt",    type=str, required=True, help="Trained checkpoint (e.g., best_model.pth)")                         # Model path
    ap.add_argument("--outdir",  type=str, default="runs/enhance_video")                                                            # Where resul are saved
    ap.add_argument("--snr_db",  type=float, default=0.0, help="Degradation SNR in dB (0 = very noisy)")                            # Noise added for the noisy reference
    ap.add_argument("--n_mels",  type=int, default=80)                                                                              # Mel bins
    ap.add_argument("--n_fft",   type=int, default=1024)                                                                            # Frequency resolution
    ap.add_argument("--hop_length", type=int, default=160)                                                                          # Frame step
    ap.add_argument("--win_length", type=int, default=400)                                                                          # Frame size
    ap.add_argument("--win_w",   type=int, default=94)                                                                              # Patch width fed to the network    
    ap.add_argument("--hop_w",   type=int, default=47)                                                                              # Patch stride
    ap.add_argument("--frames_n",   type=int, default=12)                                                                           # Visual context length
    ap.add_argument("--frame_size", type=int, nargs=2, default=[112,112])                                                           # Resize target
    ap.add_argument("--resample_sr", type=int, default=16000)                                                                       # Internal SR
    ap.add_argument("--recon", type=str, choices=["mask", "noisy_phase", "griffinlim"], default="mask")                             # How audio is rebuilt from Mel
    ap.add_argument("--smooth_t", type=int, default=3)                                                                              # Mel smoothing length
    ap.add_argument("--audio_bitrate", type=str, default="256k")                                                                    # Muxed audio quality
    ap.add_argument("--gain_db", type=float, default=0.0)                                                                           # Makeup gain
    ap.add_argument("--container", type=str, choices=["mkv", "mp4"], default="mkv")                                                 # Video file extension
    ap.add_argument("--lf_cut_db", type=float, default=3.0, help="Low-frequency cut amount (max dB) below the corner frequency")    # LF attenuation
    ap.add_argument("--lf_corner_hz", type=float, default=180.0, help="Low-cut corner frequency (Hz)")                              # Start of LF roll-off
    ap.add_argument("--lf_slope", type=float, default=1.0, help="Low-cut slope (>=1 is steeper)")                                   # Steepness control
    ap.add_argument("--presence_gain_db", type=float, default=2.5, help="Presence band boost (dB)")                                 # Boost amount
    ap.add_argument("--presence_hz", type=float, default=3500.0, help="Presence band center frequency (Hz)")                        # Center of bell
    ap.add_argument("--presence_bw_hz", type=float, default=1800.0, help="Presence band width ~FWHM (Hz)")                          # Bell bandwidth
    ap.add_argument("--hf_gain_db", type=float, default=4.0, help="High-shelf boost (dB)")                                          # Amount of HF lift
    ap.add_argument("--hf_corner_hz", type=float, default=3000.0, help="High-shelf corner frequency (Hz)")                          # Shelf start frequency
    ap.add_argument("--hf_slope", type=float, default=1.0, help="High-shelf slope (>=1 is steeper)")                                # Shelf steepness
    args = ap.parse_args()

    process_video(
        in_path=args.video, ckpt_path=args.ckpt, outdir=args.outdir, snr_db=args.snr_db,
        n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,
        win_w=args.win_w, hop_w=args.hop_w, frames_n=args.frames_n, frame_size=tuple(args.frame_size),
        resample_sr=args.resample_sr, recon_mode=args.recon, smooth_t=args.smooth_t,
        audio_bitrate=args.audio_bitrate, gain_db=args.gain_db, container=args.container,
        lf_cut_db=args.lf_cut_db, lf_corner_hz=args.lf_corner_hz, lf_slope=args.lf_slope,
        presence_gain_db=args.presence_gain_db, presence_hz=args.presence_hz, presence_bw_hz=args.presence_bw_hz,
        hf_gain_db=args.hf_gain_db, hf_corner_hz=args.hf_corner_hz, hf_slope=args.hf_slope
    )

if __name__ == "__main__":
    main()