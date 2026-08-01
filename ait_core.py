"""ait_core — AirHop v3.3.4 physics, extracted for standalone deployment.

GENERATED — do not hand-edit. Source: 03_eval_and_visualize.ipynb from the
verified handoff zip (payload hashes 24/24 OK, 2026-07-18).

SHARED PHYSICS fingerprint: 41c48d3150865466  (7 cells, byte-identical across NB 01/02/03)
Capture-gate helper sha256[:16]: 57bd9c009cd91bf7

The golden parity gate (run_parity_gate) is the acceptance test for this
module on any new machine: payload_scale 587.0, band 6000-8988 Hz, PSNR
30.19 / 40.07 clean and 16.01 under the impairment stack, tol +/-0.6 dB.
NEVER trust this module on a platform where the gate has not passed.
"""

# === SHARED PHYSICS CELL: imports + format constant ===
# Byte-identical in 01 / 02 / 03. If you edit it here, copy it to the
# other two notebooks; the parity gate below pins its behavior.
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import scipy.io.wavfile
import scipy.ndimage as ndi
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from tqdm.auto import tqdm

FORMAT_VERSION = "2.1-airhop"


# === SHARED PHYSICS CELL: encoder ===
# Byte-identical in 01 / 02 / 03. If you edit it here, copy it to the
# other two notebooks; the parity gate below pins its behavior.
class EncodeError(RuntimeError):
    """encode_dir failed on >=1 file. .offenders = limiter cases (swap in
    reserve images, re-encode ALL dirs); .failures = generic errors."""

    def __init__(self, offenders, failures):
        self.offenders, self.failures = offenders, failures
        super().__init__(f"{len(offenders)} limiter offender(s), "
                         f"{len(failures)} failed file(s)")

class LimiterExceeded(RuntimeError):
    """Payload peak above limiter_threshold: encoding this image at the
    global payload_scale would require a per-file rescale the receiver
    cannot recover. Raised instead of silently dimming."""

    def __init__(self, peak, threshold):
        self.peak = float(peak)
        self.threshold = float(threshold)
        super().__init__(
            f"pre-limiter payload peak {peak:.3f} exceeds threshold {threshold:.2f}; "
            f"decoded brightness would be silently scaled by {threshold / peak:.3f}. "
            f"Exclude this image or set payload_scale_override (then re-encode everything).")

class EncoderConfig:
    def __init__(self):
        self.input_dir = "IMG"
        self.output_dir = "WAV_CUSTOM"
        self.meta_dir = "WAV_CUSTOM_META"

        self.target_size = 256
        self.sample_rate = 48000
        self.f_min = 6000.0
        self.f_max = 9000.0

        self.n_fft = 4096
        self.win_length = 4096
        self.hop_length = 1024

        self.add_preamble = True
        self.preamble_seconds = 0.15
        self.chirp_amplitude = 0.6
        self.chirp_fade_ms = 2.0

        self.add_tail_chirp = True          # down-chirp for stretch estimation
        self.guard_seconds = 0.03           # guard after payload AND tail chirp

        self.fade_ms = 10.0
        # The all-white anchor is NOT a strict worst case (a selected-rows
        # binary image reaches ~4.3x its peak), so the limiter FAILS LOUDLY
        # by default instead of silently rescaling brightness the receiver
        # can never know about.
        self.payload_scale_override = None
        self.limiter_threshold = 0.95
        self.allow_limiter = False   # True = legacy warn+rescale (uncalibrated)
        self.phase_seed = 1234
        self.gamma = 1.0

        self.window = "hamming"

def ensure_empty_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def make_chirp(cfg, f0, f1):
    """Linear chirp f0 -> f1 over preamble_seconds, with short edge fades.
    Up-chirp: f0 < f1 (preamble). Down-chirp: f0 > f1 (tail). Amplitude 1.0;
    caller applies cfg.chirp_amplitude. The receiver rebuilds the identical
    template (fades included) from metadata, so keep this function in sync."""
    n = int(cfg.preamble_seconds * cfg.sample_rate)
    t = np.arange(n, dtype=np.float32) / cfg.sample_rate
    k = (f1 - f0) / max(cfg.preamble_seconds, 1e-9)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    x = np.sin(phase).astype(np.float32)
    fade = int(cfg.chirp_fade_ms * 1e-3 * cfg.sample_rate)
    if fade > 1 and x.size > 2 * fade:
        x[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        x[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return x

def fade_in_out(x, cfg):
    fade_len = int(cfg.fade_ms * 1e-3 * cfg.sample_rate)
    if fade_len <= 1 or x.size < 2 * fade_len:
        return x
    w_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    w_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    y = x.copy()
    y[:fade_len] *= w_in
    y[-fade_len:] *= w_out
    return y

_PAYLOAD_SCALE_CACHE = {}

def compute_payload_scale(cfg, idx_min, idx_max_excl, target_peak=0.5):
    """One global, content-independent output scale: the all-white image's
    peak is mapped to target_peak (0.5 leaves ~6 dB margin; the per-file
    limiter remains as a backstop). istft's 1/N scaling otherwise leaves the
    payload ~70 dB below the chirps — invisible in digital loopback, fatal
    over the air. Recorded in metadata as payload_scale."""
    key = (cfg.sample_rate, cfg.n_fft, cfg.win_length, cfg.hop_length,
           cfg.phase_seed, idx_min, idx_max_excl, cfg.target_size, target_peak)
    if key in _PAYLOAD_SCALE_CACHE:
        return _PAYLOAD_SCALE_CACHE[key]
    ones = np.full((cfg.target_size, cfg.target_size), 255, dtype=np.uint8)
    payload, _ = _synth_payload(ones, cfg, idx_min, idx_max_excl)
    scale = float(target_peak / (np.max(np.abs(payload)) + 1e-12))
    _PAYLOAD_SCALE_CACHE[key] = scale
    return scale

def _synth_payload(gray_img_2d, cfg, idx_min, idx_max_excl):
    """Core STFT-domain synthesis shared by encoding and scale calibration."""
    H, W = gray_img_2d.shape
    A = gray_img_2d.astype(np.float32) / 255.0
    if cfg.gamma != 1.0:
        A = np.clip(A, 0.0, 1.0) ** cfg.gamma

    n_bins = cfg.n_fft // 2 + 1

    # Newman phases — deterministic low-crest multitone; the receiver reads
    # magnitudes only, so phase choice is free.
    phases = (np.pi * (np.arange(H, dtype=np.float64) ** 2) / H).astype(np.float32)
    symbol_gain = 0.98 / (np.sqrt(H) + 1e-9)

    # Edge-replicated pad frames keep every payload sample fully overlapped.
    pad = cfg.win_length // cfg.hop_length  # 4 frames
    A_pad = np.concatenate([np.repeat(A[:, :1], pad, axis=1), A,
                            np.repeat(A[:, -1:], pad, axis=1)], axis=1)
    Wp = W + 2 * pad
    k_idx = np.arange(idx_min, idx_max_excl, dtype=np.float64)[:, None]
    # absolute frame index (pad frames at -pad..-1) keeps the carriers
    # phase-continuous across the pad/payload boundary
    t_idx = np.arange(-pad, W + pad, dtype=np.float64)[None, :]
    frame_phase = 2.0 * np.pi * (k_idx * cfg.hop_length / cfg.n_fft) * t_idx
    carriers = np.exp(1j * (phases[:, None].astype(np.float64) + frame_phase)).astype(np.complex64)

    Z = np.zeros((n_bins, Wp), dtype=np.complex64)
    Z[idx_min:idx_max_excl, :] = (A_pad * symbol_gain * carriers).astype(np.complex64)

    Zt = torch.from_numpy(Z)

    if cfg.window == "hamming":
        window = torch.hamming_window(cfg.win_length, periodic=True)
        window_used = "hamming"
    else:
        window = torch.hann_window(cfg.win_length, periodic=True)
        window[0] = 1e-4
        window[-1] = 1e-4
        window_used = "hann_patched"

    payload_samples = cfg.win_length + (W - 1) * cfg.hop_length
    full_samples = cfg.win_length + (Wp - 1) * cfg.hop_length

    x_tensor = torch.istft(
        Zt,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=False,
        onesided=True,
        length=full_samples,
    )
    x_full = x_tensor.float().numpy()
    start = pad * cfg.hop_length
    return x_full[start:start + payload_samples], window_used

def encode_image_to_audio_stft(gray_img_2d, cfg):
    H, W = gray_img_2d.shape
    if H != cfg.target_size or W != cfg.target_size:
        raise ValueError("Image must be resized to target_size first.")

    df = cfg.sample_rate / cfg.n_fft
    idx_min = int(np.ceil(cfg.f_min / df))
    idx_max_allowed = int(np.floor(cfg.f_max / df))

    if idx_min + (H - 1) > idx_max_allowed:
        raise ValueError("Not enough bins in [f_min, f_max] for H rows.")

    idx_max_excl = idx_min + H

    n_bins = cfg.n_fft // 2 + 1
    if idx_max_excl > n_bins:
        raise ValueError("idx_max_excl exceeds rFFT bins; check n_fft/target_size/f_min/f_max.")

    f_min_actual = idx_min * df
    f_max_actual = (idx_max_excl - 1) * df

    payload, window_used = _synth_payload(gray_img_2d, cfg, idx_min, idx_max_excl)
    payload_samples = payload.size

    if cfg.payload_scale_override is not None:
        payload_scale = float(cfg.payload_scale_override)
    else:
        payload_scale = compute_payload_scale(cfg, idx_min, idx_max_excl)
    payload = (payload * payload_scale).astype(np.float32)

    # Limiter: fail loudly by default. A silent per-file rescale is
    # unrecoverable at the receiver and poisons brightness calibration.
    pre_limiter_peak = float(np.max(np.abs(payload)) + 1e-9)
    if pre_limiter_peak > cfg.limiter_threshold:
        if not cfg.allow_limiter:
            raise LimiterExceeded(pre_limiter_peak, cfg.limiter_threshold)
        print(f"WARNING: payload limiter engaged (peak {pre_limiter_peak:.3f}); "
              f"decoded brightness for this file will be scaled by "
              f"{cfg.limiter_threshold / pre_limiter_peak:.3f}.")
        payload = (payload * (cfg.limiter_threshold / pre_limiter_peak)).astype(np.float32)

    # Fade only the payload edges to keep chirps strong and correlation-friendly
    payload = fade_in_out(payload, cfg)

    guard_samples = int(cfg.guard_seconds * cfg.sample_rate)
    guard = np.zeros(guard_samples, dtype=np.float32)

    parts = []
    preamble_samples = 0
    if cfg.add_preamble:
        up = make_chirp(cfg, f_min_actual, f_max_actual) * cfg.chirp_amplitude
        preamble_samples = up.size
        parts.append(up)
    parts.extend([payload, guard])
    if cfg.add_tail_chirp:
        down = make_chirp(cfg, f_max_actual, f_min_actual) * cfg.chirp_amplitude
        parts.extend([down, guard.copy()])
    x = np.concatenate(parts, axis=0)

    # Sample distance from up-chirp start to down-chirp start (Doppler/clock
    # drift shows up as a deviation of the measured spacing from this value).
    nominal_chirp_spacing = preamble_samples + payload_samples + guard_samples

    meta = {
        "format_version": FORMAT_VERSION,
        "sample_rate": cfg.sample_rate,
        "n_fft": cfg.n_fft,
        "win_length": cfg.win_length,
        "hop_length": cfg.hop_length,
        "f_min": cfg.f_min,
        "f_max": cfg.f_max,
        "f_min_actual": f_min_actual,
        "f_max_actual": f_max_actual,
        "df": df,
        "idx_min": idx_min,
        "idx_max_excl": idx_max_excl,
        "add_preamble": cfg.add_preamble,
        "preamble_seconds": cfg.preamble_seconds,
        "chirp_amplitude": cfg.chirp_amplitude,
        "chirp_fade_ms": cfg.chirp_fade_ms,
        "add_tail_chirp": cfg.add_tail_chirp,
        "nominal_chirp_spacing": int(nominal_chirp_spacing),
        "phase_seed": cfg.phase_seed,
        "phase_mode": "frame_consistent",
        "phase_scheme": "newman",
        "payload_scale": payload_scale,
        "pre_limiter_peak": pre_limiter_peak,
        "gamma": cfg.gamma,
        "H": H,
        "W": W,
        "center": False,
        "payload_samples": payload_samples,
        "guard_samples": guard_samples,
        "window": window_used,
    }
    return x, meta

def encode_dir(cfg=None):
    """Encode every image in cfg.input_dir. Writes to .partial dirs first and
    swaps them in only on full success; raises EncodeError otherwise (limiter
    offenders are listed so the caller can swap reserve images in and
    re-encode every directory at one shared payload_scale)."""
    cfg = cfg or EncoderConfig()
    _final_out, _final_meta = cfg.output_dir, cfg.meta_dir
    cfg.output_dir = _final_out.rstrip("/") + ".partial"
    cfg.meta_dir = _final_meta.rstrip("/") + ".partial"
    ensure_empty_dir(cfg.output_dir)
    ensure_empty_dir(cfg.meta_dir)

    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted([f for f in os.listdir(cfg.input_dir) if f.lower().endswith(exts)])

    print(f"Encoding {len(files)} images -> WAV in '{cfg.output_dir}' "
          f"({cfg.f_min:.0f}-{cfg.f_max:.0f} Hz, {FORMAT_VERSION})")

    ok = 0
    per_file = []
    offenders = []
    failures = []   # generic errors are collected and RAISED, never skipped
    fmt_snapshot = None
    for filename in tqdm(files):
        stem = os.path.splitext(filename)[0]
        try:
            img_path = os.path.join(cfg.input_dir, filename)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((cfg.target_size, cfg.target_size), Image.LANCZOS)
            gray = ImageOps.grayscale(img)
            gray_np = np.array(gray)

            audio, meta = encode_image_to_audio_stft(gray_np, cfg)

            wav_path = os.path.join(cfg.output_dir, stem + ".wav")
            scipy.io.wavfile.write(wav_path, cfg.sample_rate, (audio * 32767.0).astype(np.int16))

            with open(os.path.join(cfg.meta_dir, stem + ".json"), "w") as f:
                json.dump(meta, f, indent=2)

            pre = int(cfg.preamble_seconds * cfg.sample_rate) if cfg.add_preamble else 0
            seg = audio[pre:pre + meta["payload_samples"]]
            per_file.append({"stem": stem,
                             "pre_limiter_peak": meta["pre_limiter_peak"],
                             "payload_rms_dbfs": float(20 * np.log10(
                                 np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-12))})
            if fmt_snapshot is None:
                fmt_snapshot = {k: v for k, v in meta.items()
                                if k != "pre_limiter_peak"}
            ok += 1
        except LimiterExceeded as e:
            offenders.append({"stem": stem, "pre_limiter_peak": e.peak})
        except Exception as e:
            failures.append({"file": filename, "error": repr(e)})
            print(f"FAILED {filename}: {e}")

    # ---- preflight report: written for EVERY run, next to the WAVs
    if per_file:
        peaks = np.array([p["pre_limiter_peak"] for p in per_file])
        worst = per_file[int(np.argmax(peaks))]
        thr = cfg.limiter_threshold
        summary = {
            "n_encoded": ok,
            "n_limiter_offenders": len(offenders),
            "limiter_threshold": thr,
            "max_pre_limiter_peak": float(peaks.max()),
            "max_peak_file": worst["stem"],
            "headroom_db_to_limiter": float(20 * np.log10(thr / peaks.max())),
            "peak_quantiles": {q: float(np.quantile(peaks, float(q)))
                               for q in ("0.5", "0.9", "0.99", "1.0")},
        }
        report = {"summary": summary, "offenders": offenders, "files": per_file}
        with open(os.path.join(cfg.output_dir, "peaks_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Peaks: max {summary['max_pre_limiter_peak']:.3f} "
              f"({summary['max_peak_file']}), headroom to limiter "
              f"{summary['headroom_db_to_limiter']:+.1f} dB -> peaks_report.json")

    if fmt_snapshot is not None:
        with open(os.path.join(cfg.output_dir, "format.json"), "w") as f:
            json.dump(fmt_snapshot, f, indent=2)

    print(f"Done. Wrote {ok} WAV files.")

    if offenders:
        print(f"ENCODE FAILED: limiter would have engaged on {len(offenders)} "
              f"image(s) — refusing to ship silently mis-calibrated audio.")
        for o in offenders:
            print(f"  {o['stem']}: pre-limiter peak {o['pre_limiter_peak']:.3f} "
                  f"(threshold {cfg.limiter_threshold:.2f})")
    if offenders or failures:
        raise EncodeError(offenders, failures)   # originals stay untouched
    staged_replace(cfg.output_dir, _final_out)
    staged_replace(cfg.meta_dir, _final_meta)
    cfg.output_dir, cfg.meta_dir = _final_out, _final_meta
    return {"n_encoded": ok, "files": per_file}

def staged_replace(partial, final):
    """Backup-and-rollback directory swap: the previous valid `final` survives
    any failure; the backup is removed only after success."""
    partial, final = str(partial), str(final)
    bak = final + ".bak"
    shutil.rmtree(bak, ignore_errors=True)
    try:
        if os.path.exists(final):
            os.rename(final, bak)
        os.rename(partial, final)
    except Exception:
        if os.path.exists(bak) and not os.path.exists(final):
            os.rename(bak, final)
        raise
    shutil.rmtree(bak, ignore_errors=True)


# === SHARED PHYSICS CELL: channel simulator ===
# Byte-identical in 01 / 02 / 03. If you edit it here, copy it to the
# other two notebooks; the parity gate below pins its behavior.
@dataclass
class ChannelParams:
    snr_db: Optional[float] = 15.0        # in-band SNR; None = no noise
    gain_db: float = 0.0                  # level error (mic distance / volume)
    start_offset: int = 0                 # samples of lead-in before the signal
    tail_pad: int = 4800                  # samples of noise after the signal
    multipath: List[Tuple[float, float]] = field(default_factory=list)
    #   list of (delay_seconds, amplitude) echoes; direct path implicit (1.0 @ 0)
    tilt_db: float = 0.0                  # spectral tilt across [f_lo, f_hi]
    impulse_rate_hz: float = 0.0          # mean rate of impulsive clicks
    impulse_amp: float = 0.0              # click amplitude rel. to signal RMS
    bursts: List[Tuple[float, float, float]] = field(default_factory=list)

def sample_random_params(rng: np.random.Generator,
                         snr_range=(0.0, 30.0),
                         gain_db_range=(-9.0, 6.0),
                         max_start_offset=24000,
                         multipath_prob=0.7,
                         max_echoes=3,
                         max_delay_s=0.004,
                         max_echo_amp=0.5,
                         tilt_db_range=(-6.0, 6.0),
                         burst_prob=0.35,
                         max_bursts=2,
                         burst_dur_range=(0.010, 0.120),
                         burst_amp_range=(2.0, 10.0),
                         clean_prob=0.1) -> ChannelParams:
    """Draw a random channel realization for training augmentation. With
    probability clean_prob returns a benign channel (offset only) so the
    model keeps its clean-channel performance."""
    start = int(rng.integers(0, max_start_offset + 1))
    if rng.random() < clean_prob:
        return ChannelParams(snr_db=None, gain_db=float(rng.uniform(-3, 3)),
                             start_offset=start)
    echoes = []
    if rng.random() < multipath_prob:
        for _ in range(int(rng.integers(1, max_echoes + 1))):
            echoes.append((float(rng.uniform(0.0002, max_delay_s)),
                           float(rng.uniform(0.05, max_echo_amp))))
    bursts = []
    if rng.random() < burst_prob:
        for _ in range(int(rng.integers(1, max_bursts + 1))):
            bursts.append((float(rng.uniform(0.05, 0.95)),
                           float(rng.uniform(*burst_dur_range)),
                           float(rng.uniform(*burst_amp_range))))
    return ChannelParams(
        snr_db=float(rng.uniform(*snr_range)),
        gain_db=float(rng.uniform(*gain_db_range)),
        start_offset=start,
        multipath=echoes,
        tilt_db=float(rng.uniform(*tilt_db_range)),
        bursts=bursts,
    )

def deterministic_params_for(stem: str, base_seed: int = 111,
                             snr_grid=(24.0, 18.0, 12.0, 6.0, 0.0),
                             **kw) -> ChannelParams:
    """Fixed channel realization for a validation file: same every epoch, so
    validation curves are comparable across training."""
    h = base_seed & 0xFFFFFFFFFFFFFFFF
    for ch in stem:
        h = (h * 1000003 + ord(ch)) & 0xFFFFFFFFFFFFFFFF
    rng = np.random.default_rng(h)
    p = sample_random_params(rng, clean_prob=0.0, **kw)
    p.snr_db = float(snr_grid[h % len(snr_grid)])
    return p

def _inband_noise_sigma(signal_power: float, snr_db: float,
                        sr: int, f_lo: float, f_hi: float) -> float:
    target_inband_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    inband_fraction = max((f_hi - f_lo) / (sr / 2.0), 1e-6)
    total_noise_power = target_inband_noise_power / inband_fraction
    return float(np.sqrt(total_noise_power))

def apply_channel(x: np.ndarray, sr: int, params: ChannelParams,
                  rng: np.random.Generator,
                  f_lo: float = 6000.0, f_hi: float = 9000.0,
                  ref_power: Optional[float] = None) -> np.ndarray:
    """Apply the channel to a full transmitted waveform. Returns float32 audio
    of length start_offset + len(filtered signal) + tail_pad.

    ref_power: the power the SNR is defined against. Pass the PAYLOAD power
    (mean square over the payload region) — the whole-waveform default is
    chirp-dominated, which would silently redefine every SNR."""
    x = np.asarray(x, dtype=np.float32)
    sig_power = float(ref_power) if ref_power is not None else \
        float(np.mean(x.astype(np.float64) ** 2) + 1e-20)

    # 1) multipath (direct path + echoes)
    y = x.copy()
    if params.multipath:
        n_h = int(max(d for d, _ in params.multipath) * sr) + 1
        h = np.zeros(n_h + 1, dtype=np.float32)
        h[0] = 1.0
        for delay_s, amp in params.multipath:
            h[int(round(delay_s * sr))] += amp
        y = np.convolve(y, h).astype(np.float32)

    # 2) spectral tilt (linear in dB across [f_lo, f_hi], clamped outside)
    if abs(params.tilt_db) > 1e-6:
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(y.size, d=1.0 / sr)
        frac = np.clip((f - f_lo) / max(f_hi - f_lo, 1.0), 0.0, 1.0)
        g = 10.0 ** ((frac - 0.5) * params.tilt_db / 20.0)  # 0 dB mean at band center
        y = np.fft.irfft(Y * g, n=y.size).astype(np.float32)

    # 3) overall gain
    y *= 10.0 ** (params.gain_db / 20.0)

    # 4) placement in a longer recording
    out = np.zeros(params.start_offset + y.size + params.tail_pad, dtype=np.float32)
    out[params.start_offset:params.start_offset + y.size] = y

    # 5) AWGN at in-band SNR, referenced to PRE-gain signal power so the SNR
    #    knob and the gain knob stay independent
    if params.snr_db is not None:
        sigma = _inband_noise_sigma(sig_power, params.snr_db, sr, f_lo, f_hi)
        sigma *= 10.0 ** (params.gain_db / 20.0)
        out += rng.normal(0.0, sigma, size=out.size).astype(np.float32)

    # 6) localized noise bursts (relative to the gain-adjusted signal RMS)
    if params.bursts:
        sig_rms_g = np.sqrt(sig_power) * 10.0 ** (params.gain_db / 20.0)
        for t_frac, dur_s, amp in params.bursts:
            i0 = int(np.clip(t_frac, 0.0, 1.0) * out.size)
            n = min(int(dur_s * sr), out.size - i0)
            if n > 0:
                out[i0:i0 + n] += rng.normal(0.0, amp * sig_rms_g,
                                             size=n).astype(np.float32)

    # 7) impulsive noise
    if params.impulse_rate_hz > 0 and params.impulse_amp > 0:
        n_clicks = rng.poisson(params.impulse_rate_hz * out.size / sr)
        sig_rms = np.sqrt(sig_power) * 10.0 ** (params.gain_db / 20.0)
        for _ in range(n_clicks):
            i = int(rng.integers(0, out.size))
            out[i] += float(rng.choice([-1, 1])) * params.impulse_amp * sig_rms

    return out


# === SHARED PHYSICS CELL: receiver front end ===
# Byte-identical in 01 / 02 / 03. If you edit it here, copy it to the
# other two notebooks; the parity gate below pins its behavior.
@dataclass
class RxReport:
    up_start: int                 # sample index of the up-chirp start (48 kHz)
    payload_start: int
    gain: float                   # estimated channel amplitude gain
    sync_confidence: float        # correlation peak / robust floor
    stretch_est: Optional[float]  # measured/nominal chirp spacing; None if no tail
    tail_confidence: Optional[float]
    clipped_fraction: float       # fraction of pixels clipped at 1.0 after eq

class ReceiverFrontEnd:
    def __init__(self, meta: dict, bandpass: bool = True, headroom: float = 1.0,
                 min_gain: float = 1e-3):
        """meta: an encoder metadata dict (any per-file JSON from the meta dir —
        the format fields are identical across files). headroom: >1.0 maps
        white below full scale, trading contrast for less overshoot clipping."""
        self.meta = meta
        self.sr = int(meta["sample_rate"])
        self.n_fft = int(meta["n_fft"])
        self.win_length = int(meta["win_length"])
        self.hop_length = int(meta["hop_length"])
        self.idx_min = int(meta["idx_min"])
        self.idx_max_excl = int(meta["idx_max_excl"])
        self.payload_samples = int(meta["payload_samples"])
        self.preamble_seconds = float(meta["preamble_seconds"])
        self.preamble_samples = int(self.preamble_seconds * self.sr)
        self.chirp_amplitude = float(meta.get("chirp_amplitude", 0.6))
        self.chirp_fade_ms = float(meta.get("chirp_fade_ms", 0.0))
        self.f0 = float(meta["f_min_actual"])
        self.f1 = float(meta["f_max_actual"])
        self.has_tail = bool(meta.get("add_tail_chirp", False))
        self.nominal_spacing = int(meta.get("nominal_chirp_spacing",
                                            self.preamble_samples + self.payload_samples
                                            + int(meta.get("guard_samples", 0))))
        self.min_gain = min_gain
        self.headroom = float(headroom)

        # RECT analysis window (see section header)
        self.window = torch.ones(self.win_length, dtype=torch.float32)

        self.up_template = self._make_chirp(self.f0, self.f1) * self.chirp_amplitude
        self.down_template = self._make_chirp(self.f1, self.f0) * self.chirp_amplitude
        self.up_energy = float(np.sum(self.up_template.astype(np.float64) ** 2))

        self.bandpass_sos = None
        if bandpass:
            lo = max(self.f0 - 500.0, 100.0)
            hi = min(self.f1 + 500.0, self.sr / 2.0 - 100.0)
            self.bandpass_sos = scipy.signal.butter(4, [lo, hi], btype="bandpass",
                                                    fs=self.sr, output="sos")

        self.eq_map = self._build_equalization_map()

    # ---------------------------------------------------------------- setup

    def _build_equalization_map(self) -> np.ndarray:
        """Synthesize the all-white payload of this exact format, pass it
        through the same band-pass, and measure the band magnitudes. Fully
        deterministic; content-independence verified to <1e-4."""
        H = self.idx_max_excl - self.idx_min
        cfg = SimpleNamespace(sample_rate=self.sr, n_fft=self.n_fft,
                              win_length=self.win_length, hop_length=self.hop_length,
                              gamma=float(self.meta.get("gamma", 1.0)),
                              window=self.meta.get("window", "hamming"),
                              target_size=H)
        probe = np.full((H, H), 255, dtype=np.uint8)
        payload, _ = _synth_payload(probe, cfg, self.idx_min, self.idx_max_excl)
        payload = payload * float(self.meta.get("payload_scale", 1.0))
        if self.bandpass_sos is not None:
            payload = scipy.signal.sosfiltfilt(self.bandpass_sos, payload).astype(np.float32)
        band = self._band_stft(payload)
        return np.maximum(band, 1e-9) * self.headroom

    def _make_chirp(self, f0, f1):
        """Must remain identical to make_chirp (fades included)."""
        n = self.preamble_samples
        t = np.arange(n, dtype=np.float32) / self.sr
        k = (f1 - f0) / max(self.preamble_seconds, 1e-9)
        phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
        x = np.sin(phase).astype(np.float32)
        fade = int(self.chirp_fade_ms * 1e-3 * self.sr)
        if fade > 1 and x.size > 2 * fade:
            x[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            x[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return x

    # ---------------------------------------------------------------- helpers

    def _band_stft(self, seg: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32))
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window,
                          return_complex=True, center=False, onesided=True)
        band = stft.abs()[self.idx_min:self.idx_max_excl, :]
        H = self.idx_max_excl - self.idx_min
        if band.shape[1] != H:
            raise RuntimeError(f"Got {band.shape[1]} frames, expected {H}.")
        return band.numpy().astype(np.float64)

    @staticmethod
    def _resample_if_needed(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return y
        frac = Fraction(sr_out, sr_in).limit_denominator(1000)
        return scipy.signal.resample_poly(y, frac.numerator, frac.denominator).astype(np.float32)

    @staticmethod
    def _correlate(y: np.ndarray, template: np.ndarray) -> np.ndarray:
        """corr[k] = sum_i y[k+i] * template[i], k = 0..len(y)-len(template)."""
        return scipy.signal.fftconvolve(y, template[::-1], mode="valid")

    @staticmethod
    def _peak_and_confidence(corr: np.ndarray) -> Tuple[int, float, float]:
        a = np.abs(corr)
        k = int(np.argmax(a))
        peak = float(corr[k])
        floor = float(np.median(a) + 1e-12)
        return k, peak, float(a[k] / floor)

    # ------------------------------------------------------------------- main

    def process(self, y: np.ndarray, sr: Optional[int] = None):
        """Full front end. Returns (band_norm [0,1] float32 HxW, RxReport).
        band_norm is both the classical decode (times 255) and, via
        to_model_tensor(), the model input."""
        if sr is not None:
            y = self._resample_if_needed(np.asarray(y, dtype=np.float32), int(sr), self.sr)
        y = np.asarray(y, dtype=np.float32)

        if self.bandpass_sos is not None:
            # zero-phase: no group delay, so correlation timing is unbiased;
            # payload is extracted from THIS stream too — the equalization map
            # includes the same filter, and rect sidelobes would otherwise
            # leak out-of-band noise into the band rows
            y = scipy.signal.sosfiltfilt(self.bandpass_sos, y).astype(np.float32)

        # --- sync on the up-chirp
        corr_up = self._correlate(y, self.up_template)
        up_start, peak_up, conf_up = self._peak_and_confidence(corr_up)

        # --- per-transmission gain from the matched-filter peak:
        # at alignment, corr peak = gain * sum(template^2) (+ noise term)
        gain = max(abs(peak_up) / self.up_energy, self.min_gain)

        # --- tail chirp: measure spacing, LOG stretch estimate (no correction)
        stretch = None
        conf_down = None
        if self.has_tail:
            tol = int(0.05 * self.sr)  # ±50 ms search window (~±0.8% stretch)
            lo = max(up_start + self.nominal_spacing - tol, 0)
            hi = min(up_start + self.nominal_spacing + tol + self.preamble_samples, y.size)
            if hi - lo > self.preamble_samples:
                corr_dn = self._correlate(y[lo:hi], self.down_template)
                k_dn, _, conf_down = self._peak_and_confidence(corr_dn)
                stretch = ((lo + k_dn) - up_start) / self.nominal_spacing

        # --- extract payload, gain-correct, analyze, equalize
        p0 = up_start + self.preamble_samples
        seg = np.zeros(self.payload_samples, dtype=np.float32)
        avail = y[p0:p0 + self.payload_samples]
        seg[:avail.size] = avail
        seg /= gain

        band = self._band_stft(seg) / self.eq_map
        band_norm = np.clip(band, 0.0, 1.0).astype(np.float32)
        clipped = float((band_norm >= 0.999).mean())

        report = RxReport(up_start=int(up_start), payload_start=int(p0),
                          gain=float(gain), sync_confidence=float(conf_up),
                          stretch_est=stretch, tail_confidence=conf_down,
                          clipped_fraction=clipped)
        return band_norm, report

    def process_file(self, wav_path: str):
        sr, y = scipy.io.wavfile.read(wav_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if y.dtype.kind in ("i", "u"):
            y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
        else:
            y = y.astype(np.float32)
        return self.process(y, sr=sr)

    # ---------------------------------------------------------------- outputs

    @staticmethod
    def to_model_tensor(band_norm: np.ndarray) -> torch.Tensor:
        """[0,1] band slice -> 1xHxW tensor in [-1,1], matching training."""
        t = torch.from_numpy(band_norm)
        return (t * 2.0 - 1.0).unsqueeze(0).float()

    @staticmethod
    def to_classical_image(band_norm: np.ndarray) -> np.ndarray:
        """The deterministic baseline: the equalized band slice IS the
        grayscale image. Row 0 of the slice is the lowest frequency; the
        encoder writes image row 0 there, so no flip is needed."""
        return np.clip(band_norm * 255.0, 0, 255).astype(np.uint8)


# === SHARED PHYSICS CELL: U-Net model ===
# Byte-identical in 01 / 02 / 03. If you edit it here, copy it to the
# other two notebooks; the parity gate below pins its behavior.
class UNet(nn.Module):
    """Identical to v2/v2.1/v2.2 so existing checkpoints load strictly.
    Input: 1x256x256 spectrogram band in [-1,1]. Output: 3x256x256 RGB
    in [-1,1] (tanh)."""

    def __init__(self):
        super().__init__()
        self.enc1 = self._down(1, 64)
        self.enc2 = self._down(64, 128)
        self.enc3 = self._down(128, 256)
        self.enc4 = self._down(256, 512)
        self.bottleneck = self._down(512, 1024)
        self.dropout = nn.Dropout2d(0.1)
        self.dec4 = self._up(1024, 512)
        self.dec3 = self._up(1024, 256)
        self.dec2 = self._up(512, 128)
        self.dec1 = self._up(256, 64)
        self.final_up = self._up(128, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def _down(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _up(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.dropout(self.bottleneck(e4))
        d4 = torch.cat([self.dec4(b), e4], dim=1)
        d3 = torch.cat([self.dec3(d4), e3], dim=1)
        d2 = torch.cat([self.dec2(d3), e2], dim=1)
        d1 = torch.cat([self.dec1(d2), e1], dim=1)
        x = self.final_up(d1)
        return self.tanh(self.final(x))


# === SHARED PHYSICS CELL: shared utilities ===
# Byte-identical in 01 / 02 / 03. If you edit it here, copy it to the
# other two notebooks; the parity gate below pins its behavior.
def stable_hash(s: str, base: int = 111) -> int:
    h = base & 0xFFFFFFFFFFFFFFFF
    for ch in s:
        h = (h * 1000003 + ord(ch)) & 0xFFFFFFFFFFFFFFFF
    return h

PER_FILE_KEYS = {"pre_limiter_peak"}

def check_format_consistency(train_wav_dir: str, val_wav_dir: str) -> dict:
    """Train and val WAVs MUST share one format (esp. payload_scale) or the
    receiver decodes val at the wrong brightness. Each encode run writes
    format.json into its output dir; compare them and fail loudly."""
    fmts = {}
    for name, d in (("train", train_wav_dir), ("val", val_wav_dir)):
        p = os.path.join(d, "format.json")
        if not os.path.exists(p):
            raise RuntimeError(
                f"{p} is missing — '{d}' was not produced by this encoder. "
                f"Re-encode it before training.")
        with open(p) as f:
            fmts[name] = json.load(f)
    keys = (set(fmts["train"]) | set(fmts["val"])) - PER_FILE_KEYS
    diffs = sorted(k for k in keys
                   if fmts["train"].get(k) != fmts["val"].get(k))
    if diffs:
        lines = [f"  {k}: train={fmts['train'].get(k)!r} val={fmts['val'].get(k)!r}"
                 for k in diffs]
        raise RuntimeError("train/val WAV format mismatch — re-encode both "
                           "with one encoder config:\n" + "\n".join(lines))
    return fmts["train"]

def load_meta(meta_dir: str) -> dict:
    metas = sorted(f for f in os.listdir(meta_dir) if f.endswith(".json"))
    if not metas:
        raise RuntimeError(f"No metadata in {meta_dir}")
    with open(os.path.join(meta_dir, metas[0])) as f:
        return json.load(f)

def list_pairs(wav_dir: str, img_dir: str):
    pairs = []
    for w in sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav")):
        stem = os.path.splitext(w)[0]
        for ext in (".jpg", ".png", ".jpeg", ".bmp", ".JPEG"):
            p = os.path.join(img_dir, stem + ext)
            if os.path.exists(p):
                pairs.append((os.path.join(wav_dir, w), p, stem))
                break
    if not pairs:
        raise RuntimeError(f"No wav/image pairs found in {wav_dir} / {img_dir}")
    return pairs

def load_wav_float(path: str):
    sr, y = scipy.io.wavfile.read(path)
    if y.dtype.kind in ("i", "u"):
        y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
    else:
        y = y.astype(np.float32)
    if y.ndim > 1:            # downmix stereo/multichannel captures to mono
        y = y.mean(axis=1)
    return y, sr

def load_encoder_rgb(path, width, height):
    """EXACTLY the encoder's preprocessing: RGB -> LANCZOS resize. Every
    reference, target, and panel must go through this (or its grayscale
    sibling) so PSNR is measured against what was actually transmitted."""
    return Image.open(path).convert("RGB").resize((width, height),
                                                  Image.LANCZOS)

def load_encoder_gray(path, meta):
    return np.asarray(ImageOps.grayscale(
        load_encoder_rgb(path, meta["W"], meta["H"])), dtype=np.uint8)

def sha256_file(path, blk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def require_file(path, min_bytes=1, what="file"):
    p = Path(path)
    assert p.is_file(), f"missing {what}: {path}"
    assert p.stat().st_size >= min_bytes, \
        f"{what} too small ({p.stat().st_size} B < {min_bytes} B): {path}"
    return p

def psnr(a, b):
    mse = np.mean((np.asarray(a).astype(np.float64)
                   - np.asarray(b).astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

def atomic_torch_save(obj, path):
    """Write-then-rename so readers see either the old or the new complete
    file, never a torn write (monitor-copy and resume-corruption safe)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def materialize_best(run_dir):
    """last_state.pth is the transactional authority. Re-export best_model.pth
    from it when possible, so an interrupt between the state save and the
    convenience export can never feed a STALE best into evaluation."""
    run_dir = Path(run_dir)
    state_p, best_p = run_dir / "last_state.pth", run_dir / "best_model.pth"
    if state_p.exists():
        st = torch.load(state_p, map_location="cpu", weights_only=False)
        if st.get("best_model_state") is not None:
            atomic_torch_save(st["best_model_state"], best_p)
    return require_file(best_p, 1_000_000, "best_model.pth")


# === SHARED PHYSICS CELL: golden parity gate ===
# Byte-identical in 01 / 02 / 03. If you edit it here, copy it to the
# other two notebooks; the parity gate below pins its behavior.
def make_test_pattern(n=256):
    yy, xx = np.mgrid[0:n, 0:n]
    img = (xx / (n - 1) * 160).astype(np.float32)
    img += (yy / (n - 1) * 60)
    checker = (((xx // 16) + (yy // 16)) % 2) * 60.0
    img[: n // 2, n // 2:] += checker[: n // 2, n // 2:]
    disk = ((xx - 64) ** 2 + (yy - 192) ** 2) < 40 ** 2
    img[disk] = 235.0
    img[20:30, :] = 10.0
    return np.clip(img, 0, 255).astype(np.uint8)

def make_smooth(n=256):
    yy, xx = np.mgrid[0:n, 0:n]
    s = 90 + 80 * np.sin(xx / 25) * np.cos(yy / 31) \
        + 60 * np.exp(-((xx - 170) ** 2 + (yy - 80) ** 2) / 3000)
    return np.clip(s, 0, 255).astype(np.uint8)

GOLDEN = {"payload_scale": 587.0, "f_min": 6000.0, "f_max": 8988.0,
          "psnr_pattern_clean": 30.19, "psnr_smooth_clean": 40.07,
          "psnr_pattern_stack": 16.01, "attack_peak": 2.24}

def run_parity_gate(tol_db=0.6, verbose=True):
    """Reproduce the package's golden roundtrip numbers from synthetic images
    (no dataset, no checkpoint) — proof in ~5 s that the physics cells above
    are behaviorally identical to the verified v2.1 code. Numeric tolerances, not
    string matching."""
    cfg = EncoderConfig()
    fe, meta0, res = None, None, {}
    for name, img in (("pattern", make_test_pattern()), ("smooth", make_smooth())):
        audio, meta = encode_image_to_audio_stft(img, cfg)
        if fe is None:
            fe, meta0 = ReceiverFrontEnd(meta), meta
        band, rep = fe.process(audio)
        res[name] = psnr(ReceiverFrontEnd.to_classical_image(band), img)
    got = {"payload_scale": round(float(meta0["payload_scale"]), 1),
           "f_min": float(meta0["f_min_actual"]),
           "f_max": round(float(meta0["f_max_actual"]), 0),
           "psnr_pattern_clean": res["pattern"],
           "psnr_smooth_clean": res["smooth"]}
    # full impairment stack on the pattern (offset+gain+multipath+tilt+noise)
    audio, meta = encode_image_to_audio_stft(make_test_pattern(), cfg)
    pre = int(cfg.preamble_seconds * cfg.sample_rate)
    ref_power = float(np.mean(audio[pre:pre + meta["payload_samples"]] ** 2))
    p = ChannelParams(snr_db=15.0, gain_db=-8.0, start_offset=12345,
                      multipath=[(0.0011, 0.35), (0.0027, 0.18)], tilt_db=4.0)
    y = apply_channel(audio, cfg.sample_rate, p, np.random.default_rng(0),
                      f_lo=cfg.f_min, f_hi=cfg.f_max, ref_power=ref_power)
    band, rep = fe.process(y)
    got["psnr_pattern_stack"] = psnr(ReceiverFrontEnd.to_classical_image(band),
                                     make_test_pattern())
    got["sync_exact"] = (rep.up_start == 12345)
    got["gain_err_pct"] = abs(rep.gain / (10 ** (-8.0 / 20)) - 1.0) * 100
    # limiter guard: selected-rows attack (rows chosen by carrier sign)
    thr = cfg.limiter_threshold
    scale = compute_payload_scale(cfg, 512, 768)
    phases = np.pi * (np.arange(256) ** 2) / 256
    adv = None
    for n0 in (80000.0, 120000.0, 160000.0):
        th = 2 * np.pi * (512 + np.arange(256)) * n0 / 4096 + phases
        cand = np.zeros((256, 256), np.uint8)
        cand[np.cos(th) > 0, :] = 255
        pl, _ = _synth_payload(cand, cfg, 512, 768)
        if float(np.abs(pl).max() * scale) > thr:
            adv = cand
            break
    assert adv is not None, "attack construction failed to exceed threshold"
    try:
        encode_image_to_audio_stft(adv, cfg)
        raise AssertionError("limiter guard did NOT raise on selected-rows attack")
    except LimiterExceeded as e:
        got["attack_peak"] = e.peak
    # ---- compare
    fails = []
    for k in ("payload_scale", "f_min", "f_max"):
        if abs(got[k] - GOLDEN[k]) > 1.0:
            fails.append(f"{k}: {got[k]} != {GOLDEN[k]}")
    for k in ("psnr_pattern_clean", "psnr_smooth_clean", "psnr_pattern_stack"):
        if abs(got[k] - GOLDEN[k]) > tol_db:
            fails.append(f"{k}: {got[k]:.2f} vs {GOLDEN[k]} (tol {tol_db})")
    if abs(got["attack_peak"] - GOLDEN["attack_peak"]) > 0.05:
        fails.append(f"attack_peak {got['attack_peak']:.2f} vs {GOLDEN['attack_peak']}")
    if not got["sync_exact"]:
        fails.append("sync offset not exact under impairment stack")
    if got["gain_err_pct"] > 0.5:
        fails.append(f"gain err {got['gain_err_pct']:.2f}% > 0.5%")
    if verbose:
        for k, v in got.items():
            print(f"  {k:22s} {v if not isinstance(v, float) else round(v, 3)}"
                  + (f"   (golden {GOLDEN[k]})" if k in GOLDEN else ""))
    assert not fails, "PARITY GATE FAILED:\n  " + "\n  ".join(fails)
    print("PARITY GATE PASSED — this notebook's physics cells reproduce the verified v2.1 numbers")
    return got


# === NOTEBOOK-03 HELPERS: fail-closed real-capture gate ===
# The learned prior can make invalid audio look plausible. Live captures
# MUST clear deterministic-receiver checks BEFORE any U-Net inference.
# Thresholds (documented in HANDOFF.md capture-gate spec):
#   raw ADC clipping <  1.0% of samples
#   sync_confidence  >= 30
#   tail_confidence  >= 30 (must be present)
#   |stretch - 1|    <= 0.008
#   payload NOT truncated (recording covers the whole frame)
#   receiver output finite
CAPTURE_GATE = dict(max_adc_clip=0.01, min_sync=30.0,
 min_tail=30.0, max_stretch_dev=0.008)

def decode_capture_gated(y, sr, meta, model, device, gate=CAPTURE_GATE):
    y = np.asarray(y, np.float32)
    adc_clip = float(np.mean(np.abs(y) >= 0.999))
    fe_local = ReceiverFrontEnd(meta)
    band, rep = fe_local.process(y, sr=sr)

    failures = []
    if adc_clip >= gate["max_adc_clip"]:
        failures.append(f"raw ADC clipping {adc_clip:.2%} "
                        f">= {gate['max_adc_clip']:.1%}")
    if rep.sync_confidence < gate["min_sync"]:
        failures.append(f"sync_confidence {rep.sync_confidence:.1f} "
                        f"< {gate['min_sync']}")
    tail_conf = getattr(rep, "tail_confidence", None)
    if tail_conf is None:
        failures.append("tail chirp not detected")
    elif tail_conf < gate["min_tail"]:
        failures.append(f"tail_confidence {tail_conf:.1f} "
                        f"< {gate['min_tail']}")
    stretch = getattr(rep, "stretch_est", None)
    if stretch is not None and abs(stretch - 1.0) > gate["max_stretch_dev"]:
        failures.append(f"|stretch-1| {abs(stretch-1):.4f} "
                        f"> {gate['max_stretch_dev']}")
    # payload truncation: does the recording actually cover the frame?
    cap_samples = round(len(y) * meta["sample_rate"] / sr)
    payload_start = getattr(rep, "payload_start",
                            getattr(rep, "up_start", None))
    if payload_start is not None:
        need = payload_start + int(meta["payload_samples"])
        if need > cap_samples:
            failures.append(f"payload truncated: need {need} "
                            f"samples, have {cap_samples}")
    if not np.isfinite(band).all():
        failures.append("receiver output contains non-finite values")

    diag = dict(adc_clip=adc_clip, sync_confidence=rep.sync_confidence,
                tail_confidence=tail_conf, stretch=stretch,
                classical_only=len(failures) > 0, failures=failures)

    classical = ReceiverFrontEnd.to_classical_image(band)
    if failures:
        return classical, None, rep, diag
    with torch.no_grad():
        t = ReceiverFrontEnd.to_model_tensor(band).unsqueeze(0).to(device)
        pred = model(t)[0].clamp(-1, 1)
    un = ((pred * 0.5 + 0.5) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return classical, un, rep, diag
