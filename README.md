# Auditory Image Transmission

Send still images through an acoustic channel. A speaker plays a short WAV;
a microphone records it; a receiver recovers the image. The reference model
is trained on underwater reef imagery — the target application is
low-bandwidth underwater acoustic communication, where bandwidth is scarce
but visual context matters.

Two receivers run in parallel on every capture:

- **Classical** — a deterministic decode of the transmitted luminance. What
  was actually received. No learned components.
- **Learned** — a U-Net that reconstructs color from that same luminance
  using a prior learned from underwater imagery. Always shown alongside the
  classical decode; never shown alone.

Every capture passes a **fail-closed gate** before the learned
reconstruction is shown: bad audio never produces a hallucinated image.

## Demo panel

<!-- Add a hero panel image to docs/hero_panel.png and uncomment:
     ![Reference / transmitted luminance / classical decode / U-Net reconstruction](docs/hero_panel.png)
-->

Left to right: reference RGB · transmitted luminance (grayscale, what the
audio actually carried) · classical decode (what the receiver recovered
from the audio) · U-Net reconstruction (color prior applied).

## How it works

**Encoder.** Image → grayscale → modulated as a ~5.9 s WAV in the 6000–8988
Hz band at 48 kHz sample rate, with a sync preamble and tail chirp for
timing lock.

**Channel.** Whatever sits between the speaker and mic. Room reverb, gain
error between devices, clock drift, ADC clipping, ambient noise.

**Decoder.** Sync-lock on the preamble → measure timing stretch → demodulate
to recover luminance → apply fail-closed gate (clip / sync / tail / stretch
/ completeness) → if gate passes, run the U-Net on the classical output for
color reconstruction → emit classical, reconstruction, and an honest
side-by-side panel.

## Installation

Tested on Jetson Orin Nano (JetPack 6 and 7) and generic Linux x86_64. Any
system with Python 3.10+ and PyTorch should work.

```bash
./setup.sh
source venv/bin/activate
```

The setup script auto-detects your platform:

- **Jetson** (detected via `/etc/nv_tegra_release`) — installs the correct
  aarch64 CUDA torch wheel for your JetPack version. See
  [`docs/JETSON.md`](docs/JETSON.md) for JetPack-specific notes including
  the Orin sm_87 caveat.
- **Generic Linux** — installs CPU torch by default; edit the script's
  `--index-url` line for CUDA if you have an NVIDIA GPU.

The script ends by running the **parity gate**. Setup is complete only when
the gate prints `PARITY GATE PASSED` with the reference numbers
(payload_scale 587.0, band 6000–8988 Hz, PSNR 30.19 / 40.07 / 16.03
±0.6 dB). The gate is the acceptance test — if your torch/BLAS/FFT stack
has silently changed the numeric physics, the gate fails and nothing
downstream is trusted.

## Usage

```bash
# Self-test (once per machine, or after any environment change)
python3 ait.py selftest

# Encode an image to a transmittable WAV
python3 ait.py encode input.jpg tx.wav

# Play tx.wav from any device across a room.
# Record on/near the receiver:
#   - Quiet-ish room, mic 0.5–3 m from the speaker
#   - Recording level so peaks sit near −6 dBFS
#   - Start recording ~1 s before playback and stop ~1 s after
#     (preamble AND tail need clean capture)

# Decode a recording to images
python3 ait.py decode recording.wav --model best_model.pth --out decode_out/
```

Input format: any WAV — int or float PCM, any standard sample rate
(resampled internally), stereo downmixed to mono.

Output in `--out/`:

| file                       | when           | meaning                                         |
| -------------------------- | -------------- | ----------------------------------------------- |
| `classical.png`            | always         | the received information — deterministic decode |
| `reconstruction.png`       | gate pass only | U-Net learned reconstruction                    |
| `panel.png`                | always         | side-by-side, honestly labeled                  |
| `capture_diagnostics.json` | always         | clip / sync / tail / stretch / failure reasons  |

Exit codes: `0` = gate passed, `2` = gate failed (classical only).

## The capture gate

Every recording is gated before the U-Net runs:

| check              | threshold           |
| ------------------ | ------------------- |
| ADC clip           | < 1% of samples     |
| Sync confidence    | ≥ 30                |
| Tail confidence    | ≥ 30                |
| \|Stretch − 1\|    | ≤ 0.008             |
| Payload complete   | full ~5.9 s present |
| Outputs finite     | no NaN              |

Any failure suppresses the U-Net and emits only the classical decode plus
a printed list of reasons. This is the core safety property of the system:
a bad audio recording can never produce a plausible-looking but
hallucinated image.

Loopback reference numbers (playing `sample_tx.wav` directly into the
receiver without any acoustic path): clip 0.00% / sync 151 / tail 2475 /
stretch 1.0.

**Iteration guide when the gate fails:**

- Clipping → lower playback volume.
- Sync/tail confidence low → move closer or reduce ambient noise.
- Stretch out of band → something in the recording chain is resampling.

Each retry costs seconds.

## Honest wording obligations

The U-Net output is a **learned reconstruction** using a color prior, not
a direct recovery of transmitted color. Color is inferred by the receiver;
color is never transmitted. The panel makes both facts explicit by always
showing the classical decode alongside.

An enhancement variant (LPIPS + adversarial fine-tune) is available in the
codepath but is not the default weights. Its outputs are **prior-driven
enhancement — texture and color may be synthesized** and should be labeled
accordingly if used.

## Repository contents

```
ait_core.py       physics module — encoder, decoder, capture gate
ait.py            CLI (selftest / encode / decode)
extract_core.py   regenerates ait_core.py from the training notebook
                  (refuses on any physics-fingerprint mismatch)
setup.sh          platform auto-detect (Jetson / generic Linux) + parity gate
requirements.txt  numpy / scipy / pillow / tqdm  (torch installed by setup.sh)
best_model.pth    default weights (L1 + LPIPS on LSUI reef imagery)
model_meta.json   model provenance and training configuration
sample_input.jpg  example encoder input
sample_tx.wav     example encoded transmission
docs/JETSON.md    JetPack-specific notes (Orin sm_87 caveat, etc.)
```

## Provenance

`ait_core.py` is generated verbatim from the training notebook's shared
physics cells using `extract_core.py`. The extractor computes a fingerprint
across those cells and refuses to regenerate on any mismatch, so the
deployed physics is guaranteed byte-identical to the trained one.

Validation performed before shipping:

- Parity gate passed on torch 2.13 (training environment was 2.4.1 —
  cross-version check confirms the physics is platform-agnostic).
- Encode → decode loopback: classical PSNR matched the reference exactly.
- Fail-closed path verified with a clipped input: reconstruction
  suppressed, exit code 2.

The default weights were selected on plain-L1 validation from a 4-way
training race (L1 only, L1+MS-SSIM, L1+LPIPS, L1+LPIPS+adversarial) and
are the dominant model on PSNR, MS-SSIM, and LPIPS across 24 tested
operating points spanning AWGN and room-reverb channels at SNRs 0–30 dB.
See `model_meta.json` for the full configuration.

To regenerate `ait_core.py` from your own copy of the source notebook:

```bash
python3 extract_core.py /path/to/03_eval_and_visualize.ipynb
```

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgments

Training data: LSUI (Large-Scale Underwater Image) dataset, research use
only. Publishable demo panels use public-domain imagery.

Model architecture: U-Net. Perceptual loss weight tuned via Optuna.
