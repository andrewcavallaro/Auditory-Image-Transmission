#!/usr/bin/env bash
# =============================================================================
# refactor.sh — clean up the Auditory-Image-Transmission repo:
#   * rename  airhop.py         -> ait.py
#             airhop_core.py    -> ait_core.py
#             airhop_meta.json  -> model_meta.json
#             setup_jetson.sh   -> setup.sh
#             1050.jpg          -> sample_input.jpg
#             tx.wav            -> sample_tx.wav
#   * update all internal references (imports, hardcoded strings) in code
#   * add LICENSE (MIT)
#   * add docs/JETSON.md carrying the Jetson-specific notes (Orin sm_87 etc.)
#   * replace README.md with the underwater-image-transmission rewrite
#   * ensure .gitignore covers venv/ , airhop_env/ , __pycache__/ , *.pyc
#
# Run from the repo root. Does NOT commit — everything is staged for review.
# =============================================================================
set -euo pipefail

# --- preflight ---------------------------------------------------------------
if [ ! -f airhop.py ] || [ ! -f airhop_core.py ]; then
    echo "ERROR: expected to find airhop.py and airhop_core.py in the current directory."
    echo "Run this from the repo root (where README.md and airhop.py sit)."
    exit 1
fi
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "ERROR: not inside a git repository."
    exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is not clean. Commit or stash your changes first:"
    git status --short
    exit 1
fi

# --- portable in-place sed (GNU vs BSD/macOS) --------------------------------
sed_i() {
    if sed --version >/dev/null 2>&1; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

# --- 1. rename files (git mv preserves history) ------------------------------
echo "=== [1/6] Renaming files ==="
git mv airhop.py         ait.py
git mv airhop_core.py    ait_core.py
git mv airhop_meta.json  model_meta.json
git mv setup_jetson.sh   setup.sh
git mv 1050.jpg          sample_input.jpg
git mv tx.wav            sample_tx.wav

# --- 2. update internal references in code/scripts ---------------------------
echo "=== [2/6] Updating internal references ==="
for f in ait.py ait_core.py extract_core.py setup.sh; do
    [ -f "$f" ] || continue
    # Python module names and hardcoded file references
    sed_i 's/\bairhop_core\b/ait_core/g'          "$f"
    sed_i 's/\bairhop_meta\.json\b/model_meta.json/g' "$f"
    sed_i 's/\bairhop\.py\b/ait.py/g'             "$f"
    # venv directory name (setup.sh + any script that activates it)
    sed_i 's/\bairhop_env\b/venv/g'               "$f"
done

# --- 3. .gitignore additions (safe: only add if missing) ---------------------
echo "=== [3/6] Ensuring .gitignore covers venv / cache ==="
touch .gitignore
add_ignore() {
    grep -qxF "$1" .gitignore || echo "$1" >> .gitignore
}
add_ignore "venv/"
add_ignore "airhop_env/"
add_ignore "__pycache__/"
add_ignore "*.pyc"
add_ignore "out/"
add_ignore "out_gray/"
add_ignore "out_pattern/"
add_ignore "decode_out/"

# --- 4. LICENSE (MIT) --------------------------------------------------------
echo "=== [4/6] Writing LICENSE (MIT) ==="
cat > LICENSE <<'EOF'
MIT License

Copyright (c) 2026 Andrew Cavallaro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
EOF

# --- 5. docs/JETSON.md -------------------------------------------------------
echo "=== [5/6] Writing docs/JETSON.md ==="
mkdir -p docs
cat > docs/JETSON.md <<'EOF'
# Jetson-specific installation notes

`setup.sh` auto-detects Jetson platforms via `/etc/nv_tegra_release` and
installs the appropriate CUDA torch wheel:

- **JetPack 7** (Orin support arrived in JP 7.2, L4T R39, Ubuntu 24.04, CUDA
  13.2, Python 3.12) — upstream aarch64 cu132 wheel.
- **JetPack 6** (L4T R36) — jetson-ai-lab jp6/cu126 wheel.
- Older or unknown JetPack versions — CPU torch fallback.

The Jetson SBSA stack (Thor) is a separate line and is not currently
covered by these wheels.

## The Orin sm_87 caveat (as of mid-2026)

The upstream aarch64 torch release matrix builds Ampere for sm_80. sm_80
PTX can JIT to Orin's sm_87 at runtime, so U-Net GPU inference **may** work
— or may hit a missing kernel and either raise `no kernel image is available
for execution on the device` or emit silent NaN.

`ait.py` doesn't presume the outcome. At model load it runs the U-Net on
the same input on CPU and GPU (wrapped in try/except) and uses CUDA only if
the outputs are finite and match (max |delta| ≤ 2e-3). The log will say:

```
GPU sanity gate: ... PASS       # decode proceeds on GPU
GPU SANITY GATE FAILED: ...     # decode falls back to CPU cleanly
```

CPU decode is a few seconds per frame — perfectly usable, just slower.

If the sanity gate fails on your wheel and NVIDIA later ships sm_87-enabled
JP7 wheels, retry with:

```bash
pip install -U torch --index-url https://download.pytorch.org/whl/cu132
```

**Never use `--skip-gpu-check` on JP7** unless you have independently
verified GPU output. A silently-NaN U-Net is exactly the hallucination
class the capture gate exists to prevent.

## Recovery from a broken torch install

The installer will attempt to reinstall CUDA torch on re-run if it finds
only a CPU torch present. For a fully clean slate:

```bash
rm -rf venv && ./setup.sh
```

## Acceptance test

Setup is complete only when the parity gate prints `PARITY GATE PASSED`
with payload_scale 587.0, band 6000–8988 Hz, PSNR 30.19 / 40.07 / 16.03
(±0.6 dB). The gate is the acceptance test — new torch version, new BLAS,
new FFT: if any of it has changed the numeric physics, the gate fails and
nothing downstream is trusted. Never trust a run whose log lacks the gate
passing.
EOF

# --- 6. README.md ------------------------------------------------------------
echo "=== [6/6] Writing README.md ==="
cat > README.md <<'EOF'
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
EOF

# --- stage everything for review --------------------------------------------
git add -A

echo ""
echo "==============================================================="
echo "  Refactor complete. Nothing has been committed yet."
echo "==============================================================="
echo ""
echo "Staged changes:"
git status --short
echo ""
echo "Diff stats:"
git diff --cached --stat
echo ""
echo "Sanity check — no leftover 'airhop' references in code (expect empty):"
grep -nE '\bairhop\b' ait.py ait_core.py extract_core.py setup.sh 2>/dev/null || echo "(clean)"
echo ""
echo "Next steps:"
echo "  1. Review the diffs:      git diff --cached ait.py ait_core.py extract_core.py setup.sh"
echo "  2. Test the CLI locally:  python3 ait.py selftest"
echo "  3. Update LICENSE year/name if needed."
echo "  4. Add a hero panel:      cp <your panel.png> docs/hero_panel.png"
echo "                            then uncomment the image line in README.md"
echo "  5. Commit:                git commit -m 'Rename to ait_*, generic Linux support, rewritten README'"
echo "  6. Push:                  git push"
echo ""
echo "To bail out and revert everything:"
echo "  git reset --hard HEAD"
