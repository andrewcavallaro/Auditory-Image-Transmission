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
