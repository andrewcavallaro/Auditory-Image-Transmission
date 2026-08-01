#!/usr/bin/env bash
# AirHop standalone receiver — setup for Jetson Orin Nano (Super).
# Supports JetPack 7.x (Ubuntu 24.04, CUDA 13.x, Python 3.12) and
# JetPack 6.x (Ubuntu 22.04, CUDA 12.6, Python 3.10).
#
# torch source differs by JetPack generation:
#   JP7 (L4T R38+, first Orin-supported release is JP 7.2 with R39):
#       Orin is aarch64/Ampere sm_87 (SBSA is a separate line — Jetson
#       Thor). The upstream PyTorch cu132 aarch64/cp312 wheel installs
#       cleanly and reports CUDA available. The current release matrix
#       builds Ampere for sm_80, and sm_80 PTX can JIT to sm_87 at
#       runtime, so U-Net GPU inference may work — or may hit a missing
#       kernel and either raise "no kernel image is available for
#       execution on the device" or emit silent NaN. ait.py's GPU
#       sanity gate is the arbiter: it runs the U-Net on CPU and GPU with
#       the same input (wrapped in try/except) and uses CUDA only if the
#       outputs are finite and match. On sanity-gate FAIL, CPU takes over
#       — a few seconds per 5.9 s frame, perfectly usable.
#   JP6 (L4T R36): NVIDIA/community CUDA wheels from the
#       pypi.jetson-ai-lab.io jp6/cu126 index (note: .io — the old .dev
#       domain is gone).
#
# Ends by running the golden parity gate. THE GATE MUST PASS before you
# trust a single decode from this machine. Idempotent; safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

echo "== AirHop Jetson setup =="

# --- detect JetPack generation from L4T release ----------------------
L4T="unknown"
if [ -f /etc/nv_tegra_release ]; then
    L4T=$(sed -n 's/^# R\([0-9]*\).*/\1/p' /etc/nv_tegra_release | head -1)
fi
echo "L4T major release: R${L4T}"
case "$L4T" in
    3[89]|4[0-9]) GEN="jp7" ;;   # R38/R39+ -> JetPack 7.x
    36)           GEN="jp6" ;;   # R36      -> JetPack 6.x
    *)            GEN="generic"
                  echo "WARNING: could not identify JetPack; will try a" \
                       "generic torch install (may be CPU-only)." ;;
esac
echo "install profile: $GEN"

# --- base tools ------------------------------------------------------
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv libopenblas-dev

# venv keeps us clear of system packages (PEP 668 on Ubuntu 24.04) and
# of anything ROS may have installed; rm -rf venv to rebuild.
if [ ! -d venv ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python3 -m pip install -q --upgrade pip

# --- core deps (ML-free) --------------------------------------------
python3 -m pip install -q numpy scipy pillow tqdm

# --- torch -----------------------------------------------------------
have_torch=$(python3 -c "
try:
    import torch; print('cuda' if torch.cuda.is_available() else 'cpu')
except Exception: print('none')")
# Only skip if we already have a CUDA-capable torch; a leftover CPU-only
# torch from a previous failed run should NOT prevent a retry.
if [ "$have_torch" = "cuda" ]; then
    echo "torch with CUDA already present in venv — skipping install"
elif [ "$have_torch" = "cpu" ] && [ "$GEN" = "generic" ]; then
    echo "CPU torch already present, no JetPack detected — leaving as is"
else
    if [ "$have_torch" = "cpu" ]; then
        echo "found CPU-only torch in venv; attempting CUDA install "
        echo "(previous run may have fallen back). If this fails again"
        echo "and you want a clean slate: rm -rf venv && re-run."
        python3 -m pip uninstall -qy torch || true
    fi
    case "$GEN" in
        jp7)
            echo "installing upstream aarch64 torch (cu132 index) for JetPack 7..."
            python3 -m pip install -q torch \
                --index-url https://download.pytorch.org/whl/cu132 \
            || { echo "cu132 index failed; trying default PyPI (CPU fallback)";
                 python3 -m pip install -q torch; }
            ;;
        jp6)
            echo "installing JetPack 6 CUDA torch (jetson-ai-lab jp6/cu126)..."
            python3 -m pip install -q torch \
                --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
            || { echo "jp6 index failed; trying default PyPI (CPU fallback)";
                 python3 -m pip install -q torch; }
            ;;
        *)
            python3 -m pip install -q torch
            ;;
    esac
fi

# --- report ----------------------------------------------------------
python3 - <<'PY'
import torch
print(f"python OK | torch {torch.__version__} | CUDA available: "
      f"{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("(GPU U-Net use is decided per-run by ait.py's CPU-parity "
          "sanity gate — on current JetPack 7 wheels this decides at runtime "
          "whether U-Net inference stays on GPU or falls back to CPU)")
PY

echo
echo "== running golden parity gate (the acceptance test) =="
python3 ait.py selftest

echo
echo "Setup complete. Activate with:  source venv/bin/activate"
echo "Then:"
echo "  python3 ait.py encode picture.png tx.wav        # make a transmission"
echo "  python3 ait.py decode recording.wav --model last_state.pth --out out/"
