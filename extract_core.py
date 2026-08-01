#!/usr/bin/env python3
"""Regenerate ait_core.py from 03_eval_and_visualize.ipynb.

Refuses to emit unless the seven SHARED PHYSICS CELLs fingerprint to
41c48d3150865466 — the house rule, enforced mechanically. Usage:

    python3 extract_core.py /path/to/03_eval_and_visualize.ipynb
"""
import hashlib, json, sys

EXPECTED_FP = "41c48d3150865466"

def main(nb_path):
    d = json.load(open(nb_path))
    cells = ["".join(c["source"]) for c in d["cells"] if c["cell_type"] == "code"]
    shared = [c for c in cells if c.startswith("# === SHARED PHYSICS CELL")]
    assert len(shared) == 7, f"expected 7 shared cells, got {len(shared)}"
    fp = hashlib.sha256("\n".join(shared).encode()).hexdigest()[:16]
    assert fp == EXPECTED_FP, f"FINGERPRINT MISMATCH: {fp} != {EXPECTED_FP} — refusing to emit"
    gate = [c for c in cells if c.startswith(
        "# === NOTEBOOK-03 HELPERS: fail-closed real-capture gate ===")]
    assert len(gate) == 1, "capture-gate helper cell not found"
    gate_sha = hashlib.sha256(gate[0].encode()).hexdigest()[:16]
    header = (f'"""ait_core — AirHop v3.3.4 physics, extracted for standalone deployment.\n\n'
              f'GENERATED — do not hand-edit. Source: {nb_path}\n\n'
              f'SHARED PHYSICS fingerprint: {fp}  (7 cells, byte-identical across NB 01/02/03)\n'
              f'Capture-gate helper sha256[:16]: {gate_sha}\n\n'
              f'The golden parity gate (run_parity_gate) is the acceptance test for this\n'
              f'module on any new machine: payload_scale 587.0, band 6000-8988 Hz, PSNR\n'
              f'30.19 / 40.07 clean and 16.01 under the impairment stack, tol +/-0.6 dB.\n'
              f'NEVER trust this module on a platform where the gate has not passed.\n'
              f'"""\n\n')
    open("ait_core.py", "w").write(header + "\n\n".join(shared) + "\n\n" + gate[0])
    print(f"wrote ait_core.py  fingerprint={fp}  gate_cell={gate_sha}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "03_eval_and_visualize.ipynb")
