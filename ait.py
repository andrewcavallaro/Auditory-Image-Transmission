#!/usr/bin/env python3
"""AirHop standalone receiver for Jetson (or any machine).

Three commands:

  selftest                        run the golden parity gate (~5 s, no files
                                  needed). MUST pass on a new platform before
                                  anything else is trusted.
  encode  IMG OUT.wav             image -> AirHop transmission WAV (48 kHz).
  decode  REC.wav --model M.pth   recorded WAV + checkpoint -> images:
                                    classical.png        (always, on any decode)
                                    reconstruction.png   (ONLY if the capture
                                                          gate passes)
                                    panel.png            (side by side)
                                    capture_diagnostics.json

The capture gate is fail-closed and unchanged from the pod: ADC clip < 1%,
sync_confidence >= 30, tail_confidence >= 30, |stretch-1| <= 0.008, payload
complete, finite outputs. On failure you get the classical decode plus the
printed reasons — never a hallucinated U-Net image.

Wording rule carried from the project dossier: the U-Net output is a LEARNED
RECONSTRUCTION and the panel labels it so; the classical image is the
received information.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import ait_core as A


# ----------------------------------------------------------------- meta
def build_meta(cache: Path | None = None) -> dict:
    """The receiver's format metadata, regenerated deterministically by
    running the encoder once on a dummy image (the encode is CPU-
    deterministic by design; payload_scale comes out 587.0). Cached to
    JSON so repeat decodes skip the ~seconds of synthesis."""
    if cache and cache.exists():
        return json.loads(cache.read_text())
    cfg = A.EncoderConfig()
    dummy = np.zeros((cfg.target_size, cfg.target_size), np.uint8)
    _, meta = A.encode_image_to_audio_stft(dummy, cfg)
    meta = {k: v for k, v in meta.items() if k not in A.PER_FILE_KEYS}
    if cache:
        cache.write_text(json.dumps(meta, indent=2))
    return meta


# ----------------------------------------------------------------- model
def load_model(path: str, device: torch.device) -> torch.nn.Module:
    """Accepts either a bare state_dict, a best_model.pth-style export, or a
    last_state.pth training-state dict (in which case the best weights are
    preferred, falling back to the live model state). Loads strictly —
    the UNet is architecture-frozen precisely so checkpoints load exactly."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state, origin = None, "state_dict"
    if isinstance(obj, dict):
        for key in ("best_model_state", "model_state", "model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                state, origin = obj[key], key
                break
        if state is None:
            state = obj  # assume it IS the state_dict
    else:
        raise SystemExit(f"unrecognized checkpoint object: {type(obj)}")
    model = A.UNet()
    model.load_state_dict(state, strict=True)
    model.eval()
    if device.type == "cuda":
        if load_model.skip_gpu_check:
            print("WARNING: --skip-gpu-check set — trusting CUDA without "
                  "the CPU-parity sanity gate")
        elif not gpu_sanity_check(model):
            print("falling back to CPU for U-Net inference (still fast "
                  "enough; re-test after a torch upgrade with sm_87 support)")
            device = torch.device("cpu")
    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"model: {path} [{origin}] {n/1e6:.1f}M params -> {device}")
    return model, device


load_model.skip_gpu_check = False


def gpu_sanity_check(model: torch.nn.Module, tol: float = 2e-3) -> bool:
    """GPU inference gate. On JetPack 7.x the upstream aarch64 torch
    wheels build Ampere for sm_80; sm_80 PTX can JIT to Orin's sm_87 at
    runtime, so U-Net GPU inference may work or may hit a missing kernel
    with either of two failure modes:
      (a) silent NaN / mismatched output — checked by output comparison
      (b) a raised CUDA exception ("no kernel image is available for
          execution on the device", etc.) — caught by try/except so
          decode falls back to CPU instead of terminating the process
    Either failure => CPU fallback, loudly. This function is the arbiter;
    it does not presume the outcome."""
    try:
        torch.manual_seed(0)
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            ref = model.to("cpu")(x)
            got = model.to("cuda")(x.cuda()).cpu()
    except Exception as exc:
        model.to("cpu")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"GPU SANITY GATE FAILED: {type(exc).__name__}: {exc}")
        return False
    model.to("cpu")
    if not torch.isfinite(got).all():
        print("GPU SANITY GATE FAILED: non-finite U-Net output on CUDA")
        return False
    delta = float((ref - got).abs().max())
    ok = delta <= tol
    print(f"GPU sanity gate: max |cpu-gpu| = {delta:.2e} "
          f"({'PASS' if ok else f'FAIL, tol {tol}'})")
    return ok


def pick_device(arg: str, skip_gpu_check: bool = False) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if arg == "cuda":
            raise SystemExit("--device cuda requested but CUDA is not available")
        print("NOTE: CUDA not available, running U-Net on CPU (a few seconds "
              "per frame — fine for single decodes)")
        return torch.device("cpu")
    return torch.device("cuda")  # provisional; verified per-model at load


# ----------------------------------------------------------------- panel
def make_panel(classical: np.ndarray, unet: np.ndarray | None,
               gate_ok: bool) -> Image.Image:
    """Side-by-side, labeled: classical (received info) | U-Net (learned
    reconstruction). Pure PIL, no matplotlib dependency."""
    from PIL import ImageDraw
    h, w = classical.shape[:2]
    pad, cap = 8, 22
    n_panes = 2
    canvas = Image.new("RGB", (n_panes * w + (n_panes + 1) * pad,
                               h + 2 * pad + cap), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(classical).convert("RGB"), (pad, pad))
    d.text((pad + 2, pad + h + 4), "classical (received info)",
           fill=(230, 230, 230))
    x2 = 2 * pad + w
    if unet is not None:
        canvas.paste(Image.fromarray(unet), (x2, pad))
        d.text((x2 + 2, pad + h + 4), "U-Net (learned reconstruction)",
               fill=(230, 230, 230))
    else:
        d.rectangle([x2, pad, x2 + w, pad + h], outline=(120, 40, 40),
                    width=2)
        d.text((x2 + 10, pad + h // 2 - 6),
               "suppressed: capture gate failed", fill=(220, 120, 120))
        d.text((x2 + 2, pad + h + 4), "U-Net (not shown)",
               fill=(160, 160, 160))
    return canvas


# ----------------------------------------------------------------- cmds
def cmd_selftest(_args) -> int:
    got = A.run_parity_gate(verbose=True)
    return 0 if got else 1


def cmd_encode(args) -> int:
    cfg = A.EncoderConfig()
    img = Image.open(args.image).convert("L").resize(
        (cfg.target_size, cfg.target_size), Image.LANCZOS)
    audio, meta = A.encode_image_to_audio_stft(
        np.asarray(img, np.uint8), cfg)
    import scipy.io.wavfile
    scipy.io.wavfile.write(args.out, cfg.sample_rate,
                           (np.clip(audio, -1, 1) * 32767.0).astype(np.int16))
    dur = len(audio) / cfg.sample_rate
    print(f"wrote {args.out}: {dur:.2f} s @ {cfg.sample_rate} Hz, "
          f"payload_scale {meta['payload_scale']:.1f}, band "
          f"{meta['f_min_actual']:.0f}-{meta['f_max_actual']:.0f} Hz")
    print("play this through a speaker; record ~1 s before and after "
          "(peaks near -6 dBFS); then: ait.py decode <recording.wav>")
    return 0


def cmd_decode(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    load_model.skip_gpu_check = args.skip_gpu_check
    meta = build_meta(Path(args.meta) if args.meta else
                      Path(__file__).with_name("model_meta.json"))
    if args.model:
        model, device = load_model(args.model, device)
    else:
        model = None
        device = torch.device("cpu") if device.type == "cuda" else device
    if model is None:
        print("no --model given: classical decode only")

    y, sr = A.load_wav_float(args.wav)
    print(f"input: {args.wav} — {len(y)/sr:.2f} s @ {sr} Hz")
    t0 = time.time()
    if model is not None:
        cl, un, rep, diag = A.decode_capture_gated(y, sr, meta, model=model,
                                                   device=device)
    else:  # classical-only path still runs the full gate for diagnostics
        class _Null(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3],
                                   device=x.device)
        cl, un, rep, diag = A.decode_capture_gated(y, sr, meta,
                                                   model=_Null().to(device),
                                                   device=device)
        un = None
    dt = time.time() - t0

    print(f"sync {rep.sync_confidence:.0f} | tail {diag.get('tail_confidence')}"
          f" | stretch {diag.get('stretch')} | ADC clip "
          f"{diag['adc_clip']:.2%} | decode {dt:.2f} s")
    gate_ok = not diag["failures"]
    if not gate_ok:
        print("CAPTURE GATE FAILED — classical decode only "
              "(U-Net suppressed to avoid hallucinated output):")
        for r in diag["failures"]:
            print(f"  - {r}")

    Image.fromarray(cl).save(out_dir / "classical.png")
    if gate_ok and un is not None and model is not None:
        Image.fromarray(un).save(out_dir / "reconstruction.png")
    make_panel(cl, un if (gate_ok and model is not None) else None,
               gate_ok).save(out_dir / "panel.png")

    diag_out = {k: (float(v) if isinstance(v, (np.floating, float)) and
                    v is not None else v)
                for k, v in diag.items() if k != "failures"}
    diag_out["failures"] = diag["failures"]
    diag_out["decode_seconds"] = round(dt, 3)
    diag_out["device"] = str(device)
    (out_dir / "capture_diagnostics.json").write_text(
        json.dumps(diag_out, indent=2))
    print(f"outputs in {out_dir}/: classical.png"
          + (", reconstruction.png" if gate_ok and model else "")
          + ", panel.png, capture_diagnostics.json")
    return 0 if gate_ok else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="golden parity gate")
    e = sub.add_parser("encode", help="image -> transmission WAV")
    e.add_argument("image")
    e.add_argument("out")
    d = sub.add_parser("decode", help="recorded WAV (+checkpoint) -> images")
    d.add_argument("wav")
    d.add_argument("--model", default=None,
                   help="checkpoint: last_state.pth / best_model.pth / state_dict")
    d.add_argument("--out", default="decode_out")
    d.add_argument("--meta", default=None,
                   help="format meta JSON (default: regenerate+cache next to script)")
    d.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    d.add_argument("--skip-gpu-check", action="store_true",
                   help="trust CUDA without the CPU-parity sanity gate "
                        "(NOT recommended on JetPack 7 until torch ships "
                        "sm_87 kernels)")
    args = ap.parse_args()
    return {"selftest": cmd_selftest, "encode": cmd_encode,
            "decode": cmd_decode}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
