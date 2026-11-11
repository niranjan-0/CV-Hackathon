"""Step 3 — build INT8 variant from deploy/fp32.

Runs entirely on CPU. Takes about 10 minutes.
Does NOT touch fp16 (that is handled by 02b_export_fp16.py).

Usage:
  python 02_int8_only.py --deploy deploy
  python 02_int8_only.py --deploy deploy --int8-conv   # also quantize Conv layers

What it does:
  - Dynamic INT8 quantization of all MatMul and Gemm ops (attention + FFN)
  - Optionally quantizes Conv ops with --int8-conv (smaller model, may be slower on some ARM CPUs)
  - VAE copied from fp32 as-is

Expected output:
  deploy/int8/text_encoder.onnx + .data   (~180 MB)
  deploy/int8/unet.onnx + .data           (~1000 MB)
  deploy/int8/vae_decoder.onnx + .data    (~83 MB, fp32 copy)
"""
import argparse
import os
import shutil

from common import dir_size_mb

ap = argparse.ArgumentParser()
ap.add_argument("--deploy",    default="deploy")
ap.add_argument("--int8-conv", action="store_true",
                help="also quantize Conv layers (smaller, may be slower on ARM)")
args = ap.parse_args()

src = os.path.join(args.deploy, "fp32")
d8  = os.path.join(args.deploy, "int8")

# ── sanity check ──────────────────────────────────────────────────────────────
if not os.path.isdir(src):
    raise SystemExit(
        f"ERROR: {src} not found.\n"
        "Run  python 01_export_onnx.py --out deploy --taesd  first."
    )

os.makedirs(d8, exist_ok=True)

# ── INT8 dynamic quantization ─────────────────────────────────────────────────
from onnxruntime.quantization import QuantType, quantize_dynamic

ops = ["MatMul", "Gemm"]
if args.int8_conv:
    ops.append("Conv")
    print("Including Conv layers in INT8 quantization (--int8-conv)")

for name in ("text_encoder", "unet"):
    src_path = os.path.join(src, name + ".onnx")
    dst_path = os.path.join(d8, name + ".onnx")
    print(f"\nINT8: {name}  ops={ops}")
    print(f"  source : {src_path}")
    print(f"  output : {dst_path}")
    quantize_dynamic(
        src_path, dst_path,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=ops,
        use_external_data_format=True,
    )
    size_mb = (
        os.path.getsize(dst_path) +
        os.path.getsize(dst_path + ".data")
    ) / 2**20
    print(f"  done  ({size_mb:.0f} MB)")

# ── copy fp32 VAE ─────────────────────────────────────────────────────────────
print("\nCopying fp32 VAE decoder ...")
for f in ("vae_decoder.onnx", "vae_decoder.onnx.data"):
    p = os.path.join(src, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(d8, f))
        print(f"  copied {f}")
    else:
        print(f"  WARNING: {p} not found")

# ── size summary ──────────────────────────────────────────────────────────────
print("\nSizes (MB):")
for v in ("fp32", "fp16", "int8"):
    p = os.path.join(args.deploy, v)
    if not os.path.isdir(p):
        print(f"  {v:5s}  (not built yet)")
        continue
    print(f"  {v:5s}  total {dir_size_mb(p):8.0f}")
    for n in ("text_encoder", "unet", "vae_decoder"):
        s = sum(
            os.path.getsize(os.path.join(p, f))
            for f in os.listdir(p)
            if f.startswith(n + ".onnx")
        ) / 2**20
        print(f"         {n:13s} {s:8.0f}")

print("\nDone. Next step:")
print("  python 03_benchmark.py --deploy deploy --out results_gpu "
      "--device cuda --ort-device cuda --variants fp32,fp16 --seeds 2")
