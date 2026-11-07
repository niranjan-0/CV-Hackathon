"""Step 3: build FP16 and INT8 variants from deploy/fp32.

  python 02_quantize.py --deploy deploy [--int8-conv] [--only fp16|int8]

fp16 : weights+activations in fp16, I/O kept fp32 (easy for the Android side).
       VAE stays fp32 (the SD VAE can output NaN/black images in fp16).
       Time projection and timestep-embedding linear nodes stay fp32 to avoid mixed-type inputs.
int8 : dynamic quantization of MatMul/Gemm weights (attention + FFN).
       --int8-conv additionally quantizes Conv (smaller, but ConvInteger on ARM CPU may be slow: benchmark it).
"""
import argparse
import os
import shutil

import onnx
import onnxruntime as ort

from common import dir_size_mb

ap = argparse.ArgumentParser()
ap.add_argument("--deploy", default="deploy")
ap.add_argument("--int8-conv", action="store_true")
ap.add_argument("--only", choices=["fp16", "int8"],
                default=None, help="build just one variant")
args = ap.parse_args()
src = os.path.join(args.deploy, "fp32")


def save(model, path):
    for f in (path, path + ".data"):
        if os.path.exists(f):
            os.remove(f)
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=os.path.basename(path) + ".data", size_threshold=1024)


def copy_model(name, dst):
    for f in (name + ".onnx", name + ".onnx.data"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, f))


def check_loads(path):
    """Load the model on CPU right away so a broken conversion fails here, not later in the benchmark."""
    ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("   loads OK:", path)


# ---------- FP16 ----------
if args.only in (None, "fp16"):
    from onnxconverter_common import float16

    d16 = os.path.join(args.deploy, "fp16")
    os.makedirs(d16, exist_ok=True)
    for name in ("text_encoder", "unet"):
        print("fp16:", name)
        m = onnx.load(os.path.join(src, name + ".onnx"),
                      load_external_data=True)
        block = [
            n.name for n in m.graph.node
            if "time_proj" in n.name or "/time_embedding/linear_" in n.name
        ]
        print("   keeping", len(block), "timestep embedding nodes in fp32")
        m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True,
                                               node_block_list=block)
        save(m16, os.path.join(d16, name + ".onnx"))
        check_loads(os.path.join(d16, name + ".onnx"))
    copy_model("vae_decoder", d16)

# ---------- INT8 ----------
if args.only in (None, "int8"):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    d8 = os.path.join(args.deploy, "int8")
    os.makedirs(d8, exist_ok=True)
    ops = ["MatMul", "Gemm"] + (["Conv"] if args.int8_conv else [])
    for name in ("text_encoder", "unet"):
        print("int8:", name, ops)
        quantize_dynamic(os.path.join(src, name + ".onnx"), os.path.join(d8, name + ".onnx"),
                         weight_type=QuantType.QUInt8, op_types_to_quantize=ops,
                         use_external_data_format=True)
    copy_model("vae_decoder", d8)

print("\nSizes (MB):")
for v in ("fp32", "fp16", "int8"):
    p = os.path.join(args.deploy, v)
    if not os.path.isdir(p):
        continue
    print(f"  {v:5s} total {dir_size_mb(p):8.0f}")
    for n in ("text_encoder", "unet", "vae_decoder"):
        s = sum(os.path.getsize(os.path.join(p, f))
                for f in os.listdir(p) if f.startswith(n + ".onnx")) / 2**20
        print(f"        {n:13s} {s:8.0f}")
