"""Step 2b — export FP16 models directly from PyTorch on the GPU.

This is the ONLY reliable way to get a working fp16 UNet for this project.
The ONNX post-hoc converter (onnxconverter_common) crashes on this UNet with
a 'Mul type mismatch' error. This script avoids that by casting in PyTorch
before export so the graph is already fp16 when it reaches ONNX.

Usage:
  python 02b_export_fp16.py --deploy deploy

Requirements:
  - deploy/fp32/ must exist (run 01_export_onnx.py first)
  - CUDA GPU must be available (torch.cuda.is_available() == True)
  - ~4 GB free VRAM

What it does:
  1. Loads SD1.5 + LCM-LoRA from Hugging Face cache (already downloaded in Step 1)
  2. Fuses the LoRA weights into the UNet
  3. Casts the text encoder and UNet to fp16
  4. Exports both as ONNX with fp32 I/O (Android code stays unchanged)
  5. Copies the fp32 VAE decoder as-is (fp16 VAE produces black images)
  6. Loads each exported model on CPU to confirm it is not broken

Expected output:
  deploy/fp16/text_encoder.onnx + .data   (~300 MB)
  deploy/fp16/unet.onnx + .data           (~1600 MB)
  deploy/fp16/vae_decoder.onnx + .data    (~83 MB, fp32 copy)

Each model prints:  exported and loads OK: deploy/fp16/<name>.onnx
"""
import argparse
import os
import shutil
import tempfile

import onnx
import onnxruntime as ort
import torch
from diffusers import StableDiffusionPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--model",  default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--lora",   default="latent-consistency/lcm-lora-sdv1-5")
ap.add_argument("--deploy", default="deploy")
ap.add_argument("--opset",  type=int, default=17)
args = ap.parse_args()

# ── sanity checks ─────────────────────────────────────────────────────────────
if not os.path.isdir(os.path.join(args.deploy, "fp32")):
    raise SystemExit(
        f"ERROR: {args.deploy}/fp32/ not found.\n"
        "Run  python 01_export_onnx.py --out deploy --taesd  first."
    )
if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: No CUDA GPU found.\n"
        "Check  python -c \"import torch; print(torch.cuda.is_available())\"\n"
        "Make sure you installed torch with --extra-index-url https://download.pytorch.org/whl/cu124"
    )

dev = "cuda"
d16 = os.path.join(args.deploy, "fp16")
os.makedirs(d16, exist_ok=True)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Free VRAM: {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB")

# ── load SD1.5 and fuse LCM-LoRA ─────────────────────────────────────────────
print("\nLoading SD1.5 (fp32) ...")
pipe = StableDiffusionPipeline.from_pretrained(
    args.model, torch_dtype=torch.float32, safety_checker=None
)
print("Fusing LCM-LoRA ...")
pipe.load_lora_weights(args.lora)
pipe.fuse_lora()
pipe.unload_lora_weights()

te   = pipe.text_encoder.half().to(dev).eval()
unet = pipe.unet.half().to(dev).eval()
del pipe   # free CPU RAM before export


# ── wrapper modules ───────────────────────────────────────────────────────────
# Both wrappers accept fp32 inputs and return fp32 outputs.
# The cast to fp16 happens inside, so the ONNX graph has fp32 I/O
# with fp16 weights and internal ops — exactly what Android expects.

class TextEnc(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m

    def forward(self, input_ids):
        # input_ids is int64 — no cast needed
        return self.m(input_ids)[0].float()   # [1, 77, 768] fp32


class Unet(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m

    def forward(self, sample, timestep, encoder_hidden_states):
        # Inputs arrive as fp32; cast to fp16 for the actual UNet forward pass
        out = self.m(
            sample.half(),
            timestep,                           # timestep stays fp32 — sinusoidal embedding
            encoder_hidden_states.half(),
            return_dict=False,
        )[0]
        return out.float()                      # [1, 4, H/8, W/8] fp32


# ── export helper ─────────────────────────────────────────────────────────────
def export(module, example, path, in_names, out_names, dyn):
    print(f"\nExporting {os.path.basename(path)} ...")
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "m.onnx")
        kw  = dict(
            input_names=in_names, output_names=out_names,
            dynamic_axes=dyn, opset_version=args.opset,
            do_constant_folding=True,
        )
        try:
            torch.onnx.export(module, example, tmp, dynamo=False, **kw)
        except TypeError:   # older torch has no `dynamo` kwarg
            torch.onnx.export(module, example, tmp, **kw)

        m    = onnx.load(tmp, load_external_data=True)
        name = os.path.basename(path)
        # remove old files so onnx.save_model does not append
        for f in (path, path + ".data"):
            if os.path.exists(f):
                os.remove(f)
        onnx.save_model(
            m, path,
            save_as_external_data=True, all_tensors_to_one_file=True,
            location=name + ".data", size_threshold=1024,
        )

    # immediately load on CPU — if the graph is broken this will crash here
    # with a clear error rather than a silent failure later in the benchmark
    ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    size_mb = (
        os.path.getsize(path) +
        os.path.getsize(path + ".data")
    ) / 2**20
    print(f"exported and loads OK: {path}  ({size_mb:.0f} MB)")


# ── run exports ───────────────────────────────────────────────────────────────
with torch.no_grad():
    # Text encoder
    export(
        TextEnc(te),
        (torch.randint(0, 49407, (1, 77), dtype=torch.int64, device=dev),),
        os.path.join(d16, "text_encoder.onnx"),
        ["input_ids"], ["last_hidden_state"],
        None,   # no dynamic axes for text encoder
    )

    # UNet — dynamic height/width so the same model works at 256/384/512 px
    export(
        Unet(unet),
        (
            torch.randn(1, 4, 64, 64, device=dev),          # sample
            torch.tensor([999.0], device=dev),               # timestep
            torch.randn(1, 77, 768, device=dev),             # encoder_hidden_states
        ),
        os.path.join(d16, "unet.onnx"),
        ["sample", "timestep", "encoder_hidden_states"],
        ["out_sample"],
        {
            "sample":     {2: "h", 3: "w"},
            "out_sample": {2: "h", 3: "w"},
        },
    )

# ── copy fp32 VAE (never convert VAE to fp16) ─────────────────────────────────
print("\nCopying fp32 VAE decoder (fp16 VAE produces black images — this is intentional) ...")
for f in ("vae_decoder.onnx", "vae_decoder.onnx.data"):
    src = os.path.join(args.deploy, "fp32", f)
    dst = os.path.join(d16, f)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  copied {f}")
    else:
        print(f"  WARNING: {src} not found — did Step 1 finish successfully?")

# ── summary ───────────────────────────────────────────────────────────────────
total = sum(
    os.path.getsize(os.path.join(d16, f))
    for f in os.listdir(d16)
) / 2**20
print(f"\nfp16 folder total: {total:.0f} MB")
print("\nDone. Next step:")
print("  python 02_int8_only.py --deploy deploy")
