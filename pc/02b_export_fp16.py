"""Step 3b: export the FP16 variant directly from PyTorch on the GPU (replaces the fp16 part of 02_quantize.py).

  python 02b_export_fp16.py --deploy deploy

Loads SD1.5, fuses LCM-LoRA in fp32 (no precision loss), then casts to fp16 and exports.
I/O stays fp32 (casts are inside the graph), VAE stays fp32 (copied from deploy/fp32).
"""
import argparse
import os
import shutil
import tempfile

import onnx
import onnxruntime as ort
import torch
from diffusers import StableDiffusionPipeline

from common import dir_size_mb

ap = argparse.ArgumentParser()
ap.add_argument(
    "--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--lora", default="latent-consistency/lcm-lora-sdv1-5")
ap.add_argument("--deploy", default="deploy")
ap.add_argument("--opset", type=int, default=17)
args = ap.parse_args()

assert torch.cuda.is_available(
), "This script needs the GPU (torch.cuda.is_available() is False)"
dev = "cuda"
d16 = os.path.join(args.deploy, "fp16")
os.makedirs(d16, exist_ok=True)

print("Loading SD1.5 (fp32) and fusing LCM-LoRA ...")
pipe = StableDiffusionPipeline.from_pretrained(
    args.model, torch_dtype=torch.float32, safety_checker=None)
pipe.load_lora_weights(args.lora)
pipe.fuse_lora()
pipe.unload_lora_weights()
te = pipe.text_encoder.half().to(dev).eval()
unet = pipe.unet.half().to(dev).eval()


class TextEnc(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m(input_ids)[0].float()  # fp32 output


class Unet(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, sample, timestep, encoder_hidden_states):
        # timestep stays fp32: diffusers builds the sinusoidal embedding in fp32 and then casts it
        out = self.m(sample.half(), timestep,
                     encoder_hidden_states.half(), return_dict=False)[0]
        return out.float()  # fp32 output


def export(module, example, path, in_names, out_names, dyn):
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "m.onnx")
        kw = dict(input_names=in_names, output_names=out_names, dynamic_axes=dyn,
                  opset_version=args.opset, do_constant_folding=True)
        try:
            torch.onnx.export(module, example, tmp, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(module, example, tmp, **kw)
        m = onnx.load(tmp, load_external_data=True)
        name = os.path.basename(path)
        for f in (path, path + ".data"):
            if os.path.exists(f):
                os.remove(f)
        onnx.save_model(m, path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=name + ".data", size_threshold=1024)
    # fail here if the graph is invalid
    ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("exported and loads OK:", path)


with torch.no_grad():
    export(TextEnc(te), (torch.randint(0, 49407, (1, 77), dtype=torch.int64, device=dev),),
           os.path.join(d16, "text_encoder.onnx"), ["input_ids"], ["last_hidden_state"], None)
    export(Unet(unet),
           (torch.randn(1, 4, 64, 64, device=dev), torch.tensor([999.0], device=dev),
            torch.randn(1, 77, 768, device=dev)),
           os.path.join(d16, "unet.onnx"),
           ["sample", "timestep", "encoder_hidden_states"], ["out_sample"],
           {"sample": {2: "h", 3: "w"}, "out_sample": {2: "h", 3: "w"}})

for f in ("vae_decoder.onnx", "vae_decoder.onnx.data"):
    p = os.path.join(args.deploy, "fp32", f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(d16, f))
print(f"\nfp16 folder: {dir_size_mb(d16):.0f} MB")
