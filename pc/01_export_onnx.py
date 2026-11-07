"""Step 1+2: SD1.5 + LCM-LoRA (fused) -> ONNX (text encoder, UNet, VAE decoder).

Usage:
  python 01_export_onnx.py --out deploy            # standard SD VAE
  python 01_export_onnx.py --out deploy_taesd --taesd   # tiny VAE (faster decode, less RAM)

Output layout (copy `deploy*/` to the phone):
  deploy/{vocab.json, merges.txt, config.json, fp32/{text_encoder,unet,vae_decoder}.onnx}
"""
import argparse
import json
import os
import shutil
import tempfile

import onnx
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--lora", default="latent-consistency/lcm-lora-sdv1-5")
ap.add_argument("--out", default="deploy")
ap.add_argument("--taesd", action="store_true", help="swap in madebyollin/taesd")
ap.add_argument("--opset", type=int, default=17)
args = ap.parse_args()

out_fp32 = os.path.join(args.out, "fp32")
os.makedirs(out_fp32, exist_ok=True)

print("Loading SD1.5 and fusing LCM-LoRA ...")
pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, safety_checker=None)
pipe.load_lora_weights(args.lora)
pipe.fuse_lora()
pipe.unload_lora_weights()
if args.taesd:
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32)
vae_scale = 1.0 if args.taesd else float(pipe.vae.config.scaling_factor)

# tokenizer files (used by Python and by the Kotlin tokenizer)
with tempfile.TemporaryDirectory() as td:
    pipe.tokenizer.save_pretrained(td)
    for f in ("vocab.json", "merges.txt"):
        shutil.copy(os.path.join(td, f), os.path.join(args.out, f))
json.dump({"vae_scale": vae_scale, "latent_channels": 4, "hidden": 768, "max_len": 77,
           "bos": 49406, "eos": 49407, "lcm": True, "taesd": bool(args.taesd)},
          open(os.path.join(args.out, "config.json"), "w"), indent=2)


def export(module, example, path, in_names, out_names, dyn):
    """Export to ONNX; consolidate external-data into a single .data file."""
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "m.onnx")
        kw = dict(input_names=in_names, output_names=out_names, dynamic_axes=dyn,
                  opset_version=args.opset, do_constant_folding=True)
        try:
            torch.onnx.export(module, example, tmp, dynamo=False, **kw)
        except TypeError:  # older torch has no `dynamo` kwarg
            torch.onnx.export(module, example, tmp, **kw)
        m = onnx.load(tmp, load_external_data=True)
        name = os.path.basename(path)
        for f in (path, path + ".data"):
            if os.path.exists(f):
                os.remove(f)
        onnx.save_model(m, path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=name + ".data", size_threshold=1024)
    print("exported", path)


class TextEnc(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m(input_ids)[0]  # last_hidden_state [1,77,768]


class Unet(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.m(sample, timestep, encoder_hidden_states, return_dict=False)[0]


class VaeDec(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, latent_sample):
        x = self.m.decode(latent_sample, return_dict=False)[0]
        return ((x / 2) + 0.5).clamp(0, 1)  # [0,1] NCHW


pipe.text_encoder.eval(); pipe.unet.eval(); pipe.vae.eval()
with torch.no_grad():
    export(TextEnc(pipe.text_encoder), (torch.randint(0, 49407, (1, 77), dtype=torch.int64),),
           os.path.join(out_fp32, "text_encoder.onnx"), ["input_ids"], ["last_hidden_state"], None)
    export(Unet(pipe.unet),
           (torch.randn(1, 4, 64, 64), torch.tensor([999.0]), torch.randn(1, 77, 768)),
           os.path.join(out_fp32, "unet.onnx"),
           ["sample", "timestep", "encoder_hidden_states"], ["out_sample"],
           {"sample": {2: "h", 3: "w"}, "out_sample": {2: "h", 3: "w"}})
    export(VaeDec(pipe.vae), (torch.randn(1, 4, 64, 64),),
           os.path.join(out_fp32, "vae_decoder.onnx"), ["latent_sample"], ["image"],
           {"latent_sample": {2: "h", 3: "w"}, "image": {2: "H", 3: "W"}})
print("Done. Next: python 02_quantize.py --deploy", args.out)
