"""Fine-tune SD 1.5 with LoRA on your prepared dataset.
Fits in 8 GB VRAM at 512px with gradient checkpointing + fp16.

Usage:
  python 06_train_lora.py --dataset data/coco_lora --out lora_out --steps 3000

After training, fuse the new LoRA and re-export:
  python 06_train_lora.py --dataset data/coco_lora --out lora_out --steps 3000 --export-after
"""
import argparse
import os

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
import datasets as hf_datasets

ap = argparse.ArgumentParser()
ap.add_argument(
    "--model",    default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--dataset",  default="data/lora_dataset")
ap.add_argument("--out",      default="lora_out")
ap.add_argument("--steps",    type=int, default=3000)
ap.add_argument("--bs",       type=int, default=1,
                help="batch size (1 fits 8 GB)")
ap.add_argument("--lr",       type=float, default=1e-4)
ap.add_argument("--rank",     type=int, default=4,
                help="LoRA rank; 4 fits 8 GB, 8 for more capacity")
ap.add_argument("--save-every", type=int, default=500)
ap.add_argument("--export-after", action="store_true",
                help="re-export ONNX after training")
ap.add_argument("--deploy",   default="deploy_lora")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)

accelerator = Accelerator(
    gradient_accumulation_steps=4,
    mixed_precision="fp16",
    project_config=ProjectConfiguration(
        project_dir=args.out, logging_dir=os.path.join(args.out, "logs")),
)

# ── load components (frozen) ──────────────────────────────────────────────────
tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
text_enc = CLIPTextModel.from_pretrained(
    args.model, subfolder="text_encoder").requires_grad_(False)
vae = AutoencoderKL.from_pretrained(
    args.model, subfolder="vae").requires_grad_(False)
unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")
noise_sch = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

# ── attach LoRA to UNet attention layers only ─────────────────────────────────
lora_cfg = LoraConfig(r=args.rank, lora_alpha=args.rank,
                      target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                      lora_dropout=0.05)
unet = get_peft_model(unet, lora_cfg)
unet.enable_gradient_checkpointing()
unet.print_trainable_parameters()

# ── dataset ───────────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


class LoraDS(TorchDataset):
    def __init__(self, path):
        self.ds = hf_datasets.load_from_disk(path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return {"pixel_values": tf(row["image"]),
                "caption":      row["caption"]}


def collate(batch):
    px = torch.stack([b["pixel_values"] for b in batch])
    ids = tokenizer([b["caption"] for b in batch],
                    padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").input_ids
    return {"pixel_values": px, "input_ids": ids}


loader = DataLoader(LoraDS(args.dataset), batch_size=args.bs,
                    shuffle=True, collate_fn=collate, num_workers=0)

# ── optimiser ─────────────────────────────────────────────────────────────────
opt = torch.optim.AdamW(unet.parameters(), lr=args.lr)
unet, opt, loader = accelerator.prepare(unet, opt, loader)
text_enc.to(accelerator.device)
vae.to(accelerator.device)

# ── training loop ─────────────────────────────────────────────────────────────
global_step = 0
unet.train()
while global_step < args.steps:
    for batch in loader:
        with accelerator.accumulate(unet):
            # encode images to latent space
            with torch.no_grad():
                lat = vae.encode(batch["pixel_values"].to(
                    dtype=torch.float16)).latent_dist.sample()
                lat = lat * vae.config.scaling_factor
                enc = text_enc(batch["input_ids"])[0]

            noise = torch.randn_like(lat)
            t = torch.randint(0, noise_sch.config.num_train_timesteps,
                              (lat.shape[0],), device=lat.device).long()
            noisy = noise_sch.add_noise(lat, noise, t)

            pred = unet(noisy, t, enc).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

        global_step += 1
        if global_step % 50 == 0:
            print(f"step {global_step}/{args.steps}  loss={loss.item():.4f}")
        if global_step % args.save_every == 0:
            path = os.path.join(args.out, f"checkpoint_{global_step}")
            accelerator.unwrap_model(unet).save_pretrained(path)
            print(f"Saved checkpoint: {path}")
        if global_step >= args.steps:
            break

# ── save final LoRA weights ───────────────────────────────────────────────────
final = os.path.join(args.out, "final")
accelerator.unwrap_model(unet).save_pretrained(final)
print(f"Training done. LoRA saved to {final}")

# ── optional: re-export to ONNX so the phone app uses the new weights ─────────
if args.export_after:
    import subprocess
    import sys
    subprocess.run([sys.executable, "01_export_onnx.py",
                    "--out", args.deploy,
                    "--lora", final,
                    "--taesd"], check=True)
    subprocess.run([sys.executable, "02b_export_fp16.py",
                    "--deploy", args.deploy], check=True)
    print(f"Re-exported to {args.deploy}. Push to phone with adb.")
