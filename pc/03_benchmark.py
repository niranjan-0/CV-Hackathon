"""Step 3/5: baseline vs optimized comparison (CLIP score, latency, peak RAM, size, quality-vs-cost).

  python 03_benchmark.py --deploy deploy --out results --seeds 2
  python 03_benchmark.py --deploy deploy --skip-baseline          # if no baseline hardware/time
  python 03_benchmark.py --deploy deploy --fid-ref /path/to/real_images   # optional FID (needs clean-fid)

Configs benchmarked
  baseline        : SD1.5 PyTorch fp32, 25 DPM-Solver++ steps, CFG 7.5, 512px (CPU by default, so RAM is comparable to a phone)
  lcm_fp32_s4     : LCM-LoRA, ONNX fp32, 4 steps
  lcm_fp16_s4     : + fp16
  lcm_int8_s4     : + int8 dynamic
  steps ablation  : int8/fp16 at 1,2,3,4,8 steps
  resolution      : 256/384/512
  *_post          : same images after classical CV post-processing
"""
import argparse
import csv
import gc
import os
import time

import numpy as np
import torch
from PIL import Image

from common import PROMPTS, OrtT2I, PeakRSS, dir_size_mb
import cvpost

ap = argparse.ArgumentParser()
ap.add_argument("--deploy", default="deploy")
ap.add_argument("--out", default="results")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--device", default="cpu", help="baseline (PyTorch) device: cpu | cuda")
ap.add_argument("--ort-device", default="cpu", help="ONNX Runtime device for the LCM configs: cpu | cuda")
ap.add_argument("--skip-baseline", action="store_true")
ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
ap.add_argument("--fid-ref", default=None)
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--variants", default="fp32,fp16,int8")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---------------- CLIP scorer ----------------
from transformers import CLIPModel, CLIPProcessor

clip = CLIPModel.from_pretrained(args.clip).eval()
clip_proc = CLIPProcessor.from_pretrained(args.clip)


@torch.no_grad()
def clip_score(img, prompt):
    inp = clip_proc(text=[prompt], images=img, return_tensors="pt", padding=True, truncation=True)
    ie = clip.get_image_features(pixel_values=inp["pixel_values"])
    te = clip.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
    ie, te = ie / ie.norm(dim=-1, keepdim=True), te / te.norm(dim=-1, keepdim=True)
    return float((ie * te).sum()) * 100.0


rows = []


def record(name, imgs_prompts, lat_ms, peak_mb, size_mb, steps, res, extra=None):
    cs = [clip_score(im, p) for im, p in imgs_prompts]
    sharp = [cvpost.assess(np.array(im))[0]["sharpness"] for im, _ in imgs_prompts]
    colorful = [cvpost.assess(np.array(im))[0]["colorfulness"] for im, _ in imgs_prompts]
    r = dict(config=name, steps=steps, res=res, clip=np.mean(cs), clip_std=np.std(cs),
             latency_ms=np.mean(lat_ms), latency_p95_ms=np.percentile(lat_ms, 95),
             peak_rss_mb=peak_mb, model_mb=size_mb, sharpness=np.mean(sharp), colorfulness=np.mean(colorful))
    r.update(extra or {})
    rows.append(r)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    d = os.path.join(args.out, "images", name)
    os.makedirs(d, exist_ok=True)
    for i, (im, _) in enumerate(imgs_prompts):
        im.save(os.path.join(d, f"{i:03d}.png"))


# ---------------- Baseline: SD1.5 fp32, 25 steps, CFG ----------------
if not args.skip_baseline:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    print("== baseline SD1.5 ==")
    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                                   torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if args.threads:
        torch.set_num_threads(args.threads)
    size_mb = sum(p.numel() * p.element_size() for m in (pipe.unet, pipe.text_encoder, pipe.vae)
                  for p in m.parameters()) / 2**20
    ip, lat = [], []
    with PeakRSS() as pk:
        for p in PROMPTS:
            for s in range(args.seeds):
                g = torch.Generator(args.device).manual_seed(s)
                t = time.perf_counter()
                im = pipe(p, num_inference_steps=25, guidance_scale=7.5, height=512, width=512, generator=g).images[0]
                lat.append((time.perf_counter() - t) * 1000)
                ip.append((im, p))
    peak = pk.peak
    if args.device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**20
    record("baseline_sd15_fp32_s25", ip, lat, peak, size_mb, 25, 512)
    del pipe
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()  # free VRAM before ONNX Runtime allocates its own

# ---------------- Optimized configs ----------------
cfgs = []
for v in args.variants.split(","):
    cfgs.append((v, 4, 512))
for v in ("fp16", "int8"):
    if v in args.variants:
        cfgs += [(v, s, 512) for s in (1, 2, 3, 8)]
        cfgs += [(v, 4, r) for r in (256, 384)]

cache = {}
for v, steps, res in dict.fromkeys(cfgs):
    if v not in cache:
        cache.clear(); gc.collect()
        with PeakRSS() as load_pk:
            cache[v] = OrtT2I(args.deploy, v, threads=args.threads, device=args.ort_device)
        cache["load_peak_" + v] = load_pk.peak
    pipe = cache[v]
    name = f"lcm_{v}_s{steps}_{res}"
    print("==", name)
    pipe.generate(PROMPTS[0], steps, res, 0)  # warm-up
    ip, lat = [], []
    with PeakRSS() as pk:
        for p in PROMPTS:
            for s in range(args.seeds):
                im, t = pipe.generate(p, steps, res, s)
                lat.append(t["total_ms"])
                ip.append((im, p))
    peak = max(pk.peak, cache["load_peak_" + v])
    record(name, ip, lat, peak, dir_size_mb(os.path.join(args.deploy, v)), steps, res)

    if steps == 4 and res == 512:  # classical-CV post-processing variant of the same images
        post = []
        for im, p in ip:
            a = np.array(im)
            post.append((Image.fromarray(cvpost.enhance(a)), p))
        record(name + "_post", post, lat, peak, dir_size_mb(os.path.join(args.deploy, v)), steps, res)

# ---------------- Report ----------------
cols = list(rows[0].keys())
with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

base = next((r for r in rows if r["config"].startswith("baseline")), None)
with open(os.path.join(args.out, "results.md"), "w") as f:
    f.write(f"<!-- baseline device: {args.device}, ONNX Runtime device: {args.ort_device} -->\n")
    f.write("| config | steps | res | CLIP | latency ms | peak RAM MB | size MB | speedup | dCLIP |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        sp = f"{base['latency_ms'] / r['latency_ms']:.1f}x" if base else "-"
        dc = f"{r['clip'] - base['clip']:+.2f}" if base else "-"
        f.write(f"| {r['config']} | {r['steps']} | {r['res']} | {r['clip']:.2f} | {r['latency_ms']:.0f} | "
                f"{r['peak_rss_mb']:.0f} | {r['model_mb']:.0f} | {sp} | {dc} |\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
for r in rows:
    plt.scatter(r["latency_ms"] / 1000, r["clip"])
    plt.annotate(r["config"].replace("lcm_", ""), (r["latency_ms"] / 1000, r["clip"]), fontsize=6)
plt.xscale("log"); plt.xlabel("latency / image (s, log)"); plt.ylabel("CLIP score (x100)")
plt.title("Quality vs computational cost"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(args.out, "quality_vs_cost.png"), dpi=160)

# ---------------- Optional FID ----------------
if args.fid_ref:
    from cleanfid import fid
    with open(os.path.join(args.out, "fid.txt"), "w") as f:
        for r in rows:
            d = os.path.join(args.out, "images", r["config"])
            score = fid.compute_fid(d, args.fid_ref)  # with ~24 images FID is only indicative
            f.write(f"{r['config']}\t{score:.2f}\n")
            print("FID", r["config"], score)
print("Wrote", args.out)
