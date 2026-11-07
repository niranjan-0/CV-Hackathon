"""Step 4: evaluate the LCM ONNX pipeline on the hackathon datasets (MS COCO, Flickr30K, any Hugging Face dataset).

Real captions are used as prompts, the real images are the FID reference.

COCO (official files: captions_val2017.json + val2017/ folder):
  python 04_dataset_eval.py --deploy deploy --source coco \
      --coco-captions D:/data/coco/annotations/captions_val2017.json --coco-images D:/data/coco/val2017 \
      --n 300 --variant fp16 --ort-device cuda --out results_coco

Flickr30K or any Hugging Face dataset (check the exact dataset name/columns on the hackathon link):
  python 04_dataset_eval.py --deploy deploy --source hf --hf-name nlphuji/flickr30k --hf-split test \
      --caption-col caption --image-col image --n 300 --variant fp16 --ort-device cuda --out results_flickr
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from common import OrtT2I

ap = argparse.ArgumentParser()
ap.add_argument("--deploy", default="deploy")
ap.add_argument("--variant", default="fp16")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--ort-device", default="cuda")
ap.add_argument("--source", choices=["coco", "hf"], required=True)
ap.add_argument("--coco-captions")
ap.add_argument("--coco-images")
ap.add_argument("--hf-name")
ap.add_argument("--hf-split", default="test")
ap.add_argument("--caption-col", default="caption")
ap.add_argument("--image-col", default="image")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
ap.add_argument("--no-fid", action="store_true")
ap.add_argument("--out", default="results_ds")
args = ap.parse_args()


def center_crop_resize(im, size):
    im = im.convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((size, size), Image.LANCZOS)


def first_caption(c):
    if isinstance(c, (list, tuple)):
        c = c[0]
    return str(c).strip().replace("\n", " ")


def load_pairs():
    """Returns a list of (PIL image, caption), n items, deterministic for a given --seed."""
    rnd = random.Random(args.seed)
    if args.source == "coco":
        data = json.load(open(args.coco_captions, encoding="utf-8"))
        files = {im["id"]: im["file_name"] for im in data["images"]}
        caps = {}
        for a in data["annotations"]:
            caps.setdefault(a["image_id"], a["caption"])  # first caption of each image
        ids = sorted(caps)
        rnd.shuffle(ids)
        out = []
        for i in ids[: args.n]:
            out.append((Image.open(os.path.join(args.coco_images, files[i])), first_caption(caps[i])))
        return out
    from datasets import load_dataset

    ds = load_dataset(args.hf_name, split=args.hf_split)
    idx = list(range(len(ds)))
    rnd.shuffle(idx)
    out = []
    for i in idx[: args.n]:
        row = ds[i]
        out.append((row[args.image_col], first_caption(row[args.caption_col])))
    return out


os.makedirs(os.path.join(args.out, "real"), exist_ok=True)
os.makedirs(os.path.join(args.out, "gen"), exist_ok=True)

print("Loading dataset samples ...")
pairs = load_pairs()
real, prompts = [], []
for i, (im, cap) in enumerate(pairs):
    r = center_crop_resize(im, args.res)
    r.save(os.path.join(args.out, "real", f"{i:04d}.png"))
    real.append(r)
    prompts.append(cap)
print(f"{len(prompts)} pairs. Example caption: {prompts[0]!r}")

# ---------- CLIP scorer (on GPU if available) ----------
from transformers import CLIPModel, CLIPProcessor

dev = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained(args.clip).to(dev).eval()
proc = CLIPProcessor.from_pretrained(args.clip)


@torch.no_grad()
def clip_score(img, prompt):
    inp = proc(text=[prompt], images=img, return_tensors="pt", padding=True, truncation=True).to(dev)
    ie = clip.get_image_features(pixel_values=inp["pixel_values"])
    te = clip.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
    ie, te = ie / ie.norm(dim=-1, keepdim=True), te / te.norm(dim=-1, keepdim=True)
    return float((ie * te).sum()) * 100.0


# ---------- generate ----------
pipe = OrtT2I(args.deploy, args.variant, device=args.ort_device)
print("ONNX Runtime providers:", pipe.active)
pipe.generate(prompts[0], args.steps, args.res, 0)  # warm-up

rows, gen_scores, real_scores, lat = [], [], [], []
for i, p in enumerate(prompts):
    im, t = pipe.generate(p, args.steps, args.res, args.seed + i)
    im.save(os.path.join(args.out, "gen", f"{i:04d}.png"))
    g, r = clip_score(im, p), clip_score(real[i], p)
    gen_scores.append(g); real_scores.append(r); lat.append(t["total_ms"])
    rows.append(dict(idx=i, caption=p, clip_generated=g, clip_real=r, latency_ms=t["total_ms"]))
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(prompts)}  CLIP gen {np.mean(gen_scores):.2f}  real {np.mean(real_scores):.2f}")

with open(os.path.join(args.out, "per_image.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

summary = dict(source=args.source if args.source == "coco" else args.hf_name, n=len(prompts),
               variant=args.variant, steps=args.steps, res=args.res, device=args.ort_device,
               clip_generated=float(np.mean(gen_scores)), clip_real_images=float(np.mean(real_scores)),
               latency_ms_mean=float(np.mean(lat)), latency_ms_p95=float(np.percentile(lat, 95)))

if not args.no_fid:
    try:
        from cleanfid import fid
        summary["fid"] = float(fid.compute_fid(os.path.join(args.out, "gen"), os.path.join(args.out, "real")))
    except Exception as e:  # keep the CLIP results even if FID fails
        print("FID skipped:", e)

json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
print("Note: with a few hundred images FID is only indicative; use >= 1000 for a stable number.")
