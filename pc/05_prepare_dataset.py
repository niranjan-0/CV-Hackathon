"""Prepare a unified image-caption dataset from COCO, Flickr30K, or any HuggingFace dataset.
Saves to a local HuggingFace dataset folder that the training script reads.

Examples:
  # COCO (download val2017.zip + annotations_trainval2017.zip from cocodataset.org)
  python 05_prepare_dataset.py --source coco \
      --coco-images D:/data/coco/train2017 \
      --coco-captions D:/data/coco/annotations/captions_train2017.json \
      --out data/coco_lora --n 10000

  # Flickr30K via HuggingFace (needs HF login: huggingface-cli login)
  python 05_prepare_dataset.py --source hf \
      --hf-name nlphuji/flickr30k --hf-split test \
      --caption-col caption --image-col image \
      --out data/flickr_lora --n 5000

  # Combine both
  python 05_prepare_dataset.py --source coco --out data/combined --n 10000 ...
  python 05_prepare_dataset.py --source hf   --out data/combined --n 5000  --append ...
"""
import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image
from datasets import Dataset, load_dataset
import datasets as hf_datasets

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["coco", "hf"], required=True)
ap.add_argument("--coco-images")
ap.add_argument("--coco-captions")
ap.add_argument("--hf-name")
ap.add_argument("--hf-split", default="test")
ap.add_argument("--caption-col", default="caption")
ap.add_argument("--image-col",   default="image")
ap.add_argument("--out",  default="data/lora_dataset")
ap.add_argument("--n",    type=int, default=5000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--append", action="store_true",
                help="add to existing dataset folder")
ap.add_argument("--min-size", type=int, default=256,
                help="skip images smaller than this")
args = ap.parse_args()

random.seed(args.seed)


def centre_crop_512(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < args.min_size:
        return None
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
    return img.resize((512, 512), Image.LANCZOS)


def first_caption(c):
    if isinstance(c, (list, tuple)):
        c = c[0]
    return str(c).strip().replace("\n", " ")


pairs = []

if args.source == "coco":
    data = json.load(open(args.coco_captions, encoding="utf-8"))
    files = {im["id"]: im["file_name"] for im in data["images"]}
    caps = {}
    for a in data["annotations"]:
        caps.setdefault(a["image_id"], a["caption"])
    ids = list(caps.keys())
    random.shuffle(ids)
    for i in ids[:args.n * 2]:           # oversample to allow filtering
        path = os.path.join(args.coco_images, files[i])
        if not os.path.exists(path):
            continue
        img = centre_crop_512(Image.open(path))
        if img is None:
            continue
        pairs.append({"image": img, "caption": first_caption(caps[i])})
        if len(pairs) >= args.n:
            break

elif args.source == "hf":
    ds = load_dataset(args.hf_name, split=args.hf_split)
    idx = list(range(len(ds)))
    random.shuffle(idx)
    for i in idx:
        row = ds[i]
        img = centre_crop_512(row[args.image_col])
        if img is None:
            continue
        pairs.append(
            {"image": img, "caption": first_caption(row[args.caption_col])})
        if len(pairs) >= args.n:
            break

print(f"Prepared {len(pairs)} image-caption pairs")

out = Path(args.out)
if args.append and (out / "dataset_info.json").exists():
    existing = hf_datasets.load_from_disk(str(out))
    combined = hf_datasets.concatenate_datasets(
        [existing, Dataset.from_list(pairs)])
    combined.save_to_disk(str(out))
    print(f"Appended → total {len(combined)}")
else:
    Dataset.from_list(pairs).save_to_disk(str(out))
    print(f"Saved to {out}")
