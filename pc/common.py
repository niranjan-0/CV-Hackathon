"""Shared code: numpy LCM scheduler, ONNX Runtime text-to-image pipeline, memory probe, prompts."""
import json
import os
import threading
import time

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

# Try to let onnxruntime-gpu borrow CUDA libraries from torch (avoids DLL-not-found warnings)
try:
    import torch  # noqa: F401
    ort.preload_dlls()
except Exception:
    pass


def ort_providers(device="cpu"):
    """Return the provider list for onnxruntime.  device: 'cpu' | 'cuda'"""
    if device == "cuda":
        return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# 12 complex prompts used consistently across all benchmark runs
PROMPTS = [
    "a red cube on top of a blue sphere, next to a green pyramid, on a wooden table",
    "two cats sitting on a sofa, the left one is black and the right one is orange, a lamp behind them",
    "a small white boat on a calm lake in front of snow-capped mountains at sunrise",
    "a bowl of fresh strawberries to the left of a glass of milk on a rustic kitchen counter",
    "a yellow bicycle leaning against a brick wall, a potted plant on the right side",
    "a futuristic city skyline at night with neon lights reflecting on a rainy street",
    "a golden retriever wearing sunglasses sitting in a convertible car, palm trees in the background",
    "an astronaut riding a horse on a desert planet with two moons in the sky",
    "a cozy wooden cabin in a pine forest, smoke rising from the chimney, snow on the ground",
    "a plate of sushi with chopsticks on the right and a cup of green tea on the left",
    "three colorful hot air balloons floating above a green valley with a river",
    "a vintage red car parked in front of a blue house with a white picket fence",
]


class PeakRSS:
    """Samples process RSS in a background thread and keeps the peak (MB)."""

    def __init__(self, interval=0.02):
        self.p = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0.0
        self._run = False

    def __enter__(self):
        self._run = True
        self.peak = self.p.memory_info().rss / 2 ** 20
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def _loop(self):
        while self._run:
            self.peak = max(self.peak, self.p.memory_info().rss / 2 ** 20)
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._run = False
        self.t.join()
        self.peak = max(self.peak, self.p.memory_info().rss / 2 ** 20)


class LCMScheduler:
    """Numpy port of diffusers LCMScheduler (epsilon prediction + boundary-condition scalings).
    This is the reference implementation for the Kotlin port in LcmScheduler.kt."""

    def __init__(self, steps, original_steps=50, timestep_scaling=10.0, sigma_data=0.5):
        betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2
        self.ac = np.cumprod(1.0 - betas)
        k = 1000 // original_steps
        origin = np.arange(1, original_steps + 1) * k - 1   # 19, 39, ..., 999
        skip = len(origin) // steps
        self.timesteps = origin[::-skip][:steps]             # 4 steps -> [999, 759, 519, 279]
        self.scaling = timestep_scaling
        self.sd = sigma_data

    def step(self, eps, i, sample, rng):
        t = int(self.timesteps[i])
        last = i + 1 >= len(self.timesteps)
        a_t = self.ac[t]
        a_prev = 1.0 if last else self.ac[int(self.timesteps[i + 1])]
        st = t * self.scaling
        c_skip = self.sd ** 2 / (st ** 2 + self.sd ** 2)
        c_out  = st / np.sqrt(st ** 2 + self.sd ** 2)
        x0 = (sample - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t)
        denoised = c_out * x0 + c_skip * sample
        if last:
            return denoised.astype(np.float32)
        noise = rng.standard_normal(sample.shape).astype(np.float32)
        return (np.sqrt(a_prev) * denoised + np.sqrt(1 - a_prev) * noise).astype(np.float32)


class OrtT2I:
    """ONNX Runtime text-to-image pipeline.
    deploy_dir/{vocab.json, merges.txt, config.json, <variant>/{text_encoder,unet,vae_decoder}.onnx}
    """

    def __init__(self, deploy_dir, variant="fp16", threads=None, providers=None, device="cpu"):
        from transformers import CLIPTokenizer

        self.dir = deploy_dir
        self.variant = variant
        self.cfg = json.load(open(os.path.join(deploy_dir, "config.json")))
        self.tok = CLIPTokenizer(
            os.path.join(deploy_dir, "vocab.json"),
            os.path.join(deploy_dir, "merges.txt"),
        )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        prov = providers or ort_providers(device)
        d = os.path.join(deploy_dir, variant)
        mk = lambda n: ort.InferenceSession(
            os.path.join(d, n + ".onnx"), so, providers=prov
        )
        self.te, self.unet, self.vae = mk("text_encoder"), mk("unet"), mk("vae_decoder")
        self.active = self.unet.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in self.active:
            print("WARNING: CUDA requested but ONNX Runtime fell back to CPU. Active:", self.active)

    def generate(self, prompt, steps=4, size=512, seed=0):
        t = {}
        t0 = time.perf_counter()

        ids = self.tok(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="np"
        ).input_ids.astype(np.int64)
        emb = self.te.run(None, {"input_ids": ids})[0].astype(np.float32)
        t["text_ms"] = (time.perf_counter() - t0) * 1000

        rng = np.random.default_rng(seed)
        lat = size // 8
        x   = rng.standard_normal((1, 4, lat, lat)).astype(np.float32)
        sch = LCMScheduler(steps)

        t1 = time.perf_counter()
        for i, ts in enumerate(sch.timesteps):
            eps = self.unet.run(None, {
                "sample":                x,
                "timestep":              np.array([ts], dtype=np.float32),
                "encoder_hidden_states": emb,
            })[0].astype(np.float32)
            x = sch.step(eps, i, x, rng)
        t["unet_ms"] = (time.perf_counter() - t1) * 1000

        t2  = time.perf_counter()
        img = self.vae.run(None, {
            "latent_sample": (x / self.cfg["vae_scale"]).astype(np.float32)
        })[0]
        t["vae_ms"]   = (time.perf_counter() - t2) * 1000
        t["total_ms"] = (time.perf_counter() - t0) * 1000

        img = (np.clip(img[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
        return Image.fromarray(img), t


def dir_size_mb(path):
    total = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            total += os.path.getsize(os.path.join(r, f))
    return total / 2 ** 20
