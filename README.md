# T2I Mobile: resource-efficient text-to-image (SD 1.5 + LCM-LoRA, ONNX, Android)

```
pc/        Python: export -> quantize -> benchmark (CLIP, latency, RAM, size, quality-vs-cost)
android/   Kotlin app: ONNX Runtime + OpenCV, budget planner, fallback, metric logging
```

**Status:** the Python scripts are syntax-checked and the LCM timesteps were verified (4 steps = 999/759/519/279),
but nothing here has been run end-to-end: the model export and the Android build were not possible in the
authoring sandbox. Expect to fix small issues on first run. All performance numbers must be measured by you;
none are claimed here.

## 1. PC / Colab (needs internet, ~16 GB RAM for the UNet export)

For CPU-only execution, install `requirements.txt`. For NVIDIA CUDA execution on Windows,
use a clean virtual environment and install the matching CUDA PyTorch build and GPU ONNX Runtime;
do not install both `onnxruntime` and `onnxruntime-gpu` in the same environment:

```powershell
cd pc
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
python -c "import torch, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())"
```

The provider check should list `CUDAExecutionProvider` and PyTorch should report `True`. If not,
use `--ort-device cpu` or verify the NVIDIA driver.

```bash
cd pc
# CPU-only setup; skip this install command if using the CUDA setup above.
pip install -r requirements.txt
# After installing the CPU or CUDA requirements, export and quantize:
python 01_export_onnx.py --out deploy --taesd      # LCM-LoRA fused, tiny VAE; creates deploy/fp32
python 02_quantize.py --deploy deploy              # creates deploy/fp16 and deploy/int8, prints sizes
python 02_quantize.py --deploy deploy --int8-conv  # optional: also quantize Conv (smaller; may be slower on ARM)
python 03_benchmark.py --deploy deploy --out results --seeds 2
```
Run the export twice (with and without `--taesd`, different `--out`) if you want the TAESD-vs-SD-VAE ablation.

`results/results.md`, `results.csv` and `quality_vs_cost.png` are the quantitative comparison with the baseline
(SD 1.5 fp32, 25 DPM-Solver++ steps, CFG 7.5). Optional FID: `--fid-ref <folder of real images>` (needs
`pip install clean-fid`; with about 24 generated images FID is only indicative).

Optimization strategies demonstrated: (1) step reduction via LCM distillation (LoRA fused, 25 steps with CFG
down to 4 steps, no CFG, so about 12x fewer UNet evaluations), (2) FP16 and INT8 quantization, (3) TAESD
latent-space/lightweight decoder, (4) resolution scaling (256/384/512) plus Lanczos upscale.
Adaptive part: `BudgetPlanner.kt`.

## Deploy the PC website to Render

Create a Render Blueprint from this repository and deploy `render.yaml` at the repository root.
The service installs only the CPU inference dependencies and starts the Flask app with Gunicorn.
The website and `/healthz` are available without loading the model.

ONNX model binaries are excluded from Git, so generation will return HTTP 503 until Render can
read a complete model folder. Upload the exported `pc/deploy` folder (including the selected
variant's `.onnx` and `.onnx.data` files) to a Render persistent disk, then set
`T2I_DEPLOY_DIR` to that folder's absolute mount path and `T2I_VARIANT` to `fp16`, `fp32`, or
`int8` in the service environment. The instance must have enough memory for the selected model;
CPU generation may also take several minutes.

## 2. Phone

1. Open `android/` in Android Studio (let it create the Gradle wrapper / sync). minSdk 26, arm64 only.
2. Run on the phone. Check the app log line "Models dir".
3. Push the models (not bundled in the APK, too large):
   ```bash
   adb push pc/deploy/. /sdcard/Android/data/com.example.t2imobile/files/models/
   ```
   Folder must contain `vocab.json merges.txt config.json fp16/ int8/` (`fp32/` optional).
4. Restart the app. Pick Budget / Model / Provider, press Generate.
   The FP32 model uses low-strength classifier-free guidance (1.5) and applies the negative
   prompt, at the cost of a second UNet pass per step. FP16 and INT8 keep single-pass LCM.

Every run appends latency per stage and peak PSS / native heap to `bench_log.csv` in
`/sdcard/Android/data/com.example.t2imobile/files/`. Pull it with `adb pull` for the report.
Cross-check with `adb shell dumpsys meminfo com.example.t2imobile`.

## 3. What is where (maps to the problem statement)

| Requirement | Where |
|---|---|
| Resizing / resolution optimisation | `BudgetPlanner` ladder, `PostProcess.enhance(target)`, benchmark 256/384/512 |
| Colour correction, enhancement, edge-aware post | `PostProcess.kt`, `pc/cvpost.py` (gray-world WB, CLAHE, bilateral, Canny-masked unsharp) |
| Image quality assessment, artifact detection/correction | `PostProcess.assess` (Laplacian variance, clipping, contrast, colour cast) and gamma/WB fixes |
| Lightweight diffusion + pretrained adaptation | SD 1.5 + fused LCM-LoRA (+ TAESD) |
| Prompt encoding, conditional generation | `ClipTokenizer.kt`, text encoder ONNX, UNet conditioned on embeddings |
| >= 2 optimizations, baseline comparison | `03_benchmark.py` |
| CLIP, FID, latency, memory, size, quality-vs-cost | `03_benchmark.py` (PC), `bench_log.csv` (phone) |
| Complex prompts | `PROMPTS` in `pc/common.py` (multi-object, attributes, spatial) |
| Degradation study | step ablation 1/2/3/4/8 and res 256/384/512, int8 vs fp16 vs fp32 |
| Adaptive / fallback | `BudgetPlanner.plan/downgrade`, retry loop in `MainActivity` |

## 4. Known risks (check these first if something breaks)

- **FP16 on the ORT Android CPU EP** has limited native kernels and can be slower than INT8. Benchmark both and
  report the winner. NNAPI with a ~1.7 GB UNet often fails or falls back; the app reports it.
- **INT8 dynamic quantization of MatMul/Gemm only** leaves convolutions in fp32, so the UNet is about 1.8-2 GB, not 1 GB.
  `--int8-conv` shrinks it further but can hurt speed/quality. Measure.
- **SD VAE in fp16 can give black images**, so the VAE is kept fp32 in all variants.
- **UNet export**: needs `dynamo=False` (handled; torch < 2.9 pinned). If the export complains about size, run on a machine with more RAM.
- Latent noise on the phone uses `java.util.Random`, so images won't match PC images for the same seed.
  To validate the Kotlin port, compare scheduler timesteps and CLIP scores, not pixels.
- 8-12 GB devices usually allow about 3-4 GB for an app; the planner switches to low-memory mode (load/unload each model per stage).
