"""Flask server exposing the ONNX text-to-image pipeline over HTTP."""
import argparse
import base64
import io
import os
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

from common import OrtT2I  # noqa: E402  (import after argparse so errors show cleanly)

app = Flask(__name__, static_folder="web_static")
pipe = None
pipe_lock = threading.Lock()


def get_pipe():
    global pipe
    if pipe is None:
        with pipe_lock:
            if pipe is None:
                deploy_dir = os.environ.get("T2I_DEPLOY_DIR", "deploy")
                variant = os.environ.get("T2I_VARIANT", "fp16")
                device = os.environ.get("T2I_ORT_DEVICE", "cpu")
                required = [
                    os.path.join(deploy_dir, "config.json"),
                    os.path.join(deploy_dir, "vocab.json"),
                    os.path.join(deploy_dir, "merges.txt"),
                ]
                for model in ("text_encoder", "unet", "vae_decoder"):
                    required.extend((
                        os.path.join(deploy_dir, variant, model + ".onnx"),
                        os.path.join(deploy_dir, variant, model + ".onnx.data"),
                    ))
                missing = [path for path in required if not os.path.isfile(path)]
                if missing:
                    raise FileNotFoundError(
                        "ONNX model files are missing. Provide the complete model "
                        f"directory with T2I_DEPLOY_DIR. Missing: {', '.join(missing)}"
                    )

                print(f"Loading {variant} pipeline on {device} ...")
                pipe = OrtT2I(deploy_dir, variant, device=device)
                print("Providers:", pipe.active)
    return pipe


@app.route("/")
def index():
    return send_from_directory("web_static", "index.html")


@app.route("/healthz")
def health():
    return jsonify({"status": "ok", "model_loaded": pipe is not None})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    steps = int(data.get("steps", 4))
    size = int(data.get("size", 512))
    seed = int(data.get("seed", int(time.time()) & 0xFFFFFF))
    steps = max(1, min(steps, 8))
    size = size if size in (256, 384, 512) else 512
    try:
        pipeline = get_pipe()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    try:
        img, timings = pipeline.generate(prompt, steps=steps, size=size, seed=seed)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify({"image": b64, "timings": timings, "seed": seed,
                        "steps": steps, "size": size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default=os.environ.get("T2I_DEPLOY_DIR", "deploy"))
    ap.add_argument("--variant", default=os.environ.get("T2I_VARIANT", "fp16"))
    ap.add_argument("--ort-device", default=os.environ.get("T2I_ORT_DEVICE", "cuda"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    args = ap.parse_args()
    os.environ["T2I_DEPLOY_DIR"] = args.deploy
    os.environ["T2I_VARIANT"] = args.variant
    os.environ["T2I_ORT_DEVICE"] = args.ort_device
    os.makedirs("web_static", exist_ok=True)
    app.run(host=args.host, port=args.port, debug=False)
