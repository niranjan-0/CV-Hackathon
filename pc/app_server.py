"""
Minimal Flask server that exposes the existing OrtT2I pipeline over HTTP.
Run from the pc/ folder (where common.py and your deploy/ folder live):

    pip install flask
    python app_server.py --deploy deploy --ort-device cuda

Then open  http://localhost:5000  in a browser.

Nothing in common.py, 03_benchmark.py or any other script is changed.
"""
import argparse
import base64
import io
import os
import time

from flask import Flask, jsonify, request, send_from_directory

ap = argparse.ArgumentParser()
ap.add_argument("--deploy", default="deploy")
ap.add_argument("--variant", default="fp16")
ap.add_argument("--ort-device", default="cuda")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=5000)
args = ap.parse_args()

from common import OrtT2I  # noqa: E402  (import after argparse so errors show cleanly)

print(f"Loading {args.variant} pipeline on {args.ort_device} ...")
pipe = OrtT2I(args.deploy, args.variant, device=args.ort_device)
print("Providers:", pipe.active)

app = Flask(__name__, static_folder="web_static")


@app.route("/")
def index():
    return send_from_directory("web_static", "index.html")


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
        img, timings = pipe.generate(prompt, steps=steps, size=size, seed=seed)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify({"image": b64, "timings": timings, "seed": seed,
                        "steps": steps, "size": size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("web_static", exist_ok=True)
    app.run(host=args.host, port=args.port, debug=False)
