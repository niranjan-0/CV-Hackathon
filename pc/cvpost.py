"""Classical-CV post-processing + image quality assessment (Python mirror of the Android PostProcess.kt)."""
import cv2
import numpy as np


def assess(rgb):
    """rgb: HxWx3 uint8. Returns metrics dict + list of artifact flags."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    black = float((gray < 10).mean())
    white = float((gray > 245).mean())
    contrast = float(gray.std())
    cast_a, cast_b = float(lab[..., 1].mean() - 128), float(lab[..., 2].mean() - 128)
    rg = rgb[..., 0].astype(np.float32) - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0].astype(np.float32) + rgb[..., 1]) - rgb[..., 2]
    colorful = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    flags = []
    if sharp < 60: flags.append("blurry")
    if black > 0.4: flags.append("dark_region")
    if white > 0.3: flags.append("blown_highlights")
    if contrast < 25: flags.append("low_contrast")
    if max(abs(cast_a), abs(cast_b)) > 8: flags.append("color_cast")
    return dict(sharpness=sharp, black_frac=black, white_frac=white, contrast=contrast,
                cast_a=cast_a, cast_b=cast_b, colorfulness=colorful), flags


def enhance(rgb, flags=None, target=None):
    """Gray-world WB (if cast) -> CLAHE on L -> highlight gamma -> bilateral denoise -> edge-aware unsharp."""
    if flags is None:
        _, flags = assess(rgb)
    out = rgb.copy()
    if "color_cast" in flags:
        m = out.reshape(-1, 3).mean(0)
        out = np.clip(out * (m.mean() / np.maximum(m, 1e-3)), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    clip = 2.5 if "low_contrast" in flags else 1.5
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if "blown_highlights" in flags:
        lut = (255 * (np.arange(256) / 255.0) ** 1.2).astype(np.uint8)
        out = cv2.LUT(out, lut)
    out = cv2.bilateralFilter(out, 5, 25, 5)
    # edge-aware unsharp: sharpen only near edges so flat regions do not get noisy
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 60, 150), np.ones((3, 3), np.uint8))
    mask = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (0, 0), 2.0)[..., None]
    f = out.astype(np.float32)
    detail = f - cv2.GaussianBlur(f, (0, 0), 1.2)
    amount = 1.2 if "blurry" in flags else 0.6
    out = np.clip(f + amount * detail * mask, 0, 255).astype(np.uint8)
    if target and target != out.shape[0]:
        out = cv2.resize(out, (target, target), interpolation=cv2.INTER_LANCZOS4)
    return out
