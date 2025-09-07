import os
from ultralytics import YOLO
from flask import Blueprint, request, abort, send_file, current_app
import io
from PIL import Image
import requests
import rembg
from typing import Optional
import numpy as np
from scipy import ndimage

bp = Blueprint("matting", __name__, url_prefix="/matting")

# Model setup
model_path = "yolov8x-seg.pt"
if not os.path.exists(model_path):
    model = YOLO("yolov8x-seg.pt")
else:
    model = YOLO(model_path)

TIMEOUT = 20
MAX_BYTES = 25 * 1024 * 1024  # 25MB
MAX_SIDE = 2000
SESSION = rembg.new_session()

def _fetch_image(url: str) -> bytes:
    headers = {"User-Agent": "designstream-matting/1.0", "Accept": "image/*"}
    with requests.get(url, headers=headers, timeout=TIMEOUT, stream=True) as r:
        r.raise_for_status()
        cl = r.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BYTES:
            abort(413, f"image too large: {cl} > {MAX_BYTES}")
        buf, read = io.BytesIO(), 0
        for chunk in r.iter_content(64 * 1024):
            read += len(chunk)
            if read > MAX_BYTES:
                abort(413, f"image too large: streamed > {MAX_BYTES}")
            buf.write(chunk)
        return buf.getvalue()

def _normalize_unsplash(url: str) -> str:
    if "unsplash.com" in url and not url.startswith("https://images.unsplash.com"):
        return url.replace("www.unsplash.com", "images.unsplash.com")
    return url

@bp.get("/preview")
def preview():
    url = request.args.get("image_url")
    model_type = request.args.get("model", "rembg")
    if not url:
        abort(400, "image_url is required")
    
    img_bytes = _fetch_image(url)
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    except Exception:
        abort(415, "unsupported image format")
    
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    
    if model_type.lower() == "yolo":
        results = model(im)
        mask = results[0].masks.data[0].cpu().numpy() if results[0].masks else None
        if mask is not None:
            alpha = (mask * 255).astype(np.uint8)
            h, w = im.size
            # Use ndimage.zoom with proper shape preservation
            alpha_resized = ndimage.zoom(alpha, (h / alpha.shape[0], w / alpha.shape[1]), order=1, mode='constant', prefilter=True).astype(np.uint8)
            if alpha_resized.shape != (h, w):
                current_app.logger.error(f"Resized alpha shape {alpha_resized.shape} does not match image shape {im.size}")
                alpha_resized = np.resize(alpha_resized, (h, w))
            alpha_3d = np.zeros((h, w, 1), dtype=np.uint8)
            alpha_3d[:, :, 0] = np.clip(alpha_resized, 0, 255)
            matted = Image.fromarray(np.dstack((np.array(im), alpha_3d[:, :, 0])))
        else:
            current_app.logger.warning("YOLO segmentation failed, falling back to rembg")
            matted = rembg.remove(im, session=SESSION, alpha_matting=True, alpha_matting_foreground_threshold=140, alpha_matting_background_threshold=60, alpha_matting_erode_size=35)
    else:
        session = SESSION if model_type.lower() != "human" else rembg.new_session("isnet-general-human-seg")
        matted = rembg.remove(im, session=session, alpha_matting=True, alpha_matting_foreground_threshold=140, alpha_matting_background_threshold=60, alpha_matting_erode_size=35)

    if matted.mode != "RGBA":
        matted = matted.convert("RGBA")
    matted.save("/tmp/matting_debug.png", format="PNG")
    current_app.logger.info("Matting debug saved to /tmp/matting_debug.png")

    alpha = np.array(matted.getchannel("A"))
    if int((alpha < 250).sum()) == 0:
        abort(422, "matting produced no transparency")

    out = io.BytesIO()
    matted.save(out, format="PNG")
    data = out.getvalue()
    resp = send_file(io.BytesIO(data), mimetype="image/png", as_attachment=False, download_name="preview.png")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Content-Length"] = str(len(data))
    return resp