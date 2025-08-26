from flask import Blueprint, request, abort, send_file, current_app
import io
from PIL import Image
import requests
import rembg
from typing import Optional
import numpy as np  # Added this line

bp = Blueprint("matting", __name__, url_prefix="/matting")
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
    model = request.args.get("model", "general")
    if not url:
        abort(400, "image_url is required")
    img_bytes = _fetch_image(url)
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    except Exception:
        abort(415, "unsupported image format")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    session = SESSION if model != "human" else rembg.new_session("isnet-general-human-seg")
    try:
        matted = rembg.remove(
            im,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=140,
            alpha_matting_background_threshold=60,
            alpha_matting_erode_size=35,
        )
        if matted.mode != "RGBA":
            matted = matted.convert("RGBA")
        matted.save("/tmp/matting_debug.png", format="PNG")
        current_app.logger.info("Matting debug saved to /tmp/matting_debug.png")
    except Exception as e:
        current_app.logger.exception("rembg failed")
        abort(500, f"matting failed: {e}")
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