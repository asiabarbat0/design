import os
import io
from flask import Blueprint, request, abort, send_file, current_app
from PIL import Image, ImageOps
import requests
import rembg
import numpy as np

bp = Blueprint("matting", __name__, url_prefix="/matting")

# --- Try to load YOLO; fall back to rembg if unavailable ---
_YOLO_AVAILABLE = True
_YOLO_ERR = None
try:
    from ultralytics import YOLO
    model_path = "yolov8x-seg.pt"
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO("yolov8x-seg.pt")
except Exception as e:
    _YOLO_AVAILABLE = False
    _YOLO_ERR = e

TIMEOUT = 20
MAX_BYTES = 25 * 1024 * 1024  # 25MB
MAX_SIDE = 2000
SESSION = rembg.new_session()  # default rembg session


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


@bp.get("/preview")
def preview():
    """
    GET /matting/preview?image_url=...&model=rembg|yolo|human
    - rembg: default
    - yolo: uses Ultralytics segmentation if available
    - human: rembg with human segmenter ("isnet-general-human-seg")
    """
    url = request.args.get("image_url")
    model_type = (request.args.get("model") or "rembg").lower()
    if not url:
        abort(400, "image_url is required")

    # Load image → normalize orientation → force RGB
    try:
        img_bytes = _fetch_image(url)
        im = Image.open(io.BytesIO(img_bytes))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        abort(415, "unsupported image format")

    # Constrain very large images for speed
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))

    # ---- YOLO path ----
    if model_type == "yolo" and _YOLO_AVAILABLE:
        try:
            # Ultralytics predict on ndarray; quiet logs
            res = model.predict(source=np.array(im), imgsz=640, conf=0.25, verbose=False)[0]
            masks = getattr(res, "masks", None)
            if masks is None or masks.data is None:
                current_app.logger.warning("YOLO returned no masks; falling back to rembg")
                raise RuntimeError("no_yolo_masks")

            m = masks.data  # torch.Tensor [N, Hm, Wm] in [0,1]
            # -> numpy float32
            m_np = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)

            # Union all instance masks to one [Hm, Wm]
            if m_np.ndim == 3:
                union = m_np.max(axis=0)
            elif m_np.ndim == 2:
                union = m_np
            else:
                current_app.logger.warning(f"YOLO mask unexpected shape {m_np.shape}; falling back to rembg")
                raise RuntimeError("bad_yolo_shape")

            # Convert to uint8 alpha 0..255 and resize to image size (NEAREST to keep edges sharp)
            mask_u8 = (np.clip(union, 0.0, 1.0) * 255.0).astype(np.uint8)
            mask_img = Image.fromarray(mask_u8, mode="L").resize(im.size, Image.NEAREST)

            # Compose alpha into the RGB image -> RGBA (exactly 4 channels)
            rgba = im.copy()
            rgba.putalpha(mask_img)
            matted = rgba

        except Exception as e:
            current_app.logger.warning(f"YOLO segmentation failed ({e}); using rembg fallback")
            # fall through to rembg
            session = SESSION
            matted = rembg.remove(
                im,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=140,
                alpha_matting_background_threshold=60,
                alpha_matting_erode_size=35,
            )

    else:
        # ---- rembg path (default or forced, or YOLO unavailable) ----
        if model_type == "human":
            session = rembg.new_session("isnet-general-human-seg")
        else:
            session = SESSION

        matted = rembg.remove(
            im,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=140,
            alpha_matting_background_threshold=60,
            alpha_matting_erode_size=35,
        )

    # Ensure RGBA
    if matted.mode != "RGBA":
        matted = matted.convert("RGBA")

    # Debug and basic sanity
    try:
        matted.save("/tmp/matting_debug.png", "PNG")
        current_app.logger.info("Matting debug saved to /tmp/matting_debug.png")
    except Exception:
        pass

    alpha = np.array(matted.getchannel("A"))
    if int((alpha < 250).sum()) == 0:
        abort(422, "matting produced no transparency")

    # Stream PNG
    out = io.BytesIO()
    matted.save(out, "PNG")
    out.seek(0)
    resp = send_file(out, mimetype="image/png", as_attachment=False, download_name="preview.png")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Content-Length"] = str(len(out.getvalue()))
    return resp
