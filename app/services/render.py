# app/services/render.py
import io
import os
from urllib.parse import urlparse
from typing import List, Tuple
import requests
import cv2
import numpy as np  # Added missing import
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from flask import Blueprint, request, abort, send_file, current_app
from werkzeug.utils import secure_filename

bp = Blueprint("render", __name__, url_prefix="/render")
TIMEOUT = 20
MAX_BYTES = 25 * 1024 * 1024  # 25MB
RENDER_OUT_MAX_W = int(os.getenv("RENDER_OUT_MAX_W", "1920"))  # final output width cap

# ------------------ fetch / open helpers ------------------
def _fetch_bytes(url: str) -> bytes:
    with requests.get(url, timeout=TIMEOUT, stream=True, headers={"User-Agent": "designstream-render/1.0"}) as r:
        r.raise_for_status()
        cl = r.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BYTES:
            abort(413, f"asset too large: {cl} bytes")
        buf, read = io.BytesIO(), 0
        for chunk in r.iter_content(64 * 1024):
            read += len(chunk)
            if read > MAX_BYTES:
                abort(413, "asset too large while streaming")
            buf.write(chunk)
        return buf.getvalue()

def _open_rgba_from_url(url: str) -> Image.Image:
    try:
        data = _fetch_bytes(url)
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # fix orientation
        return img.convert("RGBA")
    except Exception as e:
        abort(415, f"unsupported or unreachable image: {url} ({e})")

def _open_rgba_from_upload(fs) -> Image.Image:
    try:
        # read from the uploaded file object directly
        data = fs.read()
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGBA")
    except Exception as e:
        abort(415, f"bad uploaded image {fs.filename!r}: {e}")

def _basename_from_url(url: str) -> str:
    """
    Safe filename from URL path only (drops query string).
    """
    p = urlparse(url)
    name = os.path.basename(p.path) or "image"
    return secure_filename(name)

# ------------------ compositing helpers ------------------
def _paste_cropped(base: Image.Image, fg: Image.Image, x: int, y: int):
    bx, by = base.size
    fw, fh = fg.size
    cx0, cy0, cx1, cy1 = 0, 0, fw, fh
    dx, dy = x, y
    if dx < 0:
        cx0 = min(fw, -dx); dx = 0
    if dy < 0:
        cy0 = min(fh, -dy); dy = 0
    if dx + (cx1 - cx0) > bx:
        cx1 = cx0 + max(0, bx - dx)
    if dy + (cy1 - cy0) > by:
        cy1 = cy0 + max(0, by - dy)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    crop = fg.crop((cx0, cy0, cx1, cy1))
    base.alpha_composite(crop, (dx, dy))

def _make_shadow(cut: Image.Image, opacity: float = 0.4, blur: int = 12) -> Image.Image:
    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")
    mask = cut.split()[3].point(lambda p: int(p * max(0.0, min(1.0, opacity))))
    black = Image.new("RGBA", cut.size, (0, 0, 0, 255))
    shadow = Image.composite(black, Image.new("RGBA", cut.size, (0, 0, 0, 0)), mask)
    return shadow.filter(ImageFilter.GaussianBlur(blur))

def _constrain_max_width(im: Image.Image, max_w: int) -> Image.Image:
    w, h = im.size
    if w <= max_w:
        return im
    new_h = int(h * (max_w / float(w)))
    return im.resize((max_w, new_h), Image.LANCZOS)

def _inpaint_image(room: Image.Image, mask: Image.Image) -> Image.Image:
    """Inpaint the room image using a mask (white = area to remove)."""
    # Convert PIL images to OpenCV format
    room_np = cv2.cvtColor(np.array(room), cv2.COLOR_RGBA2RGB)
    mask_np = cv2.cvtColor(np.array(mask.convert("RGB")), cv2.COLOR_RGB2GRAY)
    mask_np = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)[1]  # Ensure binary mask
    # Apply inpainting
    inpainted = cv2.inpaint(room_np, mask_np, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_RGB2RGBA))

# ------------------ main route ------------------
@bp.route("/", methods=["GET", "POST"], strict_slashes=False)
def render_preview():
    """
    GET /render?room_url=...&cutouts=url1,url2&anchor=center&fit=0.5&shadow=1
    POST /render (multipart/form-data)
         fields:
           - room: file
           - cutouts: file (repeatable)
           - mask: file (optional)
           - anchor, fit, shadow, opacity, etc. (optional)
    Returns image/png.
    """
    # 1) Load room image
    if request.method == "POST" and "room" in request.files:
        room = _open_rgba_from_upload(request.files["room"])
    else:
        room_url = request.args.get("room_url")
        if not room_url:
            abort(400, "room_url (GET) or room file (POST) is required")
        room = _open_rgba_from_url(room_url)

    # 2) Optional inpainting mask
    mask = None
    if request.method == "POST" and "mask" in request.files:
        mask = _open_rgba_from_upload(request.files["mask"])
    elif request.method == "GET" and "mask_url" in request.args:
        mask_url = request.args.get("mask_url")
        if mask_url.startswith("file://"):
            mask_path = mask_url[7:]  # Remove file:// prefix
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("RGBA")
            else:
                abort(415, f"Mask file not found: {mask_path}")
        else:
            mask = _open_rgba_from_url(mask_url)
    if mask:
        room = _inpaint_image(room, mask)

    # 2) Load cutouts
    cut_images: List[Image.Image] = []
    if request.method == "POST" and "cutouts" in request.files:
        for fs in request.files.getlist("cutouts"):
            if fs and fs.filename:
                cut_images.append(_open_rgba_from_upload(fs))
    else:
        cutouts_param = request.args.get("cutouts", "")
        urls = [u.strip() for u in cutouts_param.split(",") if u.strip()]
        for u in urls:
            cut_images.append(_open_rgba_from_url(u))
    if not cut_images:
        abort(400, "no cutouts provided")

    # 3) Params
    def _f(name: str, default: float) -> float:
        v = request.values.get(name, str(default))
        try:
            return float(v)
        except Exception:
            abort(400, f"invalid float for {name}: {v!r}")

    def _i(name: str, default: int) -> int:
        v = request.values.get(name, str(default))
        try:
            return int(v)
        except Exception:
            abort(400, f"invalid int for {name}: {v!r}")

    scale = _f("scale", 1.0)  # uniform scale for explicit width/height branch
    fit = _f("fit", 0.5)  # fraction of room height
    opacity = _f("opacity", 1.0)  # 0..1
    shadow_on = (request.values.get("shadow", "1").lower() in ("1", "true", "yes"))
    shadow_opacity = _f("shadow_opacity", 0.4)
    shadow_blur = _i("shadow_blur", 12)
    shadow_dx = _i("shadow_dx", 8)
    shadow_dy = _i("shadow_dy", 8)
    anchor = (request.values.get("anchor") or "center").lower()
    allowed_anchors = {
        "center", "topleft", "topright", "bottomleft", "bottomright",
        "top", "bottom", "left", "right"
    }
    if anchor not in allowed_anchors:
        abort(400, f"invalid anchor: {anchor!r}; allowed={sorted(allowed_anchors)}")

    # 4) Compute target size for the first cutout
    W, H = room.size
    th = max(1, int(H * max(0.0, min(1.0, fit))))
    # aspect from first cutout
    base_cut = cut_images[0]
    tw = max(1, int((th / base_cut.height) * base_cut.width))
    # cap cutouts to something sensible (e.g., 1920)
    if max(tw, th) > RENDER_OUT_MAX_W:
        r = RENDER_OUT_MAX_W / float(max(tw, th))
        tw, th = max(1, int(tw * r)), max(1, int(th * r))

    # 5) Prepare cutouts (resize, opacity)
    prepared: List[Image.Image] = []
    for c in cut_images:
        ci = c.resize((tw, th), Image.LANCZOS) if c.size != (tw, th) else c.copy()
        if opacity < 1.0:
            a = ci.getchannel("A").point(lambda p: int(p * opacity))
            ci.putalpha(a)
        prepared.append(ci)
    w, h = prepared[0].size

    # 6) Anchor to get base position
    anchors = {
        "center": ((W - w) // 2, (H - h) // 2),
        "topleft": (0, 0),
        "topright": (W - w, 0),
        "bottomleft": (0, H - h),
        "bottomright": (W - w, H - h),
        "top": ((W - w) // 2, 0),
        "bottom": ((W - w) // 2, H - h),
        "left": (0, (H - h) // 2),
        "right": (W - w, (H - h) // 2),
    }
    x0, y0 = anchors[anchor]

    # 7) Composite
    out = room.copy()
    if shadow_on:
        for i, cut in enumerate(prepared):
            x = x0 + (i * int(w * 0.1))
            y = y0 + (i * int(h * 0.1))
            sh = _make_shadow(cut, opacity=shadow_opacity, blur=shadow_blur)
            _paste_cropped(out, sh, x + shadow_dx, y + shadow_dy)
            _paste_cropped(out, cut, x, y)
    else:
        for i, cut in enumerate(prepared):
            x = x0 + (i * int(w * 0.1))
            y = y0 + (i * int(h * 0.1))
            _paste_cropped(out, cut, x, y)

    # Debug save
    debug_path = "/tmp/debug_render.png"
    try:
        out.save(debug_path, "PNG")
        current_app.logger.info(f"Debug render saved to {debug_path}")
    except Exception as e:
        current_app.logger.error(f"Failed to save debug render: {e}")

    # 8) Final output width cap (default 1920)
    out = _constrain_max_width(out, RENDER_OUT_MAX_W)

    # 9) Stream PNG
    buf = io.BytesIO()
    out.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", as_attachment=False, download_name="render.png")