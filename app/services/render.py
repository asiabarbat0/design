# app/services/render.py
import io
import os
from urllib.parse import urlparse
from typing import List, Tuple
import requests
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from flask import Blueprint, request, abort, send_file, current_app
from werkzeug.utils import secure_filename

# --- (optional) YOLO for auto-placement ---
try:
    from ultralytics import YOLO
    _YOLO = YOLO("yolov8n-seg.pt")  # small + fast
    _YOLO_OK = True
except Exception:
    _YOLO = None
    _YOLO_OK = False

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
        if url.startswith("file://"):
            local_path = url[7:]
            if not os.path.exists(local_path):
                abort(415, f"file not found: {local_path}")
            img = Image.open(local_path)
        else:
            data = _fetch_bytes(url)
            img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # fix orientation
        return img.convert("RGBA")
    except Exception as e:
        abort(415, f"unsupported or unreachable image: {url} ({e})")

def _open_rgba_from_upload(fs) -> Image.Image:
    try:
        data = fs.read()
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGBA")
    except Exception as e:
        abort(415, f"bad uploaded image {fs.filename!r}: {e}")

def _basename_from_url(url: str) -> str:
    p = urlparse(url)
    name = os.path.basename(p.path) or "image"
    return secure_filename(name)

# ------------------ cv / compositing helpers ------------------
def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _rgba_to_rgb_over(bg_bgr: np.ndarray, fg_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    H, W = bg_bgr.shape[:2]
    fh, fw = fg_rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + fw), min(H, y + fh)
    if x0 >= x1 or y0 >= y1:
        return bg_bgr
    cut = fg_rgba[(y0 - y):(y1 - y), (x0 - x):(x1 - x)]
    fg = cut[..., :3].astype(np.float32)
    a  = (cut[..., 3:4].astype(np.float32)) / 255.0
    bg = bg_bgr[y0:y1, x0:x1].astype(np.float32)
    out = fg * a + bg * (1.0 - a)
    bg_bgr[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return bg_bgr

def _make_shadow_multiplier(mask: np.ndarray, blur_px: int = 20, opacity: float = 0.35, y_offset: int = 10) -> np.ndarray:
    """mask: HxW uint8 0..255 -> returns HxWx1 float multiplier (0..1)."""
    sh = np.clip(mask.astype(np.uint8), 0, 255)
    if y_offset:
        M = np.float32([[1, 0, 0], [0, 1, y_offset]])
        sh = cv2.warpAffine(sh, M, (sh.shape[1], sh.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    sh = cv2.GaussianBlur(sh, (0, 0), blur_px)
    sh = (sh.astype(np.float32)/255.0) * opacity
    return 1.0 - sh[..., None]

def _constrain_max_width(im: Image.Image, max_w: int) -> Image.Image:
    w, h = im.size
    if w <= max_w:
        return im
    new_h = int(h * (max_w / float(w)))
    return im.resize((max_w, new_h), Image.LANCZOS)

def _perspective_transform(cut: Image.Image, src_points: List[Tuple[int, int]], dst_points: List[Tuple[int, int]]) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Warp RGBA cutout from src_points -> dst_points.
    Returns (warped_image, top_left_xy_for_paste).
    """
    src_pts = np.float32(src_points)
    dst_pts = np.float32(dst_points)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    cut_rgba = np.array(cut.convert("RGBA"))
    h, w = cut_rgba.shape[:2]

    # Output canvas size = bbox of destination quad
    min_x, min_y = np.min(dst_pts[:, 0]), np.min(dst_pts[:, 1])
    max_x, max_y = np.max(dst_pts[:, 0]), np.max(dst_pts[:, 1])
    out_w = int(max(1, np.ceil(max_x - min_x)))
    out_h = int(max(1, np.ceil(max_y - min_y)))

    # Shift so top-left of bbox is at (0,0)
    T = np.float32([[1, 0, -min_x],
                    [0, 1, -min_y],
                    [0, 0, 1]])
    MT = T @ M

    warped = cv2.warpPerspective(cut_rgba, MT, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return Image.fromarray(warped), (int(min_x), int(min_y))

# --------- mask helper: use alpha if present ----------
def _mask_from_image(img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    """
    Produce an 8-bit mask (0..255) resized to `size`.
    - If RGBA, use alpha channel.
    - Else, infer from luminance (non-white -> mask).
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    if arr.ndim == 3 and arr.shape[2] == 4:
        mask = arr[..., 3]
    else:
        gray = cv2.cvtColor(np.array(rgba.convert("RGB")), cv2.COLOR_RGB2GRAY)
        mask = cv2.threshold(255 - gray, 8, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask.astype(np.uint8)

# --------- AUTO placement helpers ----------
def _detect_couch_mask(room_rgba: Image.Image) -> np.ndarray | None:
    """Return uint8 mask (HxW) for detected couch/sofa, or None."""
    if not _YOLO_OK:
        return None
    im_rgb = room_rgba.convert("RGB")
    res = _YOLO.predict(np.array(im_rgb), imgsz=640, conf=0.25, verbose=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return None

    names = getattr(_YOLO, "names", {})
    best = None  # (area, mask)
    for i in range(len(res.boxes)):
        cls_idx = int(res.boxes.cls[i])
        label = names.get(cls_idx, str(cls_idx))
        if label not in ("couch", "sofa"):
            continue
        mk = res.masks.data[i].cpu().numpy()
        if mk.ndim == 3:
            mk = mk.squeeze(0)
        mk = (mk > 0.5).astype(np.uint8)
        area = int(mk.sum())
        if best is None or area > best[0]:
            best = (area, mk)

    if best is None:
        return None

    # Resize mask to room size
    W, H = room_rgba.size
    mask = cv2.resize(best[1] * 255, (W, H), interpolation=cv2.INTER_LINEAR)
    return mask.astype(np.uint8)

# ... (keep all existing imports and helpers as provided) ...

def _auto_replace(room: Image.Image, cutout: Image.Image,
                 fudge_scale: float = 1.0,  # Increased from 0.95 to match size
                 shadow_blur: int = 26,
                 shadow_opacity: float = 0.38,
                 shadow_dy: int = 10,
                 fallback_fit: float = 0.62) -> Image.Image:
    """
    Detect couch in the room, inpaint it, and place the new cutout
    bottom-aligned and width-matched. Falls back to a sensible manual center/bottom placement.
    """
    W, H = room.size
    room_bgr = _pil_to_bgr(room)
    mask = _detect_couch_mask(room)
    if mask is not None and mask.any():
        # Inpaint the old couch with larger dilation
        dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)), iterations=2)  # Increased dilation
        imask = (dil > 0).astype(np.uint8) * 255
        room_bgr = cv2.inpaint(room_bgr, imask, 3, cv2.INPAINT_TELEA)
        # Compute bbox of detected couch
        ys, xs = np.where(dil > 0)  # Use dilated mask for bbox
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox_w = max(1, x1 - x0)
        bbox_h = max(1, y1 - y0)
        # Scale cutout to match bbox dimensions
        ow, oh = cutout.size
        scale_w = bbox_w / float(ow)
        scale_h = bbox_h / float(oh)
        scale = min(scale_w, scale_h) * fudge_scale  # Use the tighter fit
        tw, th = max(1, int(ow * scale)), max(1, int(oh * scale))
        ci = cutout.resize((tw, th), Image.LANCZOS)
        # Align new couch to the bottom of the bbox
        x = int(x0 + (bbox_w - tw) / 2)
        y = int(y1 - th)  # Bottom alignment
        # Soft shadow under object (multiply)
        alpha = np.array(ci)[..., 3]
        hole = cv2.dilate(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), 1)
        mult = _make_shadow_multiplier(hole, blur_px=shadow_blur, opacity=shadow_opacity, y_offset=shadow_dy)
        y0p, x0p = max(0, y), max(0, x)
        y1p, x1p = min(room_bgr.shape[0], y + th), min(room_bgr.shape[1], x + tw)
        if x0p < x1p and y0p < y1p:
            m = mult[(y0p - y):(y1p - y), (x0p - x):(x1p - x)]
            room_bgr[y0p:y1p, x0p:x1p] = np.clip(room_bgr[y0p:y1p, x0p:x1p].astype(np.float32) * m, 0, 255).astype(np.uint8)
        out_bgr = _rgba_to_rgb_over(room_bgr, np.array(ci), x, y)
        return _bgr_to_pil(out_bgr)
    # ---- Fallback (no YOLO or no couch found): bottom-center with a reasonable size
    target = int(min(W, H) * fallback_fit)
    ow, oh = cutout.size
    s = target / float(max(ow, oh))
    tw, th = max(1, int(ow * s)), max(1, int(oh * s))
    x = (W - tw) // 2
    y = H - th - int(0.03 * H)
    ci = cutout.resize((tw, th), Image.LANCZOS)
    # shadow
    alpha = np.array(ci)[..., 3]
    hole = cv2.dilate(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), 1)
    mult = _make_shadow_multiplier(hole, blur_px=shadow_blur, opacity=shadow_opacity, y_offset=shadow_dy)
    y0p, x0p = max(0, y), max(0, x)
    y1p, x1p = min(room_bgr.shape[0], y + th), min(room_bgr.shape[1], x + tw)
    if x0p < x1p and y0p < y1p:
        m = mult[(y0p - y):(y1p - y), (x0p - x):(x1p - x)]
        room_bgr[y0p:y1p, x0p:x1p] = np.clip(room_bgr[y0p:y1p, x0p:x1p].astype(np.float32) * m, 0, 255).astype(np.uint8)
    out_bgr = _rgba_to_rgb_over(room_bgr, np.array(ci), x, y)
    return _bgr_to_pil(out_bgr)

# ------------------ MAIN ROUTE ------------------
@bp.route("/", methods=["GET", "POST"], strict_slashes=False)
def render_preview():
    """
    GET /render?room_url=...&cutouts=url1,url2&anchor=center&fit=0.5&shadow=1
    POST /render (multipart/form-data)
         fields:
           - room: file
           - cutouts: file (repeatable)
           - mask: file (optional) <-- explicit inpaint mask; alpha used if present
           - perspective_src: x1,y1,x2,y2,x3,y3,x4,y4
           - perspective_dst: x1,y1,x2,y2,x3,y3,x4,y4
           - anchor, fit, shadow, opacity, offset_x, offset_y (optional)
           - mode: "manual" (default) or "auto"
    Returns image/png.
    """
    # 1) Room image
    if request.method == "POST" and "room" in request.files:
        room = _open_rgba_from_upload(request.files["room"])
    else:
        room_url = request.args.get("room_url")
        if not room_url:
            abort(400, "room_url (GET) or room file (POST) is required")
        room = _open_rgba_from_url(room_url)
    # 2) Cutouts
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
    # ---- NEW: auto mode short-circuit ----
    mode = (request.values.get("mode") or "manual").lower()
    if mode == "auto":
        out = _auto_replace(
            room,
            cut_images[0],
            fudge_scale=float(request.values.get("fudge_scale", 1.0)),
            shadow_blur=int(request.values.get("shadow_blur", 26)),
            shadow_opacity=float(request.values.get("shadow_opacity", 0.38)),
            shadow_dy=int(request.values.get("shadow_dy", 10)),
        )
        out = _constrain_max_width(out, RENDER_OUT_MAX_W)
        buf = io.BytesIO()
        out.save(buf, "PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=False, download_name="render.png")
    # 3) Params (manual mode)
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
    fit = _f("fit", 0.5)  # fraction of room shorter side
    opacity = _f("opacity", 1.0)  # 0..1
    shadow_on = (request.values.get("shadow", "1").lower() in ("1", "true", "yes"))
    shadow_opacity = _f("shadow_opacity", 0.35)
    shadow_blur = _i("shadow_blur", 20)
    shadow_dx = _i("shadow_dx", 8)
    shadow_dy = _i("shadow_dy", 8)
    offset_x = _i("offset_x", 0)  # manual nudge after anchoring
    offset_y = _i("offset_y", 0)
    anchor = (request.values.get("anchor") or "center").lower()
    allowed_anchors = {
        "center", "topleft", "topright", "bottomleft", "bottomright",
        "top", "bottom", "left", "right"
    }
    if anchor not in allowed_anchors:
        abort(400, f"invalid anchor: {anchor!r}; allowed={sorted(allowed_anchors)}")
    # 4) Base sizing from fit
    W, H = room.size
    target = int(min(W, H) * max(0.05, min(1.0, fit)))
    # 5) Anchor position (default rectangle where object will sit)
    def _anchor_xy(tw: int, th: int) -> Tuple[int, int]:
        mapping = {
            "center": ((W - tw)//2, (H - th)//2),
            "topleft": (0, 0),
            "topright": (W - tw, 0),
            "bottomleft": (0, H - th),
            "bottomright": (W - tw, H - th),
            "top": ((W - tw)//2, 0),
            "bottom": ((W - tw)//2, H - th),
            "left": (0, (H - th)//2),
            "right": (W - tw, (H - th)//2),
        }
        return mapping[anchor]
    # 6) Optional perspective points
    perspective_src = request.values.get("perspective_src")
    perspective_dst = request.values.get("perspective_dst")
    use_persp = bool(perspective_src and perspective_dst)
    if use_persp:
        try:
            src_coords = list(map(int, perspective_src.split(',')))
            dst_coords = list(map(int, perspective_dst.split(',')))
            if len(src_coords) != 8 or len(dst_coords) != 8:
                abort(400, "perspective_src/dst must be 8 ints each (x1,y1,...,x4,y4)")
            src_points = [(src_coords[i], src_coords[i+1]) for i in range(0, 8, 2)]
            dst_points = [(dst_coords[i], dst_coords[i+1]) for i in range(0, 8, 2)]
        except Exception:
            abort(400, "invalid perspective points")
    else:
        src_points = dst_points = None
    # 7) Prepare cutouts with positions
    placements: List[Tuple[Image.Image, int, int]] = []
    for i, c in enumerate(cut_images):
        # scale by fit
        ow, oh = c.size
        scale = target / float(max(ow, oh))
        tw, th = max(1, int(ow * scale)), max(1, int(oh * scale))
        ci = c.resize((tw, th), Image.LANCZOS)
        if use_persp and i == 0:
            # Warp only the first item by the provided quad
            src = [(0, 0), (0, th), (tw, th), (tw, 0)]
            warped, top_left = _perspective_transform(ci, src, dst_points)
            ci = warped
            x, y = top_left
        else:
            x, y = _anchor_xy(tw, th)
        # cascade slightly if multiple items, then allow manual nudging
        x += int(i * tw * 0.1) + offset_x
        y += int(i * th * 0.1) + offset_y
        if opacity < 1.0:
            a = ci.getchannel("A").point(lambda p: int(p * opacity))
            ci.putalpha(a)
        placements.append((ci, x, y))
    # 8) Build inpaint mask from (dilated) alpha footprints of all cutouts
    room_bgr = _pil_to_bgr(room)  # HxWx3
    mask_union = np.zeros((H, W), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    for ci, x, y in placements:
        rgba = np.array(ci)
        alpha = rgba[..., 3]
        hole = cv2.dilate(alpha, kernel, iterations=1)
        fh, fw = hole.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + fw), min(H, y + fh)
        if x0 < x1 and y0 < y1:
            mask_union[y0:y1, x0:x1] = np.maximum(
                mask_union[y0:y1, x0:x1],
                hole[(y0 - y):(y1 - y), (x0 - x):(x1 - x)]
            )
    # 8b) If user provided explicit mask, OR it in (use ALPHA if present)
    if request.method == "POST" and "mask" in request.files:
        m_img = _open_rgba_from_upload(request.files["mask"])
        m = _mask_from_image(m_img, (W, H))
        mask_union = np.maximum(mask_union, m)
    elif request.method == "GET" and request.args.get("mask_url"):
        m_img = _open_rgba_from_url(request.args["mask_url"])
        m = _mask_from_image(m_img, (W, H))
        mask_union = np.maximum(mask_union, m)
    # 9) Inpaint background under where objects go
    if mask_union.any():
        imask = (mask_union > 0).astype(np.uint8) * 255
        room_bgr = cv2.inpaint(room_bgr, imask, 3, cv2.INPAINT_TELEA)
    # 10) Optional soft shadow (multiply)
    if shadow_on and mask_union.any():
        mult = _make_shadow_multiplier(mask_union, blur_px=shadow_blur, opacity=shadow_opacity, y_offset=shadow_dy)
        room_bgr = np.clip(room_bgr.astype(np.float32) * mult, 0, 255).astype(np.uint8)
    # 11) Alpha composite all objects
    out_bgr = room_bgr
    for ci, x, y in placements:
        out_bgr = _rgba_to_rgb_over(out_bgr, np.array(ci), x, y)
    out = _bgr_to_pil(out_bgr)
    # 12) Output width cap
    out = _constrain_max_width(out, RENDER_OUT_MAX_W)
    # 13) Stream PNG
    buf = io.BytesIO()
    out.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", as_attachment=False, download_name="render.png")