from flask import Blueprint, request, abort, send_file
import io, requests, numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

bp = Blueprint("replace", __name__, url_prefix="/replace")

# config
IOPAINT_URL = "http://127.0.0.1:8080/api/v1/inpaint"
MODEL = YOLO("yolov8x-seg.pt")   # downloads on first run
FURNITURE = {"chair","sofa","bed","dining table","tv","potted plant","bench","couch","table","person"}

def _get_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=20); r.raise_for_status(); return r.content

def _png_bytes(arr: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", arr)
    if not ok: raise RuntimeError("png encode failed")
    return enc.tobytes()

@bp.post("/")
def replace_item():
    """
    JSON: { image_url, instance_id, new_item_url }
    - Detect target instance (instance_id from /detect)
    - Inpaint the old item
    - Fit & insert new PNG into the original bbox
    """
    data = request.get_json(force=True)
    image_url = data.get("image_url")
    instance_id = data.get("instance_id")
    new_item_url = data.get("new_item_url")
    if not image_url or instance_id is None or not new_item_url:
        abort(400, "image_url, instance_id, new_item_url required")

    base_rgba = Image.open(io.BytesIO(_get_bytes(image_url))).convert("RGBA")
    W, H = base_rgba.size

    # --- 1) detect + get mask & bbox for the chosen instance ---
    res = MODEL.predict(source=np.array(base_rgba.convert("RGB")), verbose=False)[0]
    if res.masks is None or instance_id >= len(res.boxes):
        abort(404, "instance_id not found")

    # filter to furniture-ish classes (optional)
    label = res.names[int(res.boxes.cls[instance_id].item())]
    if label not in FURNITURE:
        abort(422, f"instance is '{label}', not a furniture class")

    # mask -> uint8 0/255, bbox ints
    m = (res.masks.data[instance_id].cpu().numpy() > 0.5).astype(np.uint8) * 255   # [H,W]
    x1,y1,x2,y2 = map(int, res.boxes.xyxy[instance_id].cpu().numpy().tolist())
    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(W,x2),min(H,y2)

    # --- 2) inpaint via IOPaint API (white=edit area) ---
    img_png = _png_bytes(cv2.cvtColor(np.array(base_rgba.convert("RGB")), cv2.COLOR_RGB2BGR))
    mask_png = _png_bytes(m)
    files = {
        "image": ("image.png", img_png, "image/png"),
        "mask":  ("mask.png",  mask_png, "image/png"),
    }
    r = requests.post(IOPAINT_URL, files=files, timeout=120)
    if r.status_code != 200:
        abort(502, f"inpaint error: {r.text[:200]}")
    bg_inpaint_rgb = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)  # BGR
    bg_inpaint = Image.fromarray(cv2.cvtColor(bg_inpaint_rgb, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # --- 3) place the new item PNG inside the original bbox ---
    new_png = Image.open(io.BytesIO(_get_bytes(new_item_url))).convert("RGBA")
    bw, bh = x2 - x1, y2 - y1
    nw, nh = new_png.size
    scale = min(bw / max(nw,1), bh / max(nh,1))
    new_size = (max(1, int(nw * scale)), max(1, int(nh * scale)))
    new_png = new_png.resize(new_size, Image.LANCZOS)
    ox = x1 + (bw - new_size[0]) // 2
    oy = y1 + (bh - new_size[1]) // 2

    out = bg_inpaint.copy()
    out.alpha_composite(new_png, dest=(ox, oy))

    buf = io.BytesIO()
    out.save(buf, "PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png", download_name="replaced.png")
