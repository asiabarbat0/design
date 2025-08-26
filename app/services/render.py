from flask import Blueprint, request, abort, send_file, current_app, jsonify
import io, requests
from PIL import Image, ImageFilter
from PIL import ImageEnhance
from werkzeug.exceptions import HTTPException
import os
from werkzeug.utils import secure_filename  # Added for file upload security
import time  # Added for time.time()

bp = Blueprint("render", __name__, url_prefix="/render")
TIMEOUT = 15
MAX_BYTES = 25 * 1024 * 1024  # 25MB

def _fetch_bytes(url: str) -> bytes:
    current_app.logger.info(f"Fetching bytes from {url}")
    with requests.get(url, timeout=TIMEOUT, stream=True) as r:
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

def _open_rgba(url: str) -> Image.Image:
    current_app.logger.info(f"Opening RGBA from {url}")
    try:
        img = Image.open(io.BytesIO(_fetch_bytes(url))).convert("RGBA")
        return img
    except Exception as e:
        current_app.logger.error(f"Failed to open {url}: {e}")
        abort(415, f"unsupported or unreachable image: {url} ({e})")

def _paste_cropped(base: Image.Image, fg: Image.Image, x: int, y: int):
    """Paste fg onto base at (x,y) with alpha, cropping if it goes out of bounds."""
    bx, by = base.size
    fw, fh = fg.size
    cx0, cy0 = 0, 0
    cx1, cy1 = fw, fh
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
    """Create a black soft shadow from the cutout's alpha."""
    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")
    mask = cut.split()[3].point(lambda p: int(p * max(0.0, min(1.0, opacity))))
    black = Image.new("RGBA", cut.size, (0, 0, 0, 255))
    shadow = Image.composite(black, Image.new("RGBA", cut.size, (0, 0, 0, 0)), mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    return shadow

def _parse_int(name: str, val: str) -> int:
    try:
        n = int(val)
    except (TypeError, ValueError):
        abort(400, f"invalid {name}: {val!r} (must be integer)")
    return n

# --- NEW: force final output size (defaults to 1920w unless overridden) ---
def _final_resize(img: Image.Image, default_width: int = 1920) -> Image.Image:
    """Resize final composite to a target width/height (preserve aspect)."""
    W, H = img.size
    out_w = request.values.get("out_width")
    out_h = request.values.get("out_height")

    if out_w:
        w = _parse_int("out_width", out_w)
        if w <= 0:
            abort(400, f"invalid out_width: {w}")
        h = max(1, int(H * (w / W)))
    elif out_h:
        h = _parse_int("out_height", out_h)
        if h <= 0:
            abort(400, f"invalid out_height: {h}")
        w = max(1, int(W * (h / H)))
    else:
        if W <= default_width:
            return img
        w = default_width
        h = max(1, int(H * (w / W)))

    return img.resize((w, h), Image.LANCZOS)

@bp.route("/render", methods=["GET", "POST"])  # NOTE: With url_prefix='/render', this is '/render/render'
def preview():
    current_app.logger.info(
        f"Render preview called with method={request.method}, cutouts={request.args.get('cutouts')}"
    )
    base_static = "/Users/asiabarbato/Downloads/designstreamaigrok/static"
    room_path = os.path.join(base_static, "unsplash-image-mw_mj-noYHM.png")
    cutout_paths = []

    if request.method == "POST":
        # Handle uploaded files
        if "room" not in request.files:
            abort(400, "No room image uploaded")
        room_file = request.files["room"]
        if room_file.filename == "":
            abort(400, "No selected room image")
        room_path = os.path.join(base_static, secure_filename(room_file.filename))
        room_file.save(room_path)

        if "cutouts" not in request.files:
            abort(400, "No cutout images uploaded")
        cutout_files = request.files.getlist("cutouts")
        if not cutout_files or all(f.filename == "" for f in cutout_files):
            abort(400, "No selected cutout images")
        cutout_paths = [os.path.join(base_static, secure_filename(f.filename)) for f in cutout_files]
        for i, path in enumerate(cutout_paths):
            cutout_files[i].save(path)

    else:  # GET method
        cutouts_param = request.args.get("cutouts")
        if cutouts_param:
            cutout_paths = [
                os.path.join(base_static, os.path.basename(n.strip()))
                for n in cutouts_param.split(",")
            ]

    cuts = []
    try:
        if not os.path.exists(room_path):
            abort(400, f"room image not found: {os.path.basename(room_path)}")
        room = Image.open(room_path).convert("RGBA")
        current_app.logger.info(f"Room image path: {room_path}, size={room.size}")
        # Save the *source room* separately for debugging
        room.save("/tmp/debug_room.png", "PNG")

        for i, path in enumerate(cutout_paths):
            if os.path.exists(path):
                cut = Image.open(path).convert("RGBA")
                current_app.logger.info(f"Cutout {i} loaded from {path}, size={cut.size}")
                cut.save(f"/tmp/debug_cutout{i}.png", "PNG")
                cuts.append(cut)
            else:
                current_app.logger.warning(f"Cutout {path} not found, skipping")
        if not cuts:
            abort(400, "No valid cutout images found")

        # Validate and process parameters
        scale_arg = request.args.get("scale", "1.0") if request.method == "GET" else request.form.get("scale", "1.0")
        try:
            scale = float(scale_arg)
        except ValueError:
            abort(400, f"non-numeric scale: {scale_arg!r}")
        if scale <= 0:
            abort(400, f"invalid scale: {scale} (must be > 0)")

        fit_value = request.args.get("fit") if request.method == "GET" else request.form.get("fit")
        if fit_value is not None:
            try:
                fit_scale = float(fit_value)
            except ValueError:
                abort(400, f"non-numeric fit: {fit_value!r}")
            if not (0 < fit_scale <= 1):
                abort(400, f"fit {fit_value} out of range (0 < fit <= 1)")
        else:
            fit_scale = 0.5

        width_arg = request.args.get("width") if request.method == "GET" else request.form.get("width")
        height_arg = request.args.get("height") if request.method == "GET" else request.form.get("height")
        if width_arg or height_arg:
            tw = _parse_int("width", width_arg) if width_arg else int(cuts[0].width * scale)
            th = _parse_int("height", height_arg) if height_arg else int(cuts[0].height * scale)
        else:
            W, H = room.size
            th = int(H * fit_scale)
            tw = int((th / cuts[0].height) * cuts[0].width)
        if tw <= 0 or th <= 0:
            abort(400, f"invalid target size: {tw}x{th}")

        for i, cut in enumerate(cuts):
            if (tw, th) != cut.size:
                cuts[i] = cut.resize((tw, th), Image.LANCZOS)
            # Size optimization
            if max(tw, th) > 1920:
                ratio = 1920 / max(tw, th)
                tw_new, th_new = int(tw * ratio), int(th * ratio)
                cuts[i] = cuts[i].resize((tw_new, th_new), Image.LANCZOS)
                current_app.logger.info(f"Resized cutout to fit 1920px max: {tw_new}x{th_new}")
            tw, th = cuts[0].size  # Update tw, th after optimization

        w, h = cuts[0].size
        opacity_arg = request.args.get("opacity", "1.0") if request.method == "GET" else request.form.get("opacity", "1.0")
        try:
            opacity = float(opacity_arg)
        except ValueError:
            abort(400, f"non-numeric opacity: {opacity_arg!r}")
        if not (0.0 <= opacity <= 1.0):
            abort(400, f"invalid opacity: {opacity} (must be between 0 and 1)")
        if opacity < 1.0:
            for i, cut in enumerate(cuts):
                a = cut.getchannel("A").point(lambda p: int(p * opacity))
                c2 = cut.copy()
                c2.putalpha(a)
                cuts[i] = c2

        anchor = (request.args.get("anchor") or "center").lower() if request.method == "GET" else request.form.get("anchor", "center").lower()
        allowed_anchors = {"center", "topleft", "topright", "bottomleft", "bottomright", "top", "bottom", "left", "right"}
        if anchor not in allowed_anchors:
            abort(400, f"invalid anchor: {anchor!r}; allowed={sorted(allowed_anchors)}")
        x_arg = request.args.get("x") if request.method == "GET" else request.form.get("x")
        y_arg = request.args.get("y") if request.method == "GET" else request.form.get("y")
        if x_arg is not None and y_arg is not None:
            x_base = _parse_int("x", x_arg)
            y_base = _parse_int("y", y_arg)
        else:
            W, H = room.size
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
            x_base, y_base = anchors[anchor]

        out = room.copy()
        if request.args.get("shadow", "1").lower() in ("1", "true", "yes"):
            s_op_arg = request.args.get("shadow_opacity", "0.4") if request.method == "GET" else request.form.get("shadow_opacity", "0.4")
            s_bl_arg = request.args.get("shadow_blur", "12") if request.method == "GET" else request.form.get("shadow_blur", "12")
            s_dx_arg = request.args.get("shadow_dx", "8") if request.method == "GET" else request.form.get("shadow_dx", "8")
            s_dy_arg = request.args.get("shadow_dy", "8") if request.method == "GET" else request.form.get("shadow_dy", "8")
            try:
                s_op = float(s_op_arg)
            except ValueError:
                abort(400, f"non-numeric shadow_opacity: {s_op_arg!r}")
            if not (0.0 <= s_op <= 1.0):
                abort(400, f"invalid shadow_opacity: {s_op} (0..1)")
            s_bl = _parse_int("shadow_blur", s_bl_arg)
            s_dx = _parse_int("shadow_dx", s_dx_arg)
            s_dy = _parse_int("shadow_dy", s_dy_arg)
            for i, cut in enumerate(cuts):
                x = x_base + (i * int(w * 0.1))
                y = y_base + (i * int(h * 0.1))
                shadow = _make_shadow(cut, opacity=s_op, blur=s_bl)
                _paste_cropped(out, shadow, x + s_dx, y + s_dy)
                _paste_cropped(out, cut, x, y)
        else:
            for i, cut in enumerate(cuts):
                x = x_base + (i * int(w * 0.1))
                y = y_base + (i * int(h * 0.1))
                _paste_cropped(out, cut, x, y)

        curtains = request.args.get("curtains", "false").lower() if request.method == "GET" else request.form.get("curtains", "false").lower()
        if curtains in ("1", "true", "yes"):
            curtain_path = os.path.join(base_static, "curtain.png")
            if os.path.exists(curtain_path):
                curtain = Image.open(curtain_path).convert("RGBA")
                current_app.logger.info(f"Curtain loaded from {curtain_path}, size={curtain.size}")
                if curtain.size != room.size:
                    curtain = curtain.resize(room.size, Image.LANCZOS)
                    current_app.logger.info(f"Curtain resized to {room.size}")
                _paste_cropped(out, curtain, 0, 0)
            else:
                current_app.logger.warning(f"Curtain file {curtain_path} not found")

        light_effect = (request.args.get("lights", "none").lower() if request.method == "GET"
                      else request.form.get("lights", "none").lower())
        if light_effect in ("soft", "bright"):
            light_path = os.path.join(base_static, f"light_{light_effect}.png")
            if os.path.exists(light_path):
                light = Image.open(light_path).convert("RGBA")
                current_app.logger.info(f"Light {light_effect} loaded from {light_path}, size={light.size}")
                if light.size != room.size:
                    light = light.resize(room.size, Image.LANCZOS)
                    current_app.logger.info(f"Light resized to {room.size}")
                if light_effect == "bright":
                    enhancer = ImageEnhance.Brightness(light)
                    light = enhancer.enhance(4.5)
                    current_app.logger.info("Applied bright enhancement with factor 4.5")
                _paste_cropped(out, light, 0, 0)
            else:
                current_app.logger.warning(f"Light file {light_path} not found")

        # --- NEW: final resize to 1920-wide unless overridden with out_width/out_height ---
        out = _final_resize(out, default_width=1920)

        # stream PNG
        buf = io.BytesIO()
        out.save(buf, "PNG")
        png_bytes = buf.getvalue()
        with open("/tmp/debug_render.png", "wb") as f:
            f.write(png_bytes)
        return send_file(
            io.BytesIO(png_bytes),
            mimetype="image/png",
            as_attachment=False,
            download_name="render.png",
        )

    except HTTPException:
        raise
    except Exception as e:
        current_app.logger.exception("render failed")
        abort(500, f"render failed: {e}")

@bp.route("/api/render", methods=["GET", "POST"])
def api_render():
    print("API render route hit")  # Debug print
    current_app.logger.info(f"API render called with method={request.method}, cutouts={request.args.get('cutouts')}")
    base_static = "/Users/asiabarbato/Downloads/designstreamaigrok/static"
    room_path = os.path.join(base_static, "unsplash-image-mw_mj-noYHM.png")
    cutout_paths = []

    if request.method == "POST":
        if "room" not in request.files:
            return jsonify({"error": "No room image uploaded", "code": 400}), 400
        room_file = request.files["room"]
        if room_file.filename == "":
            return jsonify({"error": "No selected room image", "code": 400}), 400
        room_path = os.path.join(base_static, secure_filename(room_file.filename))
        room_file.save(room_path)

        if "cutouts" not in request.files:
            return jsonify({"error": "No cutout images uploaded", "code": 400}), 400
        cutout_files = request.files.getlist("cutouts")
        if not cutout_files or all(f.filename == "" for f in cutout_files):
            return jsonify({"error": "No selected cutout images", "code": 400}), 400
        cutout_paths = [os.path.join(base_static, secure_filename(f.filename)) for f in cutout_files]
        for i, path in enumerate(cutout_paths):
            cutout_files[i].save(path)

    else:  # GET method
        cutouts_param = request.args.get("cutouts")
        if cutouts_param:
            cutout_paths = [
                os.path.join(base_static, os.path.basename(n.strip()))
                for n in cutouts_param.split(",")
            ]

    cuts = []
    try:
        if not os.path.exists(room_path):
            return jsonify({"error": f"Room image not found: {os.path.basename(room_path)}", "code": 400}), 400
        room = Image.open(room_path).convert("RGBA")
        current_app.logger.info(f"Room image path: {room_path}, size={room.size}")
        room.save("/tmp/debug_room.png", "PNG")  # Save the source room

        for i, path in enumerate(cutout_paths):
            if os.path.exists(path):
                cut = Image.open(path).convert("RGBA")
                current_app.logger.info(f"Cutout {i} loaded from {path}, size={cut.size}")
                cut.save(f"/tmp/debug_cutout{i}.png", "PNG")
                cuts.append(cut)
            else:
                current_app.logger.warning(f"Cutout {path} not found, skipping")
        if not cuts:
            return jsonify({"error": "No valid cutout images found", "code": 400}), 400

        # Validate and process parameters
        scale_arg = request.args.get("scale", "1.0") if request.method == "GET" else request.form.get("scale", "1.0")
        try:
            scale = float(scale_arg)
        except ValueError:
            return jsonify({"error": f"non-numeric scale: {scale_arg!r}", "code": 400}), 400
        if scale <= 0:
            return jsonify({"error": f"invalid scale: {scale} (must be > 0)", "code": 400}), 400

        fit_value = request.args.get("fit") if request.method == "GET" else request.form.get("fit")
        if fit_value is not None:
            try:
                fit_scale = float(fit_value)
            except ValueError:
                return jsonify({"error": f"non-numeric fit: {fit_value!r}", "code": 400}), 400
            if not (0 < fit_scale <= 1):
                return jsonify({"error": f"fit {fit_value} out of range (0 < fit <= 1)", "code": 400}), 400
        else:
            fit_scale = 0.5

        width_arg = request.args.get("width") if request.method == "GET" else request.form.get("width")
        height_arg = request.args.get("height") if request.method == "GET" else request.form.get("height")
        if width_arg or height_arg:
            tw = _parse_int("width", width_arg) if width_arg else int(cuts[0].width * scale)
            th = _parse_int("height", height_arg) if height_arg else int(cuts[0].height * scale)
        else:
            W, H = room.size
            th = int(H * fit_scale)
            tw = int((th / cuts[0].height) * cuts[0].width)
        if tw <= 0 or th <= 0:
            return jsonify({"error": f"invalid target size: {tw}x{th}", "code": 400}), 400

        for i, cut in enumerate(cuts):
            if (tw, th) != cut.size:
                cuts[i] = cut.resize((tw, th), Image.LANCZOS)
            if max(tw, th) > 1920:
                ratio = 1920 / max(tw, th)
                tw_new, th_new = int(tw * ratio), int(th * ratio)
                cuts[i] = cuts[i].resize((tw_new, th_new), Image.LANCZOS)
                current_app.logger.info(f"Resized cutout to fit 1920px max: {tw_new}x{th_new}")
            tw, th = cuts[0].size

        w, h = cuts[0].size
        opacity_arg = request.args.get("opacity", "1.0") if request.method == "GET" else request.form.get("opacity", "1.0")
        try:
            opacity = float(opacity_arg)
        except ValueError:
            return jsonify({"error": f"non-numeric opacity: {opacity_arg!r}", "code": 400}), 400
        if not (0.0 <= opacity <= 1.0):
            return jsonify({"error": f"invalid opacity: {opacity} (must be between 0 and 1)", "code": 400}), 400
        if opacity < 1.0:
            for i, cut in enumerate(cuts):
                a = cut.getchannel("A").point(lambda p: int(p * opacity))
                c2 = cut.copy()
                c2.putalpha(a)
                cuts[i] = c2

        anchor = (request.args.get("anchor") or "center").lower() if request.method == "GET" else request.form.get("anchor", "center").lower()
        allowed_anchors = {"center", "topleft", "topright", "bottomleft", "bottomright", "top", "bottom", "left", "right"}
        if anchor not in allowed_anchors:
            return jsonify({"error": f"invalid anchor: {anchor!r}; allowed={sorted(allowed_anchors)}", "code": 400}), 400
        x_arg = request.args.get("x") if request.method == "GET" else request.form.get("x")
        y_arg = request.args.get("y") if request.method == "GET" else request.form.get("y")
        if x_arg is not None and y_arg is not None:
            x_base = _parse_int("x", x_arg)
            y_base = _parse_int("y", y_arg)
        else:
            W, H = room.size
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
            x_base, y_base = anchors[anchor]

        out = room.copy()
        if request.args.get("shadow", "1").lower() in ("1", "true", "yes"):
            s_op_arg = request.args.get("shadow_opacity", "0.4") if request.method == "GET" else request.form.get("shadow_opacity", "0.4")
            s_bl_arg = request.args.get("shadow_blur", "12") if request.method == "GET" else request.form.get("shadow_blur", "12")
            s_dx_arg = request.args.get("shadow_dx", "8") if request.method == "GET" else request.form.get("shadow_dx", "8")
            s_dy_arg = request.args.get("shadow_dy", "8") if request.method == "GET" else request.form.get("shadow_dy", "8")
            try:
                s_op = float(s_op_arg)
            except ValueError:
                return jsonify({"error": f"non-numeric shadow_opacity: {s_op_arg!r}", "code": 400}), 400
            if not (0.0 <= s_op <= 1.0):
                return jsonify({"error": f"invalid shadow_opacity: {s_op} (0..1)", "code": 400}), 400
            s_bl = _parse_int("shadow_blur", s_bl_arg)
            s_dx = _parse_int("shadow_dx", s_dx_arg)
            s_dy = _parse_int("shadow_dy", s_dy_arg)
            for i, cut in enumerate(cuts):
                x = x_base + (i * int(w * 0.1))
                y = y_base + (i * int(h * 0.1))
                shadow = _make_shadow(cut, opacity=s_op, blur=s_bl)
                _paste_cropped(out, shadow, x + s_dx, y + s_dy)
                _paste_cropped(out, cut, x, y)
        else:
            for i, cut in enumerate(cuts):
                x = x_base + (i * int(w * 0.1))
                y = y_base + (i * int(h * 0.1))
                _paste_cropped(out, cut, x, y)

        curtains = request.args.get("curtains", "false").lower() if request.method == "GET" else request.form.get("curtains", "false").lower()
        if curtains in ("1", "true", "yes"):
            curtain_path = os.path.join(base_static, "curtain.png")
            if os.path.exists(curtain_path):
                curtain = Image.open(curtain_path).convert("RGBA")
                current_app.logger.info(f"Curtain loaded from {curtain_path}, size={curtain.size}")
                if curtain.size != room.size:
                    curtain = curtain.resize(room.size, Image.LANCZOS)
                    current_app.logger.info(f"Curtain resized to {room.size}")
                _paste_cropped(out, curtain, 0, 0)
            else:
                current_app.logger.warning(f"Curtain file {curtain_path} not found")

        light_effect = (request.args.get("lights", "none").lower() if request.method == "GET"
                      else request.form.get("lights", "none").lower())
        if light_effect in ("soft", "bright"):
            light_path = os.path.join(base_static, f"light_{light_effect}.png")
            if os.path.exists(light_path):
                light = Image.open(light_path).convert("RGBA")
                current_app.logger.info(f"Light {light_effect} loaded from {light_path}, size={light.size}")
                if light.size != room.size:
                    light = light.resize(room.size, Image.LANCZOS)
                    current_app.logger.info(f"Light resized to {room.size}")
                if light_effect == "bright":
                    enhancer = ImageEnhance.Brightness(light)
                    light = enhancer.enhance(4.5)
                    current_app.logger.info("Applied bright enhancement with factor 4.5")
                _paste_cropped(out, light, 0, 0)
            else:
                current_app.logger.warning(f"Light file {light_path} not found")

        # --- NEW: final resize to 1920-wide unless overridden ---
        out = _final_resize(out, default_width=1920)

        # stream PNG and return JSON
        buf = io.BytesIO()
        out.save(buf, "PNG")
        png_bytes = buf.getvalue()
        render_path = "/tmp/debug_render.png"
        with open(render_path, "wb") as f:
            f.write(png_bytes)
        render_url = f"/static/debug_render.png?{int(time.time())}"
        return jsonify({"url": render_url, "status": "success"}), 200

    except HTTPException:
        raise
    except Exception as e:
        current_app.logger.exception("render failed")
        return jsonify({"error": f"render failed: {e}", "code": 500}), 500

@bp.route("/test")
def test_route():
    return "Blueprint is working"
