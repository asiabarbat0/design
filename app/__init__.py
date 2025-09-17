from flask import Flask, jsonify, current_app, render_template, send_file
from pathlib import Path
from werkzeug.exceptions import HTTPException
from app.services.render import bp as render_bp
import traceback

def create_app(static_folder=None):
    app_dir = Path(__file__).parent
    base_dir = app_dir.parent

    app = Flask(
        __name__,
        static_folder=static_folder if static_folder else str(app_dir / "static"),
        static_url_path="/static",
        template_folder=str(app_dir / "templates"),
    )

    # ...existing blueprint registration and routes...

    # --- Blueprints ---
    try:
        from app.services.render import bp as render_bp
        app.register_blueprint(render_bp)
        print("[render] enabled")
    except Exception as e:
        print(f"[render] disabled: {e}")

    try:
        from app.services.recommender import recommender_bp
        app.register_blueprint(recommender_bp)
        print("[recommender] enabled")
    except Exception as e:
        print(f"[recommender] disabled: {e}")

    try:
        from app.services.matting import bp as matting_bp
        app.register_blueprint(matting_bp)
        print("[matting] enabled")
    except Exception as e:
        print(f"[matting] disabled: {e}")

    try:
        from app.services.auto_matting import register_blueprint as register_auto_matting
        register_auto_matting(app)
        print("[auto_matting] enabled")
    except Exception as e:
        print(f"[auto_matting] disabled: {e}")

    try:
        from app.services.matting_studio import register_blueprint as register_matting_studio
        register_matting_studio(app)
        print("[matting_studio] enabled")
    except Exception as e:
        print(f"[matting_studio] disabled: {e}")

    try:
        from app.services.matting_studio_admin import register_blueprint as register_matting_studio_admin
        register_matting_studio_admin(app)
        print("[matting_studio_admin] enabled")
    except Exception as e:
        print(f"[matting_studio_admin] disabled: {e}")

    try:
        from app.services.renderer import register_blueprint as register_renderer
        register_renderer(app)
        print("[renderer] enabled")
    except Exception as e:
        print(f"[renderer] disabled: {e}")

    # --- Serve widget.js explicitly from absolute path ---
    @app.get("/widget.js")
    def serve_widget():
        js_path = app_dir / "static" / "widget.js"
        if not js_path.exists():
            return jsonify({"error": f"widget.js not found at {js_path}"}), 404
        return send_file(str(js_path), mimetype="application/javascript", max_age=3600, conditional=True)

    @app.route("/")
    def index():
        try:
            return render_template("index.html")
        except Exception as e:
            current_app.logger.error(f"Template error: {e}")
            return "Error loading template", 500

    @app.route("/images")
    def image_viewer():
        try:
            return render_template("image_viewer.html")
        except Exception as e:
            current_app.logger.error(f"Image viewer error: {e}")
            return "Error loading image viewer", 500

    @app.route("/sofa-replacement")
    def sofa_replacement_viewer():
        try:
            return render_template("sofa_replacement_viewer.html")
        except Exception as e:
            current_app.logger.error(f"Sofa replacement viewer error: {e}")
            return "Error loading sofa replacement viewer", 500

    @app.route("/furniture-pipeline")
    def furniture_pipeline_viewer():
        try:
            return render_template("furniture_pipeline_viewer.html")
        except Exception as e:
            current_app.logger.error(f"Furniture pipeline viewer error: {e}")
            return "Error loading furniture pipeline viewer", 500

    @app.route("/accurate-sofa-replacement")
    def accurate_sofa_replacement_viewer():
        try:
            return render_template("accurate_sofa_replacement_viewer.html")
        except Exception as e:
            current_app.logger.error(f"Accurate sofa replacement viewer error: {e}")
            return "Error loading accurate sofa replacement viewer", 500

    @app.route("/natural-sofa-replacement")
    def natural_sofa_replacement_viewer():
        try:
            return render_template("natural_sofa_replacement_viewer.html")
        except Exception as e:
            current_app.logger.error(f"Natural sofa replacement viewer error: {e}")
            return "Error loading natural sofa replacement viewer", 500

    @app.route("/new-room-replacement")
    def new_room_replacement_viewer():
        try:
            return render_template("new_room_replacement_viewer.html")
        except Exception as e:
            current_app.logger.error(f"New room replacement viewer error: {e}")
            return "Error loading new room replacement viewer", 500

    @app.route("/actual-new-room")
    def actual_new_room_viewer():
        try:
            return render_template("actual_new_room_viewer.html")
        except Exception as e:
            current_app.logger.error(f"Actual new room viewer error: {e}")
            return "Error loading actual new room viewer", 500

    @app.route("/precise-sofa-replacement")
    def precise_sofa_replacement_viewer():
        try:
            return render_template("precise_sofa_replacement_viewer.html")
        except Exception as e:
            current_app.logger.error(f"Precise sofa replacement viewer error: {e}")
            return "Error loading precise sofa replacement viewer", 500

    @app.route("/manual-sofa-replacement")
    def manual_sofa_replacement_viewer():
        try:
            return render_template("manual_sofa_replacement_viewer.html")
        except Exception as e:
            current_app.logger.error(f"Manual sofa replacement viewer error: {e}")
            return "Error loading manual sofa replacement viewer", 500

    @app.route("/ai-furniture-replacement")
    def ai_furniture_replacement_viewer():
        try:
            return render_template("ai_furniture_replacement_viewer.html")
        except Exception as e:
            current_app.logger.error(f"AI furniture replacement viewer error: {e}")
            return "Error loading AI furniture replacement viewer", 500

    # --- Static Image Routes ---
    @app.route("/static/<filename>")
    def serve_static(filename):
        """Serve static files from the static directory"""
        try:
            static_path = app_dir / "static" / filename
            if static_path.exists():
                return send_file(str(static_path))
            else:
                return jsonify({"error": f"File not found: {filename}"}), 404
        except Exception as e:
            current_app.logger.error(f"Static file error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.errorhandler(400)
    @app.errorhandler(413)
    @app.errorhandler(415)
    @app.errorhandler(422)
    @app.errorhandler(502)
    def _json_known(err):
        return jsonify(error=getattr(err, "description", str(err)), code=err.code), err.code

    @app.errorhandler(Exception)
    def _catch_all(e):
        if isinstance(e, HTTPException):
            orig = getattr(e, "original_exception", None)
            if orig is not None:
                current_app.logger.error("500 original: %r\n%s", orig, traceback.format_exc())
                return jsonify(error=f"{type(orig).__name__}: {orig}", code=500), 500
            return jsonify(error=e.description, code=e.code), e.code
        current_app.logger.error("UNHANDLED: %r\n%s", e, traceback.format_exc())
        return jsonify(error=f"{type(e).__name__}: {e}", code=500), 500

    return app
