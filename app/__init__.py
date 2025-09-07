from flask import Flask, jsonify, current_app, render_template, send_file
from pathlib import Path
from werkzeug.exceptions import HTTPException
import traceback

def create_app():
    app_dir = Path(__file__).resolve().parent      # .../app
    base_dir = app_dir.parent                      # repo root

    app = Flask(
        __name__,
        static_folder=str(app_dir / "static"),     # serve app/static/*
        static_url_path="/static",
        template_folder=str(base_dir / "templates"),
    )

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
