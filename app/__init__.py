from flask import Flask, jsonify, current_app, render_template
from pathlib import Path
from werkzeug.exceptions import HTTPException
import traceback

def create_app():
    base_dir = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        static_folder=str(base_dir / "static"),
        static_url_path="/static",
    )
    # app = Flask(__name__, static_folder="static", static_url_path="/static")  # Keep commented out
    
    # Move imports inside create_app() to avoid duplicate registrations
    from app.services.matting import bp as matting_bp
    from app.services.render import bp as render_bp
    # from app.services.replace import bp as replace_bp  # Comment out if replace.py is missing
    
    print(f"Registering matting_bp: {matting_bp}")
    app.register_blueprint(matting_bp)
    print(f"Registering render_bp: {render_bp}")
    app.register_blueprint(render_bp)
    # app.register_blueprint(replace_bp)  # Comment out if replace.py is missing
    
    @app.route('/')
    def index():
        print("Hit index route")
        try:
            result = render_template('index.html')
            print(f"Rendering result: {result[:50]}...")
            return result
        except Exception as e:
            print(f"Template error: {e}")
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