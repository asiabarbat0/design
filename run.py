import os
from app import create_app
from flask import render_template, send_from_directory
from app.services.recommender import recommender_bp
from app.services.matting import bp as matting_bp
from app.services.render import bp as render_bp
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
app = create_app()
app.static_folder = str(base_dir / "app/static")  # Explicitly set to match project structure

@app.get("/widget.js")
def widget_js():
    print(f"Attempting to serve widget.js from {app.static_folder}/widget.js")
    return send_from_directory(app.static_folder, "widget.js", mimetype="application/javascript")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)