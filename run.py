from app import create_app
from flask import render_template, send_from_directory
from app.services.recommender import recommender_bp
from app.services.matting import bp as matting_bp
from app.services.render import bp as render_bp

app = create_app()

@app.get("/widget.js")
def widget_js():
    return send_from_directory(".", "widget.js", mimetype="application/javascript")

if __name__ == "__main__":
    app.run(debug=True, port=5001)