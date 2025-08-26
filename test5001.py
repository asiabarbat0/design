from flask import Flask
app = Flask(__name__)

@app.get("/")
def ok():
    return "it works on 5001 ✅"

app.run(host="127.0.0.1", port=5001)
