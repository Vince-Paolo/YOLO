from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder for uploads
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model
model = YOLO("yolov8n.pt")

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files['image']

    if file.filename == '':
        return render_template("index.html", error="No selected file")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save uploaded image
        file.save(filepath)

        # Run YOLO detection
        results = model(filepath)

        # Save output (with boxes)
        results[0].save(filename=filepath)

        return render_template("index.html", image=filepath)

    return render_template("index.html", error="Invalid file type")

if __name__ == "__main__":
    app.run(debug=True)