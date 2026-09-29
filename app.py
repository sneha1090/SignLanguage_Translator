import os
import cv2
import json
import numpy as np

from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

import warnings
warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================

IMG_SIZE = 224
MODEL_PATH = "modelnet_model.h5"
LABELS_PATH = "labels.json"

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "mp4", "avi", "mov", "mkv",
    "jpg", "jpeg", "png"
}


# =========================================================
# LOAD LABELS
# =========================================================

with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "modelnet_model.h5 not found. "
        "Please run: python train_model.py"
    )

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(
        "labels.json not found. "
        "Please run: python train_model.py"
    )


model = load_model(MODEL_PATH)

print("===================================")
print("✅ Trained model loaded")
print("✅ Number of classes:", len(LABELS))
print("✅ Labels:", LABELS)
print("===================================")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def preprocess_frame(frame):

    # Resize image to the same size used during training
    img = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    # Convert pixel values from 0-255 to 0-1
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def predict_frame(frame):

    if frame is None:
        return "Invalid Image", 0.0

    processed = preprocess_frame(frame)

    predictions = model.predict(
        processed,
        verbose=0
    )

    class_index = np.argmax(predictions)

    confidence = float(
        np.max(predictions)
    )

    label = LABELS[class_index]

    return label, confidence


# =========================================================
# VIDEO PREDICTION
# =========================================================

def extract_frames_and_predict(
    video_path,
    step=5
):

    cap = cv2.VideoCapture(video_path)

    sequence = []
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % step == 0:

            label, confidence = predict_frame(frame)

            if label != "Invalid Image":

                # Only keep reasonably confident predictions
                if confidence >= 0.50:
                    sequence.append(label)

        frame_count += 1

    cap.release()


    # Remove consecutive duplicates

    collapsed = []

    for char in sequence:

        if not collapsed or char != collapsed[-1]:

            collapsed.append(char)


    return " ".join(collapsed)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# IMAGE PREDICTION
# =========================================================

@app.route(
    "/predict_image",
    methods=["POST"]
)
def predict_image():

    if "file" not in request.files:

        return jsonify({
            "error": "No file uploaded"
        }), 400


    file = request.files["file"]


    if file.filename == "":

        return jsonify({
            "error": "No file selected"
        }), 400


    # Read uploaded image

    npimg = np.frombuffer(
        file.read(),
        np.uint8
    )


    img = cv2.imdecode(
        npimg,
        cv2.IMREAD_COLOR
    )


    if img is None:

        return jsonify({
            "error": "Could not read image"
        }), 400


    label, confidence = predict_frame(img)


    return jsonify({

        "prediction": label,

        "confidence": confidence

    })


# =========================================================
# VIDEO PREDICTION
# =========================================================

@app.route(
    "/predict_video",
    methods=["POST"]
)
def predict_video():

    if "file" not in request.files:

        return jsonify({
            "error": "No file uploaded"
        }), 400


    file = request.files["file"]


    if file.filename == "":

        return jsonify({
            "error": "No file selected"
        }), 400


    filename = secure_filename(
        file.filename
    )


    video_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(video_path)


    sequence = extract_frames_and_predict(
        video_path,
        step=5
    )


    return jsonify({

        "prediction": sequence

    })


# =========================================================
# LIVE WEBCAM
# =========================================================

def generate_frames():

    cap = cv2.VideoCapture(0)


    while True:

        success, frame = cap.read()


        if not success:

            break


        label, confidence = predict_frame(
            frame
        )


        display_text = (
            f"{label} ({confidence:.2f})"
        )


        cv2.putText(

            frame,

            display_text,

            (10, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2

        )


        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )


        frame_bytes = buffer.tobytes()


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"

        )


    cap.release()


@app.route("/predict_live")
def predict_live():

    return Response(

        generate_frames(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )

    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )