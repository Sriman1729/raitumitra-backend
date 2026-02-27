import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# =========================
# Configuration
# =========================

MODEL_PATH = "plant_disease_full_model.keras"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6

# =========================
# Load Model (once)
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

with open("diseaseMaster.json", "r") as f:
    DISEASE_DB = json.load(f)

CLASS_NAMES = list(DISEASE_DB.keys())

# =========================
# Flask Setup
# =========================

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Backend running"}), 200

# =========================
# Preprocess
# =========================

def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# =========================
# Prediction Route
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")

    input_tensor = preprocess(img)
    preds = model.predict(input_tensor)[0]

    if len(preds) != len(CLASS_NAMES):
        return jsonify({
            "error": f"Model output size mismatch"
        }), 500

    top3_idx = preds.argsort()[-3:][::-1]

    top3 = [
        {
            "label": CLASS_NAMES[i],
            "confidence": float(preds[i])
        }
        for i in top3_idx
    ]

    best_idx = top3_idx[0]
    best_label = CLASS_NAMES[best_idx]
    confidence = float(preds[best_idx])

    disease_info = DISEASE_DB.get(best_label, {})
    suggested_actions = disease_info.get("suggested_actions", [])
    severity_from_json = disease_info.get("severity", "Unknown")

    crop, disease = best_label.split("___")
    friendly = disease.replace("_", " ")

    is_unknown = confidence < CONFIDENCE_THRESHOLD
    severity = severity_from_json if not is_unknown else "Unknown"

    return jsonify({
        "best": {
            "label": best_label,
            "friendly": friendly,
            "crop": crop,
            "confidence": confidence,
            "suggested_actions": suggested_actions
        },
        "top3": top3,
        "severity": severity,
        "is_unknown": is_unknown
    })

# =========================
# Run
# =========================

if __name__ == "__main__":
    app.run(debug=True, port=8000)
