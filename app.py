import io
import base64
import json
import numpy as np
import cv2
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
LAST_CONV_LAYER = "Conv_1"

# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

with open("diseaseMaster.json", "r") as f:
    DISEASE_DB = json.load(f)

CLASS_NAMES = list(DISEASE_DB.keys())

# =========================
# Flask Setup
# =========================

app = Flask(__name__)

# Strong CORS fix for Render + localhost
CORS(app, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# =========================
# Preprocess
# =========================

def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# =========================
# Grad-CAM
# =========================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# =========================
# Health Check Route
# =========================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Backend running"}), 200

# =========================
# Prediction Route
# =========================

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():

    if request.method == "OPTIONS":
        return jsonify({"status": "OK"}), 200

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")

    input_tensor = preprocess(img)
    preds = model.predict(input_tensor)[0]

    if len(preds) != len(CLASS_NAMES):
        return jsonify({
            "error": f"Model output size ({len(preds)}) does not match CLASS_NAMES ({len(CLASS_NAMES)})"
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

    # ===== Grad-CAM =====
    heatmap = make_gradcam_heatmap(input_tensor, model, LAST_CONV_LAYER)
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(img)
    superimposed_img = heatmap * 0.4 + original
    superimposed_img = np.uint8(superimposed_img)

    _, buffer = cv2.imencode(".jpg", superimposed_img)
    gradcam_base64 = base64.b64encode(buffer).decode()

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    leaf_preview = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({
        "leaf_preview": leaf_preview,
        "gradcam": gradcam_base64,
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
# Run Server
# =========================

if __name__ == "__main__":
    app.run(debug=True, port=8000)