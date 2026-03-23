from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os
import json

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# ─────────────────────────────────────────────
#  Model Loading
#  Place your trained model file (model.pkl /
#  lung_cancer_model.pkl) in this same folder.
#  The loader tries common filenames automatically.
# ─────────────────────────────────────────────
MODEL = None
MODEL_NAMES = [
    "model.pkl",
    "lung_cancer_model.pkl",
    "lung_cancer.pkl",
    "classifier.pkl",
    "rf_model.pkl",
    "svm_model.pkl",
]

def load_model():
    global MODEL
    base = os.path.dirname(os.path.abspath(__file__))
    for name in MODEL_NAMES:
        path = os.path.join(base, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                MODEL = pickle.load(f)
            print(f"[INFO] Loaded model from: {path}")
            return True
    print("[WARN] No pre-trained model file found. Running in DEMO mode.")
    return False

load_model()


# ─────────────────────────────────────────────
#  Feature order must match your training data
# ─────────────────────────────────────────────
FEATURE_ORDER = [
    "GENDER",           # 1=Male, 0=Female
    "AGE",              # numeric
    "SMOKING",          # 1=Yes, 0=No
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
]


def demo_predict(features):
    """
    Fallback heuristic prediction when no model file is present.
    For development / demo purposes only.
    """
    risk_factors = [
        features[2],   # SMOKING
        features[3],   # YELLOW_FINGERS
        features[6],   # CHRONIC_DISEASE
        features[9],   # WHEEZING
        features[11],  # COUGHING
        features[12],  # SHORTNESS_OF_BREATH
        features[14],  # CHEST_PAIN
    ]
    age_factor = 1 if features[1] > 60 else 0
    score = sum(risk_factors) + age_factor
    probability = round(min(score / 8.0, 0.98), 2)
    prediction = 1 if probability >= 0.50 else 0
    return prediction, probability


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Build feature vector in correct order
        features = []
        for key in FEATURE_ORDER:
            val = data.get(key)
            if val is None:
                return jsonify({"error": f"Missing field: {key}"}), 400
            features.append(float(val))

        feature_array = np.array([features])

        if MODEL is not None:
            prediction = int(MODEL.predict(feature_array)[0])
            try:
                proba = MODEL.predict_proba(feature_array)[0]
                probability = round(float(proba[1]), 2)
            except AttributeError:
                probability = 1.0 if prediction == 1 else 0.0
        else:
            prediction, probability = demo_predict(features)

        result = {
            "prediction": prediction,
            "probability": probability,
            "result_label": "HIGH RISK — Lung Cancer Detected" if prediction == 1 else "LOW RISK — No Cancer Detected",
            "risk_level": "HIGH" if probability >= 0.7 else ("MODERATE" if probability >= 0.4 else "LOW"),
            "demo_mode": MODEL is None,
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
