from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import tensorflow as tf
import os
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://newsentihybridge-md7s.vercel.app", "https://cnn-senti.fly.dev"])

# === Load artifacts ===
cnn_model = tf.keras.models.load_model("backend/model/cnn_model.keras")
tokenizer = joblib.load("backend/preprocess/tokenizer.pkl")

# Optional: Load RF model and TF-IDF
rf_path = "backend/model/rf_model.pkl"
tfidf_path = "backend/preprocess/tfidf_vectorizer.pkl"
rf_available = os.path.exists(rf_path)
tfidf_available = os.path.exists(tfidf_path)

rf_model = joblib.load(rf_path) if rf_available else None
tfidf_vectorizer = joblib.load(tfidf_path) if tfidf_available else None

# === Text Cleaner ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|[^a-z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# === Label Mapper ===
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# === CNN Predict ===
def predict_cnn(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=30, padding='post')
    probs = cnn_model.predict(pad, verbose=0)[0]
    class_idx = int(np.argmax(probs))
    return {
        "label": label_map[class_idx],
        "index": class_idx,
        "confidence": round(float(probs[class_idx]), 4)
    }

# === RF Predict ===
def predict_rf(text):
    text = clean_text(text)
    tfidf = tfidf_vectorizer.transform([text])
    probs = rf_model.predict_proba(tfidf)[0]
    class_idx = int(np.argmax(probs))
    return {
        "label": label_map[class_idx],
        "index": class_idx,
        "confidence": round(float(probs[class_idx]), 4)
    }

@app.route("/")
def home():
    return jsonify({
        "status": "up", 
        "message": "Hybrid Sentiment Classifier running ✅",
        "models_available": {
            "cnn": True,
            "rf": rf_available and tfidf_available
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        cnn_result = predict_cnn(text)

        if rf_available and tfidf_available:
            rf_result = predict_rf(text)
        else:
            rf_result = None

        return jsonify({
            "cnn": cnn_result,
            "rf": rf_result,
            "models_available": {
                "cnn": True,
                "rf": bool(rf_result)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
