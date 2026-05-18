from flask import Flask, request, jsonify
import requests
import json
from datetime import datetime
import re
import os
import time

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_NAME = "distilbert/distilbert-base-uncased"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def normalize_blanks(text):
    text = re.sub(r"\.{3,}", "[MASK]", text)
    text = re.sub(r"…+", "[MASK]", text)
    text = re.sub(r"_{2,}", "[MASK]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def call_huggingface_fill_mask(text):
    payload = {
        "inputs": text,
        "parameters": {
            "top_k": 3
        }
    }

    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    raw = response.text

    try:
        data = response.json()
    except Exception:
        raise Exception(f"HF response is not JSON. Status={response.status_code}, Body={raw[:300]}")

    if isinstance(data, dict) and "estimated_time" in data:
        time.sleep(float(data["estimated_time"]) + 1)

        response = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json=payload,
            timeout=60
        )

        raw = response.text

        try:
            data = response.json()
        except Exception:
            raise Exception(f"HF retry response is not JSON. Status={response.status_code}, Body={raw[:300]}")

    if isinstance(data, dict) and "error" in data:
        raise Exception(data["error"])

    if data and isinstance(data[0], list):
        data = data[0]

    return data


def predict_all_masks(text):
    results_all = []
    mask_count = text.count("[MASK]")
    current_text = text

    for i in range(mask_count):
        temp_text = current_text.replace("[MASK]", "something")
        temp_text = temp_text.replace("something", "[MASK]", 1)

        results = call_huggingface_fill_mask(temp_text)

        predictions = [
            {
                "token": r["token_str"],
                "score": float(r["score"])
            }
            for r in results[:3]
        ]

        results_all.append({
            "mask_index": i + 1,
            "predictions": predictions
        })

        best_word = predictions[0]["token"]
        current_text = current_text.replace("[MASK]", best_word, 1)

    return results_all


@app.route("/", methods=["GET"])
def home():
    return "AI Backend çalışıyor"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        raw_text = data["text"]

        normalized_text = normalize_blanks(raw_text)
        result = predict_all_masks(normalized_text)

        return jsonify({
            "status": "success",
            "normalized_text": normalized_text,
            "mask_count": normalized_text.count("[MASK]"),
            "results": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json

    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "original_text": data.get("original_text"),
        "mask_index": data.get("mask_index"),
        "predictions": data.get("predictions"),
        "selected_word": data.get("selected_word"),
        "is_accepted": data.get("is_accepted")
    }

    with open("feedback_data.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")

    return jsonify({
        "status": "success",
        "message": "Feedback saved"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)