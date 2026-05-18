from flask import Flask, request, jsonify
from transformers import pipeline
import json
from datetime import datetime
import re

app = Flask(__name__)
model = pipeline("fill-mask", model="bert-base-uncased")


def normalize_blanks(text):
    text = re.sub(r"\.{3,}", "[MASK]", text)
    text = re.sub(r"…+", "[MASK]", text)
    text = re.sub(r"_{2,}", "[MASK]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_all_masks(text):
    results_all = []
    mask_count = text.count("[MASK]")
    current_text = text

    for i in range(mask_count):
        temp_text = current_text.replace("[MASK]", "something")
        temp_text = temp_text.replace("something", "[MASK]", 1)

        results = model(temp_text, top_k=3)

        if results and isinstance(results[0], list):
            results = results[0]

        predictions = [
            {
                "token": r["token_str"],
                "score": float(r["score"])
            }
            for r in results
        ]

        results_all.append({
            "mask_index": i + 1,
            "predictions": predictions
        })

        best_word = results[0]["token_str"]
        current_text = current_text.replace("[MASK]", best_word, 1)

    return results_all


@app.route("/", methods=["GET"])
def home():
    return "AI Backend çalışıyor"


@app.route("/predict", methods=["POST"])
def predict():
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
    app.run(debug=True)