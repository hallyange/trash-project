import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# YOLO 모델 로드
model = YOLO("trash_model_yolo_cls_best.pt")

@app.route("/")
def home():
    return "✅ YOLO Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    # 예측
    results = model.predict(img, imgsz=224, device="cpu", verbose=False)[0]
    top1 = results.names[int(results.probs.top1)]
    conf = float(results.probs.top1conf)

    # Top3
    probs = results.probs.data.tolist()
    top3_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    top3 = [
        {"class": results.names[i], "conf": round(probs[i], 3)} for i in top3_idx
    ]

    return jsonify({
        "top1": top1,
        "confidence": round(conf, 3),
        "top3": top3
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


