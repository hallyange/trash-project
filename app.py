from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io, os

app = Flask(__name__)

# 안전하게 모델 경로 지정 (같은 폴더에 있을 때)
model_path = os.path.join(os.path.dirname(__file__), "trash_model_yolo_cls_best.pt")
model = YOLO(model_path)

@app.route("/")
def home():
    return "✅ YOLO Flask API is running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    results = model.predict(img, imgsz=224, device="cpu", verbose=False)[0]
    top1 = results.names[int(results.probs.top1)]
    conf = float(results.probs.top1conf)

    probs = results.probs.data.tolist()
    top3_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    top3 = [{"class": results.names[i], "conf": round(probs[i], 3)} for i in top3_idx]

    return jsonify({"top1": top1, "confidence": round(conf, 3), "top3": top3})

if __name__ == "__main__":
    # Render가 지정한 PORT 환경변수 사용
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
