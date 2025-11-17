import os
import io
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ----- Lazy-load: 첫 요청 때만 모델 로드 (콜드스타트 단축) -----
MODEL = None
MODEL_PATH = os.environ.get("MODEL_PATH", "trash_model_yolo_cls_best.pt")  # 필요시 ENV로 덮어쓰기

def get_model():
    global MODEL
    if MODEL is None:
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        MODEL = YOLO(model_path)  # CPU 추론
    return MODEL

@app.route("/")
def home():
    return "✅ YOLO Flask API is running on Render!"

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    model = get_model()
    res = model.predict(img, imgsz=224, device="cpu", verbose=False)[0]

    # Top-1
    top1_idx = int(res.probs.top1)
    top1_name = res.names[top1_idx]
    top1_conf = float(res.probs.top1conf)

    # Top-3 (클래스 수가 3 미만이면 자동 제한)
    probs = res.probs.data.tolist()
    top3_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    top3 = [{"class": res.names[i], "conf": round(probs[i], 3)} for i in top3_idx]

    return jsonify({"top1": top1_name, "confidence": round(top1_conf, 3), "top3": top3})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))     # Render가 주는 포트 사용
    app.run(host="0.0.0.0", port=port)
