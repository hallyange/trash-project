import os, io, json
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

# ===== 설정 =====
MODEL_PATH = os.environ.get("MODEL_PATH", "trash_model_yolo_cls_best.pt")
IMG_SIZE   = int(os.environ.get("IMG_SIZE", "224"))

app = Flask(__name__)
_MODEL = None  # lazy

def _model_file():
    p = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Model not found: {p}")
    # Git LFS 미다운로드 등으로 1KB짜리 포인터 파일만 있을 때 방지
    if os.path.getsize(p) < 500_000:
        raise ValueError(f"Model file too small ({os.path.getsize(p)} bytes). "
                         "Check Git LFS or upload the real .pt.")
    return p

def get_model():
    global _MODEL
    if _MODEL is None:
        # 여기서만 ultralytics/torch import → 서버는 즉시 뜸
        from ultralytics import YOLO
        model_file = _model_file()
        _MODEL = YOLO(model_file)  # CPU 추론
    return _MODEL

@app.route("/")
def index():
    # 초간단 업로드 UI
    return """
<!doctype html><meta charset="utf-8"/>
<title>Trash Classifier (.pt)</title>
<style>body{font-family:system-ui;margin:40px auto;max-width:720px}
#p{width:320px;height:320px;object-fit:cover;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.1)}
button{padding:10px 16px;border:0;border-radius:10px;background:#2563eb;color:#fff;cursor:pointer}
pre{background:#0b1020;color:#ccf;padding:12px;border-radius:10px;overflow:auto}</style>
<h1>🧠 Trash Classifier (.pt)</h1>
<input id=f type=file accept="image/*"><button id=b>예측</button>
<div style="margin-top:12px"><img id=p></div>
<h3>결과</h3><pre id=o>대기 중...</pre>
<script>
const f=document.getElementById('f'), o=document.getElementById('o'), p=document.getElementById('p');
f.onchange=()=>{ if(f.files[0]) p.src=URL.createObjectURL(f.files[0]); };
document.getElementById('b').onclick=async()=>{
  if(!f.files[0]) return alert('이미지 선택');
  o.textContent='예측 중...';
  const fd=new FormData(); fd.append('file', f.files[0]);
  const r=await fetch('/predict',{method:'POST',body:fd});
  o.textContent=JSON.stringify(await r.json(), null, 2);
};
</script>
"""

@app.route("/health")
def health():
    # 서버 생존 확인(모델 로드 안함)
    ok, msg = True, "ok"
    try:
        _ = _model_file()
    except Exception as e:
        ok, msg = False, str(e)
    return jsonify({"ok": ok, "msg": msg})

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

    # lazy-load
    try:
        model = get_model()
    except Exception as e:
        return jsonify({"error": f"Model load failed: {e}"}), 500

    res = model.predict(img, imgsz=IMG_SIZE, device="cpu", verbose=False)[0]

    # Top-1/Top-3
    top1_idx = int(res.probs.top1)
    top1_name = res.names[top1_idx]
    top1_conf = float(res.probs.top1conf)
    probs = res.probs.data.cpu().numpy()
    topk = probs.argsort()[-3:][::-1]
    top3 = [{"class": res.names[i], "conf": float(probs[i])} for i in topk]

    return jsonify({"top1": top1_name, "confidence": round(top1_conf, 3), "top3": top3})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)