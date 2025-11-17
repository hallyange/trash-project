import os, io, base64
import onnxruntime as ort
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# -------- 설정 --------
MODEL_PATH = os.environ.get("MODEL_PATH", "trash_model_yolo_cls_best.onnx")
CLASS_NAMES = os.environ.get("CLASS_NAMES", "can,paper,plastic").split(",")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))

# -------- ONNX 세션 Lazy-Load --------
SESS = None
def get_session():
    global SESS
    if SESS is None:
        model_file = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model not found: {model_file}")
        SESS = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
    return SESS

def preprocess(pil_img, size=224):
    img = pil_img.convert("RGB").resize((size, size))
    x = np.asarray(img).astype(np.float32) / 255.0           # [H,W,C]
    x = np.transpose(x, (2, 0, 1))                           # [C,H,W]
    x = np.expand_dims(x, 0)                                 # [1,C,H,W]
    return x

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)

@app.route("/")
def home():
    return "✅ YOLO ONNX Flask API is running (no torch)"

@app.route("/health")
def health():
    try:
        get_session()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(f.read()))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    x = preprocess(img, IMG_SIZE)
    sess = get_session()

    # ONNX 입출력 이름 추정: YOLO cls export는 보통 input "images"
    feeds = {sess.get_inputs()[0].name: x}
    out = sess.run(None, feeds)[0]          # [1, num_classes]
    probs = out[0]
    # 혹시 로짓이면 softmax 적용
    if probs.max() > 1.0 or probs.min() < 0.0:
        probs = softmax(probs)

    top1_idx = int(np.argmax(probs))
    top1 = CLASS_NAMES[top1_idx] if top1_idx < len(CLASS_NAMES) else str(top1_idx)
    conf = float(probs[top1_idx])

    topk = probs.argsort()[-3:][::-1]
    top3 = [{"class": (CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)),
             "conf": float(probs[i])} for i in topk]

    return jsonify({"top1": top1, "confidence": round(conf, 3), "top3": top3})

@app.route("/demo")
def demo():
    # 간단 업로드 UI
    return """
<!doctype html>
<html lang="ko"><head><meta charset="utf-8"/>
<title>YOLO Classifier Demo</title>
<style>
body{font-family:ui-sans-serif,system-ui;margin:40px auto;max-width:720px}
h1{margin:0 0 12px}
#preview{width:320px;height:320px;object-fit:cover;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1)}
.row{display:flex;gap:12px;align-items:center}
button{padding:10px 16px;border-radius:8px;border:0;background:#2563eb;color:#fff;cursor:pointer}
pre{background:#0b1020;color:#ccf;padding:12px;border-radius:8px;overflow:auto}
</style></head>
<body>
<h1>🧠 Trash Classifier (ONNX)</h1>
<p>이미지를 선택하고 <b>예측하기</b>를 누르세요.</p>
<div class="row">
  <input id="file" type="file" accept="image/*"/>
  <button id="btn">예측하기</button>
</div>
<div style="margin-top:12px"><img id="preview"/></div>
<h3>결과</h3><pre id="out">대기 중...</pre>
<script>
const file = document.getElementById('file');
const prev = document.getElementById('preview');
const out = document.getElementById('out');
document.getElementById('btn').onclick = async () => {
  if(!file.files[0]) return alert('이미지를 선택하세요');
  const fd = new FormData(); fd.append('file', file.files[0]);
  out.textContent = '예측 중...';
  try{
    const r = await fetch('/predict',{method:'POST',body:fd});
    const j = await r.json(); out.textContent = JSON.stringify(j,null,2);
  }catch(e){ out.textContent = '요청 실패: '+e; }
};
file.onchange = ()=>{ if(file.files[0]) prev.src = URL.createObjectURL(file.files[0]); }
</script>
</body></html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
