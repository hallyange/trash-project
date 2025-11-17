
   
 import os, io
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image

# ========= 설정 =========
MODEL_PATH = os.environ.get("MODEL_PATH", "trash_model_yolo_cls_best.onnx")  # 리포 루트에 두면 됨
CLASS_NAMES = os.environ.get("CLASS_NAMES", "can,paper,plastic").split(",")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))

app = Flask(__name__)
SESS = None  # lazy-load

def get_session():
    global SESS
    if SESS is None:
        model_file = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model not found: {model_file}")
        SESS = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
    return SESS

def preprocess(img: Image.Image, size=224):
    img = img.convert("RGB").resize((size, size))
    x = np.asarray(img).astype(np.float32) / 255.0   # [H,W,C] 0~1
    x = np.transpose(x, (2, 0, 1))                  # [C,H,W]
    return x[None, ...]                             # [1,C,H,W]

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)

@app.route("/")
def index():
    # 아주 간단한 업로드 페이지
    return """
<!doctype html>
<html lang="ko"><head><meta charset="utf-8"/>
<title>Trash Classifier</title>
<style>
body{font-family:system-ui, -apple-system; max-width:720px; margin:40px auto}
#preview{width:320px;height:320px;object-fit:cover;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.1)}
button{padding:10px 16px;border:0;border-radius:10px;background:#2563eb;color:#fff;cursor:pointer}
pre{background:#0b1020;color:#ccf;padding:12px;border-radius:10px;overflow:auto}
.row{display:flex;gap:12px;align-items:center}
</style></head>
<body>
  <h1>🧠 Trash Classifier</h1>
  <p>이미지를 선택하고 <b>예측하기</b>를 누르세요.</p>
  <div class="row">
    <input id="file" type="file" accept="image/*"/>
    <button id="btn">예측하기</button>
  </div>
  <div style="margin-top:12px"><img id="preview"/></div>
  <h3>결과</h3><pre id="out">대기 중...</pre>
<script>
const f = document.getElementById('file');
const prev = document.getElementById('preview');
const out = document.getElementById('out');
f.onchange = ()=>{ if(f.files[0]) prev.src = URL.createObjectURL(f.files[0]); };
document.getElementById('btn').onclick = async ()=>{
  if(!f.files[0]) return alert('이미지를 선택하세요');
  out.textContent = '예측 중...';
  const fd = new FormData(); fd.append('file', f.files[0]);
  try{
    const r = await fetch('/predict', { method: 'POST', body: fd });
    const j = await r.json();
    out.textContent = JSON.stringify(j, null, 2);
  }catch(e){ out.textContent = '요청 실패: ' + e; }
};
</script>
</body></html>
"""

@app.route("/health")
def health():
    try:
        get_session()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files: return jsonify({"error":"No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "": return jsonify({"error":"Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    x = preprocess(img, IMG_SIZE)
    sess = get_session()
    # 입력 이름 자동 추정 (export 버전에 따라 다를 수 있음)
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: x})[0][0]   # [num_classes]
    # 값이 확률이 아니면 softmax
    probs = out if (0.0 <= out.min() and out.max() <= 1.0) else softmax(out)

    top1_idx = int(np.argmax(probs))
    top1_name = CLASS_NAMES[top1_idx] if top1_idx < len(CLASS_NAMES) else str(top1_idx)
    top1_conf = float(probs[top1_idx])

    topk = probs.argsort()[-3:][::-1]
    top3 = [{"class": (CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)),
             "conf": float(probs[i])} for i in topk]

    return jsonify({"top1": top1_name, "confidence": round(top1_conf, 3), "top3": top3})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render가 지정하는 포트 사용
    app.run(host="0.0.0.0", port=port)