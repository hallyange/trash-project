from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# ------------------------
# 1) 모델 로드
# ------------------------
MODEL_PATH = "trash_model_yolo_cls_best.pt"  # 레포에 올린 파일 이름

model = YOLO(MODEL_PATH)  # Ultralytics가 내부에서 torch까지 알아서 씀

# YOLO 모델 안에 저장된 클래스 이름 (예: {0: 'paper', 1: 'can', 2: 'plastic'})
CLASS_NAMES_FROM_MODEL = model.names

# 한국어로 보여주고 싶으면 여기서 매핑
DISPLAY_NAME_MAP = {
    "paper": "종이",
    "can": "캔",
    "plastic": "플라스틱",
    "plastic_bottle": "플라스틱",   # 혹시 이런 이름이면 그냥 플라스틱으로 표시
}


# ------------------------
# 2) 예측 함수
# ------------------------
def predict(image_bytes: bytes):
    # 바이트 → PIL 이미지
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # YOLO 분류 실행
    results = model(image)  # 리스트 형태로 결과가 옴
    r = results[0]

    # 최고 확률 클래스
    top_idx = int(r.probs.top1)
    confidence = float(r.probs.top1conf)

    # YOLO 모델 안에 있는 클래스 이름
    class_name_raw = CLASS_NAMES_FROM_MODEL.get(top_idx, str(top_idx))

    # 한국어 표시 이름으로 변환 (없으면 원래 이름)
    display_name = DISPLAY_NAME_MAP.get(class_name_raw, class_name_raw)

    return display_name, confidence


# ------------------------
# 3) HTML 템플릿
# ------------------------
HTML_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title>쓰레기 분류 AI</title>
</head>
<body>
    <h1>쓰레기 분류 AI (YOLO 분류 모델)</h1>
    <p>이미지를 업로드하면 종이 / 캔 / 플라스틱류를 분류해줍니다.</p>

    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">분류하기</button>
    </form>

    {% if result %}
        <h2>결과</h2>
        <p>예측 클래스: {{ result.class_name }}</p>
        <p>확률: {{ result.confidence }}</p>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, result=None)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "이미지 파일을 'file' 필드로 보내주세요."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "파일 이름이 비어 있습니다."}), 400

    image_bytes = file.read()
    class_name, confidence = predict(image_bytes)

    # 폼 업로드면 HTML로 결과 보여주기
    if request.content_type and "multipart/form-data" in request.content_type:
        result = {
            "class_name": class_name,
            "confidence": round(confidence, 4),
        }
        return render_template_string(HTML_TEMPLATE, result=result)

    # 기본 JSON 응답
    return jsonify({
        "class_name": class_name,
        "confidence": confidence,
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
