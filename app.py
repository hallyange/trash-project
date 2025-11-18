# app.py  (서버/Render용 - PyTorch 없이 ONNX로 추론)

import io
import os

from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

MODEL_PATH = "trash_model_yolo_cls_best.onnx"

# 네 쓰레기 클래스 이름에 맞게 수정!
CLASS_NAMES = [
    "general_trash",
    "paper",
    "plastic",
    "glass",
    "metal",
    "food_waste",
]

# ------------------------
# 1) ONNX 모델 로드
# ------------------------
ort_session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name


# ------------------------
# 2) 이미지 전처리
# ------------------------
def transform_image(image_bytes: bytes) -> np.ndarray:
    """
    업로드된 이미지를 ONNX 입력용 numpy 배열로 변환
    (학습 때 사용한 전처리와 최대한 맞추기!)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # 학습 크기에 맞게 수정

    img_array = np.array(image).astype("float32") / 255.0  # [0,1] 스케일

    # (H, W, C) -> (1, C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    # 만약 Normalize 했으면 여기서 추가
    # mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    # std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    # img_array = (img_array - mean) / std

    return img_array


# ------------------------
# 3) 예측 함수
# ------------------------
def predict(image_bytes: bytes):
    x = transform_image(image_bytes)

    outputs = ort_session.run(
        [output_name],
        {input_name: x}
    )
    logits = outputs[0][0]        # (num_classes,)
    # numpy softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    class_name = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)

    return class_name, top_prob


# ------------------------
# 4) HTML 템플릿
# ------------------------
HTML_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title>쓰레기 분류 AI (ONNX)</title>
</head>
<body>
    <h1>쓰레기 분류 AI (ONNX)</h1>
    <p>이미지를 업로드하면 어떤 쓰레기인지 분류해줍니다. (서버는 PyTorch 안 씀)</p>

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

    # 브라우저에서 폼으로 보낸 경우 HTML로 결과 표시
    if request.content_type and "multipart/form-data" in request.content_type:
        result = {
            "class_name": class_name,
            "confidence": round(confidence, 4),
        }
        return render_template_string(HTML_TEMPLATE, result=result)

    # 기본 JSON
    return jsonify({
        "class_name": class_name,
        "confidence": confidence,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
