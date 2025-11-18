import io
import os

from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import torch
import torchvision.transforms as transforms

# ------------------------
# 1) 기본 설정
# ------------------------
app = Flask(__name__)

# 모델 파일 이름 (GitHub / Render에 올려둘 .pt 파일 이름)
MODEL_PATH = "trash_model_yolo_cls_best.pt"

# 쓰레기 분류 클래스 이름 (네 모델에 맞게 수정!)
# 예시: 일반, 종이, 플라스틱, 유리, 금속, 음식물
CLASS_NAMES = [
    "general_trash",
    "paper",
    "plastic",
    "glass",
    "metal",
    "food_waste",
]

# ------------------------
# 2) 모델 로드
# ------------------------
def load_model():
    # CPU 환경용
    device = torch.device("cpu")
    model = torch.load(MODEL_PATH, map_location=device)

    # 만약 torch.save(model.state_dict()) 로 저장한 거라면:
    # model = MyModelClass(...)
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.eval()
    return model, device


model, device = load_model()

# ------------------------
# 3) 이미지 전처리 함수
# ------------------------
def transform_image(image_bytes: bytes) -> torch.Tensor:
    """
    업로드된 이미지 바이트를 torch.Tensor로 변환
    (네가 학습할 때 썼던 전처리랑 최대한 비슷해야 함)
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 학습할 때 쓴 사이즈로 맞춰줘
        transforms.ToTensor(),
        # 만약 학습할 때 Normalize 썼으면 여기에 추가
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)  # (1, C, H, W)


# ------------------------
# 4) 예측 함수
# ------------------------
def predict(image_bytes: bytes):
    tensor = transform_image(image_bytes)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        # outputs가 (batch, num_classes) 라고 가정
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probabilities, dim=0)

    predicted_class = CLASS_NAMES[top_idx.item()] if top_idx.item() < len(CLASS_NAMES) else str(top_idx.item())
    confidence = float(top_prob.item())

    return predicted_class, confidence


# ------------------------
# 5) 간단한 웹 페이지 (파일 업로드용)
# ------------------------
HTML_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title>쓰레기 분류 AI 데모</title>
</head>
<body>
    <h1>쓰레기 분류 AI</h1>
    <p>이미지를 업로드하면 어떤 쓰레기인지 분류해줍니다.</p>

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
    # 간단 HTML 렌더 (템플릿 파일 없이 문자열로 처리)
    return render_template_string(HTML_TEMPLATE, result=None)


# ------------------------
# 6) /predict API (폼 + JSON 둘 다 지원)
# ------------------------
@app.route("/predict", methods=["POST"])
def predict_route():
    # 1) 브라우저 폼 업로드 (multipart/form-data)
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "파일이 없습니다."}), 400

        image_bytes = file.read()
        class_name, confidence = predict(image_bytes)

        # 브라우저에서 바로 보는 경우 HTML로 결과 보여주기
        if request.content_type and "multipart/form-data" in request.content_type:
            result = {
                "class_name": class_name,
                "confidence": round(confidence, 4),
            }
            return render_template_string(HTML_TEMPLATE, result=result)

    # 2) JSON + base64 형태로 보낼 수도 있음(선택)
    #   여기선 단순화해서 주석 처리해 둘게.

    else:
        return jsonify({"error": "이미지 파일을 'file' 필드로 보내주세요."}), 400

    # 기본 JSON 반환
    return jsonify({
        "class_name": class_name,
        "confidence": confidence,
    })


# ------------------------
# 7) 로컬 실행용
# ------------------------
if __name__ == "__main__":
    # Render에서는 gunicorn이 이 app을 띄우고,
    # 로컬 테스트 용으로만 사용
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
