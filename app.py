import os
from flask import Flask, request, render_template_string, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

# --- 1. 모델 설정 ---
MODEL_PATH = 'trash_model_functional.keras'  # 모델 파일명
CLASS_NAMES = ['can', 'paper', 'plastic']    # 분류할 클래스 이름

# --- 2. 모델 로드 ---
CUSTOM_OBJECTS = {
    'Concatenate': tf.keras.layers.Concatenate,
    'Add': tf.keras.layers.Add,
    'Average': tf.keras.layers.Average,
    'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
    'concatenate': tf.keras.layers.Concatenate,
}

try:
    model = load_model(MODEL_PATH, compile=False, custom_objects=CUSTOM_OBJECTS)
    print(f"✅ 모델 '{MODEL_PATH}' 로드 완료.")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    model = None

# --- 3. Flask 앱 생성 ---
app = Flask(__name__)

# --- 4. 간단한 HTML 페이지 ---
HTML_TEMPLATE = """
<!doctype html>
<html>
<head><title>쓰레기 분류 AI</title></head>
<body style="font-family: sans-serif; text-align: center; margin-top: 50px;">
<h1>쓰레기 이미지 분류기</h1>
<p>(플라스틱, 캔, 종이 중 하나로 예측)</p>
<form method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="image/*" required>
  <input type="submit" value="분류하기">
</form>
{% if prediction %}
  <h2>예측 결과: {{ prediction }}</h2>
  <p>확신도: {{ confidence | round(2) }}%</p>
{% endif %}
</body>
</html>
"""

# --- 5. 메인 라우트 (웹 + API 겸용) ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    confidence = None

    if model is None:
        return "모델이 로드되지 않았습니다. 서버 관리자에게 문의하세요."

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return render_template_string(HTML_TEMPLATE, prediction="파일을 선택해주세요.")

        try:
            # 이미지 처리
            img = Image.open(file.stream).convert('RGB').resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, 0)
            img_array = img_array / 127.5 - 1.0  # MobileNet 스타일 정규화

            # 예측
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            predicted_class_index = np.argmax(score)
            prediction = CLASS_NAMES[predicted_class_index]
            confidence = float(np.max(score) * 100)

        except Exception as e:
            return f"이미지 예측 중 오류 발생: {e}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, confidence=confidence)

# --- 6. Flask 실행 ---
if __name__ == '__main__':
    # Render에서 자동으로 포트 환경 변수를 전달합니다.
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)