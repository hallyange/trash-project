import os, io, urllib.request, pathlib
import numpy as np
from PIL import Image
import gradio as gr

# ===== 설정 =====
MODEL_PATH = os.environ.get("MODEL_PATH", "trash_model_yolo_cls_best.pt")  # 리포 루트에 .pt 배치
MODEL_URL  = os.environ.get("MODEL_URL", "")  # (선택) .pt 다운로드 URL (없으면 미사용)
IMG_SIZE   = int(os.environ.get("IMG_SIZE", "224"))

_MODEL = None  # lazy load

def _ensure_model_file():
    """모델 파일 존재/크기 확인 + 필요시 MODEL_URL에서 다운로드"""
    path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    if os.path.isfile(path) and os.path.getsize(path) > 500_000:  # 500KB 이하면 LFS 포인터일 수 있음
        return path

    if not os.path.isfile(path):
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    if MODEL_URL:
        print(f"[INFO] downloading model from {MODEL_URL} -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)

    if not os.path.isfile(path) or os.path.getsize(path) <= 500_000:
        raise FileNotFoundError(
            f"Model not found or too small: {path}. "
            "Put a real .pt in the repo or set MODEL_URL env."
        )
    return path

def _get_model():
    """첫 호출 때만 YOLO 로드(콜드스타트 단축)"""
    global _MODEL
    if _MODEL is None:
        # 여기서 import → 서버 기동은 즉시
        from ultralytics import YOLO
        model_file = _ensure_model_file()
        _MODEL = YOLO(model_file)  # CPU 추론
    return _MODEL

def _preprocess(pil_img: Image.Image, size: int) -> Image.Image:
    # 분류는 PIL 그대로 넣어도 되지만, 리사이즈 맞춰 안정화
    return pil_img.convert("RGB").resize((size, size))

def infer(image: Image.Image):
    """Gradio용 추론 함수: Label 컴포넌트에 확률 dict 반환"""
    if image is None:
        return {}
    model = _get_model()
    img = _preprocess(image, IMG_SIZE)
    r = model.predict(img, imgsz=IMG_SIZE, device="cpu", verbose=False)[0]

    # r.probs.data: (num_classes,)
    probs = r.probs.data.cpu().numpy().astype(float)
    names = r.names  # {idx: name}
    # Label 컴포넌트는 {label: prob} dict를 넣으면 상위 k개 보여줌
    return {names[i]: float(probs[i]) for i in range(len(probs))}

# ===== Gradio UI =====
title = "🧠 Trash Classifier (.pt / Gradio)"
desc = "이미지를 올리면 can/paper/plastic 중 하나로 분류합니다. (Top-3 확률 막대 표시)"

demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="pil", label="이미지 업로드"),
    outputs=gr.Label(num_top_classes=3, label="예측 결과 (Top-3)"),
    title=title,
    description=desc,
    allow_flagging="never",
)

if __name__ == "__main__":
    # Render 포트에 맞춰 바인딩
    port = int(os.environ.get("PORT", 7860))
    demo.queue(api_open=False).launch(server_name="0.0.0.0", server_port=port, show_api=False, share=False, inbrowser=False)