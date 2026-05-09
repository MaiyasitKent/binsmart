import onnxruntime as ort
import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification
import io

hf_model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0")
LABELS = list(hf_model.config.id2label.values())
processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
sess = ort.InferenceSession("models/model.onnx")

def run_inference(image_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("ไฟล์เสียหรือไม่ใช่รูปภาพ")

    inputs = processor(images=img, return_tensors="np")
    arr = inputs["pixel_values"].astype(np.float32)

    logits = sess.run(None, {"pixel_values": arr})[0]
    
    # คำนวณ softmax เพื่อให้ได้ probability
    exp_logits = np.exp(logits[0] - np.max(logits[0]))
    probs = exp_logits / exp_logits.sum()
    
    idx = int(np.argmax(probs))
    confidence_pct = float(probs[idx]) * 100

    return {
        "label": LABELS[idx],
        "confidence": round(float(logits[0][idx]), 4),
        "confidence_percent": f"{confidence_pct:.2f}%"
    }