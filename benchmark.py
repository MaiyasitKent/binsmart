import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import time
import os

processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
hf_model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0")
LABELS = list(hf_model.config.id2label.values())

img = Image.open("test.jpg").convert("RGB")

# ===== PyTorch Original =====
inputs_pt = processor(images=img, return_tensors="pt")
hf_model.eval()

times = []
for _ in range(10):
    with torch.no_grad():
        hf_model(**inputs_pt)

for _ in range(100):
    start = time.time()
    with torch.no_grad():
        out = hf_model(**inputs_pt)
    times.append((time.time() - start) * 1000)

pt_latency = sum(times) / len(times)
pt_label = LABELS[out.logits.argmax(-1).item()]
print(f"[PyTorch Original]")
print(f"  Latency : {pt_latency:.2f} ms")
print(f"  Size    : {os.path.getsize('models/model_original.pt') / 1e6:.2f} MB")
print(f"  Label   : {pt_label}")
print()

# ===== ONNX =====
inputs_np = processor(images=img, return_tensors="np")
arr = inputs_np["pixel_values"].astype(np.float32)

sess_onnx = ort.InferenceSession("models/model.onnx")

times = []
for _ in range(10):
    sess_onnx.run(None, {"pixel_values": arr})

for _ in range(100):
    start = time.time()
    logits = sess_onnx.run(None, {"pixel_values": arr})[0]
    times.append((time.time() - start) * 1000)

onnx_latency = sum(times) / len(times)
onnx_label = LABELS[np.argmax(logits[0])]
print(f"[ONNX]")
print(f"  Latency : {onnx_latency:.2f} ms")
print(f"  Size    : {os.path.getsize('models/model.onnx') / 1e6:.2f} MB")
print(f"  Label   : {onnx_label}")
print()

# ===== Quantized =====
sess_quant = ort.InferenceSession("models/model_quantized.onnx")

times = []
for _ in range(10):
    sess_quant.run(None, {"pixel_values": arr})

for _ in range(100):
    start = time.time()
    logits_q = sess_quant.run(None, {"pixel_values": arr})[0]
    times.append((time.time() - start) * 1000)

quant_latency = sum(times) / len(times)
quant_label = LABELS[np.argmax(logits_q[0])]
print(f"[Quantized INT8]")
print(f"  Latency : {quant_latency:.2f} ms")
print(f"  Size    : {os.path.getsize('models/model_quantized.onnx') / 1e6:.2f} MB")
print(f"  Label   : {quant_label}")
print()

# ===== Summary =====
print("=" * 45)
print(f"{'Model':<20} {'Latency':>10} {'Size':>10}")
print("=" * 45)
print(f"{'PyTorch Original':<20} {pt_latency:>9.2f}ms {os.path.getsize('models/model_original.pt') / 1e6:>9.2f}MB")
print(f"{'ONNX':<20} {onnx_latency:>9.2f}ms {os.path.getsize('models/model.onnx') / 1e6:>9.2f}MB")
print(f"{'Quantized INT8':<20} {quant_latency:>9.2f}ms {os.path.getsize('models/model_quantized.onnx') / 1e6:>9.2f}MB")
print("=" * 45)