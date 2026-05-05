from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import time
import os

processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0")
model.eval()

# บันทึกโมเดลลงไฟล์เพื่อวัดขนาด
torch.save(model.state_dict(), "model_original.pt")

img = Image.open("test.jpg")
inputs = processor(images=img, return_tensors="pt")

# วัด latency 100 รอบแล้วเอาค่าเฉลี่ย
times = []
for _ in range(100):
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.time()
    times.append((end - start) * 1000)

avg_latency = sum(times) / len(times)
model_size = os.path.getsize("model_original.pt") / 1e6

print(f"Latency (avg 100 runs): {avg_latency:.2f} ms")
print(f"Model size: {model_size:.2f} MB")
print(f"Prediction: {outputs.logits.argmax(-1)}")