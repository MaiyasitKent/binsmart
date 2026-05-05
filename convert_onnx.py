import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image

processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0")
model.eval()

class EfficientNetExplicit(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.efficientnet = model.efficientnet
        self.classifier = model.classifier
    
    def forward(self, pixel_values):
        outputs = self.efficientnet(pixel_values)
        # last_hidden_state shape: [batch, 1280, 7, 7]
        x = outputs.last_hidden_state
        # average pool
        x = torch.mean(x, dim=[2, 3])
        # classifier
        x = self.classifier(x)
        return x

wrapped = EfficientNetExplicit(model)
wrapped.eval()

img = Image.open("test.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt")
dummy = inputs["pixel_values"]

with torch.no_grad():
    out = wrapped(dummy)
    print(f"PyTorch output max: {out.max():.4f}")
    print(f"Top label: {out.argmax(-1).item()}")

torch.onnx.export(
    wrapped,
    dummy,
    "model.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    opset_version=18,
    dynamo=False
)
print("Export สำเร็จ!")