from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import os

quant_pre_process("models/model.onnx", "models/model_preprocessed.onnx")

quantize_dynamic(
    "models/model_preprocessed.onnx",
    "models/model_quantized.onnx",
    weight_type=QuantType.QInt8
)

print(f"Original ONNX size:    {os.path.getsize('models/model.onnx') / 1e6:.2f} MB")
print(f"Quantized size:        {os.path.getsize('models/model_quantized.onnx') / 1e6:.2f} MB")