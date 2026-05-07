---
title: BinSmart
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# BinSmart — Waste Classification API

API สำหรับจำแนกประเภทขยะจากรูปภาพด้วย EfficientNet-B0 (ONNX)

## Model

- Model: `google/efficientnet-b0`
- Format: ONNX (exported from PyTorch)
- Classes: 1000 classes (ImageNet)

## Installation

```bash
git clone https://github.com/MaiyasitKent/binsmart.git
cd binsmart
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python convert_onnx.py
mkdir -p models && mv model.onnx models/
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run with Docker

```bash
docker build -t binsmart .
docker run -p 8000:8000 binsmart
```

## API Usage

### Predict

```bash
curl -X POST "https://maiyasitkent-binsmart.hf.space/predict" \
  -F "file=@image.jpg;type=image/jpeg"
```

### Response

```json
{
  "label": "plastic bag",
  "confidence": 6.379
}
```

### Health Check

```bash
curl "https://maiyasitkent-binsmart.hf.space/"
```

## Run Tests

```bash
pytest tests/ -v
```

## CI/CD

GitHub Actions รัน pytest ทุกครั้งที่ push ขึ้น main branch

## Performance

| Model | Latency | Size |
|---|---|---|
| PyTorch Original | 23.82 ms | 21.47 MB |
| ONNX | 8.59 ms | 21.17 MB |
| Quantized INT8 | 86.18 ms | 5.66 MB |

## Load Test Results (JMeter)

| Environment | Avg Latency | Throughput | Error % |
|---|---|---|---|
| Local (uvicorn) | 29 ms | 50.6/sec | 0.00% |
| Local (Docker) | 26 ms | 57.1/sec | 0.00% |
| Cloud (HuggingFace) | 388 ms | 36.9/sec | 0.00% |
