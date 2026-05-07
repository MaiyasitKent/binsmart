---
title: BinSmart
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# BinSmart — Waste Classification API

API สำหรับจำแนกประเภทขยะด้วย EfficientNet-B0

## Usage

```bash
curl -X POST "https://maiyasitkent-binsmart.hf.space/predict" \
  -F "file=@image.jpg;type=image/jpeg"
```
