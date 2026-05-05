from fastapi import FastAPI, UploadFile, File, HTTPException
from concurrent.futures import ProcessPoolExecutor
from app.model import run_inference
from app.schemas import PredictionResponse
import asyncio

app = FastAPI(title="BinSmart API")
executor = ProcessPoolExecutor(max_workers=4)

@app.get("/")
def root():
    return {"message": "BinSmart Waste Classification API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg","image/jpg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="ไฟล์ต้องเป็นรูปภาพเท่านั้น")

    contents = await file.read()

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="ไฟล์ใหญ่เกิน 5MB")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_inference, contents)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))