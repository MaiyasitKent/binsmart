from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io

client = TestClient(app)

def make_image_bytes():
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_returns_json():
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", make_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert "label" in response.json()
    assert "confidence" in response.json()

def test_wrong_file_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400

def test_corrupted_file():
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", b"not_an_image", "image/jpeg")}
    )
    assert response.status_code == 400

def test_file_too_large():
    big_file = b"x" * (6 * 1024 * 1024)
    response = client.post(
        "/predict",
        files={"file": ("big.jpg", big_file, "image/jpeg")}
    )
    assert response.status_code == 400