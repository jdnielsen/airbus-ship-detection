from fastapi import UploadFile
from fastapi.testclient import TestClient
from app.app import app
from PIL import Image
import numpy as np
import io


def test_app_no_file():
    test_file = UploadFile("test")
    
    with TestClient(app) as client:
        response = client.post(
            "/detect_ships/",
            files={"file": (test_file.filename, test_file.file, "")}
        )
        assert response.status_code == 400


def test_app_file():
    fake_img = Image.fromarray(np.zeros((5, 5)))
    fake_img = fake_img.convert("RGB")
    buffer = io.BytesIO()
    fake_img.save(buffer, format="JPEG")
    buffer.seek(0)

    with TestClient(app) as client:
        response = client.post(
            "/detect_ships/",
            files={"file": ("test", buffer, "image/jpeg")}
        )
        assert response.status_code == 200
