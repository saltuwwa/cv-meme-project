"""
main.py - FastAPI app for Qazaq Meme Vision

Serves the main page, accepts images at /predict,
runs the model and returns prediction + meme path.
"""

from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from model_utils import load_model, predict_image

# ИНИЦИАЛИЗАЦИЯ
app = FastAPI(title="Qazaq Meme Vision")

# Пути к статике и шаблонам
BACKEND_ROOT = Path(__file__).resolve().parent
STATIC_DIR = BACKEND_ROOT / "static"
TEMPLATES_DIR = BACKEND_ROOT / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Загружаем модель при старте (один раз)
model, classes, transform = load_model()


#  ROUTES 

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with webcam and meme mirror interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image, returns class, confidence, meme path.
    Used by frontend for real-time polling.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Expected image (image/jpeg, image/png, ...)")

    try:
        image_bytes = await file.read()
        result = predict_image(model, transform, classes, image_bytes)
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")
