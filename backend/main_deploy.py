"""
main_deploy.py — FastAPI entrypoint for production / Render.

Run locally from repo root:
    uvicorn backend.main_deploy:app --host 0.0.0.0 --port 8000

Run on Render:
    uvicorn backend.main_deploy:app --host 0.0.0.0 --port $PORT
"""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .model_utils import load_model, predict_image

BACKEND_ROOT = Path(__file__).resolve().parent
STATIC_DIR = BACKEND_ROOT / "static"
TEMPLATES_DIR = BACKEND_ROOT / "templates"

app = FastAPI(title="Qazaq Meme Vision")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

model, classes, transform = load_model()


@app.get("/health")
async def health():
    """Health check for Render / load balancers."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Deploy UI (instruction card + webcam demo)."""
    return templates.TemplateResponse(request=request, name="index_deploy.html")


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accept image upload (field name \"file\").
    Returns JSON: class, confidence, meme_path.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Expected image (image/jpeg, image/png, ...)")

    try:
        image_bytes = await file.read()
        result = predict_image(model, transform, classes, image_bytes)
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")
