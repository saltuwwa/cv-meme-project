"""
model_utils.py - Model loading and prediction for FastAPI

Loads trained ResNet18 from models/meme_classifier.pth,
applies prediction to image bytes, returns class and confidence.
"""

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

# Config
BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
MODEL_PATH = PROJECT_ROOT / "models" / "meme_classifier.pth"
IMG_SIZE = 224

# Class name -> meme filename in static/memes/
CLASS_TO_MEME = {
    "salam": "salam-meme.jpg",
    "sleep": "sleeps-meme.jpg",
    "hmm": "hmm-meme.jpg",
    "cat-tongue": "cat_tongue-meme.jpg",
}


def load_model():
    """
    Load model from checkpoint at app startup.
    Returns (model, classes, transform) for predict_image.
    """
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, classes, transform


def predict_image(model, transform, classes, image_bytes: bytes) -> dict:
    """
    Predict class for image bytes.
    Returns: { "class": str, "confidence": float, "meme_path": str }
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(1).item()
        pred_class = classes[pred_idx]
        confidence = probs[0][pred_idx].item()

    meme_filename = CLASS_TO_MEME.get(pred_class, "salam-meme.jpg")
    meme_path = f"/static/memes/{meme_filename}"

    return {
        "class": pred_class,
        "confidence": round(confidence, 4),
        "meme_path": meme_path,
    }
