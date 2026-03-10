# Qazaq Meme Vision 🤙

Computer vision project that classifies meme poses in real time via webcam. Uses ResNet18 (PyTorch) for image classification and FastAPI for the web interface.

**Classes:** `salam`, `sleep`, `hmm`, `cat-tongue`

---

## Project structure

```
cv-meme-project/
├── dataset/                 # Your images (not in repo — add locally)
│   ├── salam/
│   ├── sleep/
│   ├── hmm/
│   ├── cat-tongue/
│   ├── train/               # Created by split_dataset.py (80%)
│   └── val/                 # Created by split_dataset.py (20%)
├── scripts/
│   ├── split_dataset.py     # 80/20 train/val split
│   ├── train.py             # Model training (ResNet18)
│   ├── predict_one.py       # CLI prediction for one image
│   └── capture_data.py      # Webcam tool to collect dataset
├── backend/
│   ├── main.py              # FastAPI app
│   ├── model_utils.py       # Model loading & inference
│   ├── templates/
│   │   └── index.html      # Web UI
│   ├── static/
│   │   ├── style.css
│   │   ├── app.js
│   │   └── memes/          # Meme images (4 files)
│   └── requirements.txt
├── models/                  # Trained model saved here (after training)
│   └── meme_classifier.pth  # Not in repo — train locally
├── memes/                   # Source meme images
├── requirements.txt        # Root deps (torch, etc.)
└── README.md
```

---

## Quick start

### 1. Setup

```bash
git clone <your-repo-url>
cd cv-meme-project
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### 2. Dataset

Put images into:

```
dataset/salam/
dataset/sleep/
dataset/hmm/
dataset/cat-tongue/
```

Use `scripts/capture_data.py` to capture photos via webcam (requires `opencv-python`):

```bash
python scripts/capture_data.py
# Enter class name, press S to save, ESC to exit
```

### 3. Split dataset

```bash
python scripts/split_dataset.py
```

Creates `dataset/train/` and `dataset/val/` (80% / 20%).

### 4. Train model

```bash
python scripts/train.py
```

- ResNet18 + transfer learning, 15 epochs  
- Best model saved to `models/meme_classifier.pth`

### 5. Run web app

```bash
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 → Turn Camera On → make a pose.

---

## How it works (under the hood)

### Training pipeline

1. **Transfer learning**  
   ResNet18 was pretrained on ImageNet (1.2M images, 1000 classes). Convolutional layers already extract general features (edges, textures). We freeze most of the network and only train the new classifier head.

2. **Architecture change**  
   Original: `Linear(512 → 1000)` for ImageNet. Our setup: `Linear(512 → 4)` for 4 meme classes. All other layers stay the same.

3. **Data flow per batch**  
   - Images 224×224 (ImageNet normalization).  
   - Augmentations for train: RandomHorizontalFlip, RandomRotation(15°), ColorJitter.  
   - Forward pass → logits → CrossEntropyLoss → backward → Adam step.

4. **Checkpointing**  
   After each epoch, the model is saved only if validation accuracy improves. Final file: `models/meme_classifier.pth`.

### Backend (FastAPI)

| Route       | Method | Description                           |
|------------|--------|----------------------------------------|
| `/`        | GET    | Serves HTML page with webcam UI        |
| `/predict` | POST   | Accepts image, returns class + confidence |
| `/static/*`| GET    | CSS, JS, meme images                   |

**Startup:** `load_model()` runs once when the app starts. Model stays in memory.

**`/predict` flow:**
1. Receive image bytes (JPEG/PNG from FormData).
2. `PIL.Image.open(BytesIO(bytes))` → RGB.
3. Resize 224×224, normalize (ImageNet mean/std).
4. `model(tensor)` → logits → softmax → argmax → class index.
5. Return `{ class, confidence, meme_path }` as JSON.

### Frontend (real-time mirror)

- **Polling** — `setInterval` every 1000 ms. Capture current frame from `<video>`, draw to hidden `<canvas>`, `canvas.toBlob()` → FormData → fetch POST `/predict`.
- **Confidence threshold** — If `confidence < 0.75` → show placeholder (“Make a pose!”). If `≥ 0.75` → show meme + class.
- **No overlap** — `isRequestInFlight` flag skips the next tick if the previous request hasn’t completed.

---

## What each file does

| File                | Role |
|---------------------|------|
| `split_dataset.py`  | Splits images 80/20 into train/val |
| `train.py`          | Trains ResNet18, saves best checkpoint |
| `predict_one.py`    | CLI: predict class for one image |
| `capture_data.py`   | Webcam capture for building dataset |
| `backend/main.py`   | FastAPI app, routes, model init |
| `backend/model_utils.py` | Load model, run inference |
| `backend/templates/index.html` | Web UI |
| `backend/static/app.js` | Webcam, polling, result display |

---

## Tech stack

- **PyTorch** — ResNet18
- **FastAPI** — Web API
- **Vanilla JS** — No React/Vue; camera + fetch

---

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Qazaq Meme Vision"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**Ignored by .gitignore:** `venv/`, `dataset/` contents, `models/*.pth`, `__pycache__/`.  
Add your dataset locally and train the model before running the web app.
