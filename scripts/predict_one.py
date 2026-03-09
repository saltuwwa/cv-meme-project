"""
predict_one.py - Предсказание класса для одного изображения

Что делает этот скрипт:
- Загружает обученную модель из models/meme_classifier.pth
- Применяет те же трансформации, что и при обучении
- Выводит предсказанный класс и вероятности по всем классам

Использование:
  python scripts/predict_one.py path/to/image.jpg
  python scripts/predict_one.py dataset/val/salam/some_meme.png

Запуск после обучения: python scripts/train.py
"""

import sys
from pathlib import Path

import torch
from torchvision import models, transforms

# ============ КОНФИГУРАЦИЯ ============
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "meme_classifier.pth"
IMG_SIZE = 224


def load_model(checkpoint_path: Path, device: torch.device):
    """
    Загружает модель из checkpoint.
    Восстанавливает ResNet18 с правильным числом классов.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    num_classes = len(classes)

    # Создаём модель с той же архитектурой
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def get_inference_transform():
    """Трансформации должны совпадать с val_transform в train.py."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict(image_path: str) -> None:
    """Предсказывает класс для одного изображения."""
    from PIL import Image

    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Ошибка: файл не найден: {img_path}")
        return

    if not MODEL_PATH.exists():
        print(f"Ошибка: модель не найдена. Запустите scripts/train.py")
        print(f"Ожидаемый путь: {MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(MODEL_PATH, device)
    transform = get_inference_transform()

    # Загружаем и препроцессим изображение
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(1).item()
        pred_class = classes[pred_idx]
        pred_prob = probs[0][pred_idx].item()

    print(f"\nИзображение: {img_path.name}")
    print(f"Предсказание: {pred_class} ({pred_prob*100:.1f}%)")
    print("\nВероятности по классам:")
    for i, cls in enumerate(classes):
        p = probs[0][i].item() * 100
        bar = "█" * int(p / 5) + "░" * (20 - int(p / 5))
        print(f"  {cls:12} {p:5.1f}% {bar}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python scripts/predict_one.py <путь_к_изображению>")
        print("Пример: python scripts/predict_one.py dataset/val/salam/meme.jpg")
        sys.exit(1)

    predict(sys.argv[1])
