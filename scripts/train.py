"""
train.py - Обучение модели классификации мемов (transfer learning, ResNet18)

Что делает этот скрипт:
- Загружает предобученный ResNet18 из torchvision
- Заменяет последний слой на 4 класса (salam, sleep, hmm, cat-tongue)
- Обучает на dataset/train/, валидирует на dataset/val/
- Выводит loss и accuracy для каждой эпохи
- Сохраняет лучшую модель в models/meme_classifier.pth

Требует: сначала запустить scripts/split_dataset.py

Запуск: python scripts/train.py
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# КОНФИГУРАЦИЯ
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "meme_classifier.pth"

CLASSES = ["salam", "sleep", "hmm", "cat-tongue"]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
# Для воспроизводимости
TORCH_SEED = 42

# Размер изображений для ResNet (стандарт ImageNet)
IMG_SIZE = 224


def set_seed(seed: int):
    """Фиксирует seed для воспроизводимости."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms():
    """
    Аугментации для train (случайные преобразования улучшают обобщение).
    Для val — только resize + нормализация (без аугментаций).
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        # Сильнее по освещению — модель лучше обобщает на разный свет
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4),
        transforms.RandomGrayscale(p=0.1),  # иногда ч/б — меньше зависимость от цвета
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return train_transform, val_transform


def create_model() -> nn.Module:
    """
    Создаёт модель ResNet18 с заменённым классификатором.
    Предобученные веса загружаются автоматически (ImageNet).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Заменяем последний полносвязный слой (было 1000 классов ImageNet → 4 наших)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    """Одна эпоха обучения. Возвращает (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    """Валидация. Возвращает (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def main():
    set_seed(TORCH_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # Проверяем наличие данных
    if not TRAIN_DIR.exists():
        print(f"Ошибка: {TRAIN_DIR} не найден. Запустите scripts/split_dataset.py")
        return

    # Датасеты и загрузчики
    train_transform, val_transform = get_transforms()
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=val_transform)

    # Важно: порядок классов в class_to_idx должен совпадать с CLASSES
    print("Классы:", train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 для Windows без ошибок; на Linux можно 4
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Модель, функция потерь, оптимизатор
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Папка для сохранения модели
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    print(f"\nОбучение: {EPOCHS} эпох, batch_size={BATCH_SIZE}\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        # Сохраняем лучшую модель по val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dict = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "classes": train_dataset.classes,  # порядок из ImageFolder
            }
            torch.save(save_dict, MODEL_SAVE_PATH)
            print(f"  → Сохранена лучшая модель (val_acc={val_acc:.2f}%)")

    print(f"\nОбучение завершено. Модель сохранена: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
