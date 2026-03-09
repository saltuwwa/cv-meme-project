"""
split_dataset.py - Разделение датасета на train и validation (80/20)

Что делает этот скрипт:
- Читает изображения из dataset/<class_name>/
- Случайно разделяет каждую категорию в пропорции 80% train / 20% val
- Создаёт структуру папок dataset/train/ и dataset/val/ с подпапками по классам
- Копирует изображения в соответствующие папки (не перемещает - оригиналы остаются)

Запуск: python scripts/split_dataset.py
"""

import os
import random
import shutil
from pathlib import Path

# ============ КОНФИГУРАЦИЯ ============
# Папка с исходными данными (классы в подпапках)
DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
# Соотношение train/val (0.8 = 80% на обучение, 20% на валидацию)
TRAIN_RATIO = 0.8
# Семя для воспроизводимости результатов
RANDOM_SEED = 42

# Классы (должны совпадать с именами папок в dataset/)
CLASSES = ["salam", "sleep", "hmm", "cat-tongue"]

# Расширения изображений, которые обрабатываем
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_files(class_path: Path) -> list[Path]:
    """Возвращает список путей к изображениям в папке класса."""
    images = []
    for f in class_path.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return images


def split_dataset():
    """Основная функция разделения датасета."""
    random.seed(RANDOM_SEED)

    # Создаём папки train и val
    train_root = DATASET_ROOT / "train"
    val_root = DATASET_ROOT / "val"
    train_root.mkdir(exist_ok=True)
    val_root.mkdir(exist_ok=True)

    total_train = 0
    total_val = 0

    for class_name in CLASSES:
        class_path = DATASET_ROOT / class_name
        if not class_path.exists():
            print(f"  [ПРОПУСК] Папка {class_path} не найдена")
            continue

        images = get_image_files(class_path)
        if not images:
            print(f"  [ПРОПУСК] Нет изображений в {class_path}")
            continue

        # Перемешиваем и разбиваем
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Создаём подпапки для класса
        train_class_dir = train_root / class_name
        val_class_dir = val_root / class_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)

        # Копируем файлы
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)

        total_train += len(train_images)
        total_val += len(val_images)
        print(f"  {class_name}: train={len(train_images)}, val={len(val_images)}")

    print(f"\nИтого: train={total_train}, val={total_val}")
    print(f"Готово! Данные в {train_root} и {val_root}")


if __name__ == "__main__":
    print(f"Разделение датасета: {DATASET_ROOT}")
    print(f"Train/Val: {TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}%\n")
    split_dataset()
