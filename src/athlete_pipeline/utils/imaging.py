import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from ..config.constants import (
    CACHE_FILE,
    DEFAULT_PORTION,
    FALLBACK,
    IMAGE_EXT,
    NUTRITION,
    PORTIONS,
)


def get_image_date(path):
    try:
        exif = Image.open(path)._getexif()
        if exif:
            for tag in [36867, 36868, 306]:
                value = exif.get(tag)
                if value:
                    return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass

    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime)
    except Exception:
        return None


def get_food_images(pid_dir):
    food_dir = Path(pid_dir) / "food-images"
    if not food_dir.exists():
        return []

    images = [
        {"path": path, "filename": path.name, "dt": get_image_date(path)}
        for path in sorted(food_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXT
    ]
    for image in images:
        image["date"] = image["dt"].date() if image["dt"] else None
    return sorted(images, key=lambda item: item["dt"] or datetime.min)


def classify_image(path, food_clf):
    food_class = "unknown"
    confidence = 0.0

    if food_clf:
        try:
            pred = food_clf(Image.open(path).convert("RGB"))[0]
            food_class = pred["label"].lower().replace(" ", "_").replace("-", "_")
            confidence = round(float(pred["score"]), 3)
        except Exception:
            pass

    kcal, protein, carbs, fat, _fiber = (
        NUTRITION[food_class] if confidence >= 0.20 and food_class in NUTRITION else FALLBACK
    )
    portion = PORTIONS.get(food_class, DEFAULT_PORTION)
    scale = portion / 100

    return {
        "food_class": food_class,
        "confidence": confidence,
        "portion_g": portion,
        "calories_kcal": round(kcal * scale),
        "protein_g": round(protein * scale, 1),
        "carbs_g": round(carbs * scale, 1),
        "fat_g": round(fat * scale, 1),
    }


def load_food_cache(cache_file=CACHE_FILE):
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return {}
    with cache_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_food_cache(cache, cache_file=CACHE_FILE):
    cache_file = Path(cache_file)
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, default=str)
