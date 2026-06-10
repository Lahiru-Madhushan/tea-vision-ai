import os
from pathlib import Path

import cv2
import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _MODEL_DIR / "final_tea_model2.h5"

CLASS_NAMES = [
    "Tea algal leaf spot",
    "Brown Blight",
    "Gray Blight",
    "Helopeltis",
    "Red spider",
    "Green mirid bug",
    "Healthy leaf",
]

_model = None


def get_model_path() -> Path:
    return _MODEL_PATH


def model_exists() -> bool:
    return _MODEL_PATH.exists()


def _load_model():
    global _model

    if _model is not None:
        return _model

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {_MODEL_PATH}. "
            "Place final_tea_model2.h5 in DiseasePrediction/prediction Model2/"
        )

    from tensorflow.keras.models import load_model

    _model = load_model(str(_MODEL_PATH))
    return _model


def _preprocess_image(image_path: str) -> np.ndarray:
    """Match training pipeline: BGR -> RGB, resize 128x128, normalize."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_disease_detailed(image_path: str) -> dict:
    model = _load_model()
    predictions = model.predict(_preprocess_image(image_path), verbose=0)[0]
    predicted_class = int(np.argmax(predictions))
    disease = CLASS_NAMES[predicted_class]
    confidence = float(predictions[predicted_class])
    probabilities = {
        name: float(predictions[i]) for i, name in enumerate(CLASS_NAMES)
    }
    return {
        "disease": disease,
        "confidence": confidence,
        "probabilities": probabilities,
        "model": _MODEL_PATH.name,
    }


def predict_disease(image_path: str):
    result = predict_disease_detailed(image_path)
    print(f"Model: {result['model']}")
    print(f"Predicted Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.2%}")
    return result["disease"], result["confidence"]
